import hydra
from omegaconf import DictConfig, open_dict
from data import (
    get_data,
    get_collators,
)
from data.qa import QADataset
from data.pretraining import PretrainingDataset
from model import get_model
from trainer import load_trainer
from evals import get_evaluator
from trainer.utils import seed_everything
from torch.utils.data import Subset, ConcatDataset
import numpy as np
import os
import logging
import json

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    

     # Print training configuration
    logger.info(f"-----Training Configuration-----")
    if hasattr(cfg.trainer, 'args'):
        trainer_args = cfg.trainer.args
        logger.info(f"Learning Rate: {getattr(trainer_args, 'learning_rate', 'N/A')}")
        logger.info(f"Batch Size: {getattr(trainer_args, 'per_device_train_batch_size', 'N/A')}")
        logger.info(f"Num Epochs: {getattr(trainer_args, 'num_train_epochs', 'N/A')}")
        logger.info(f"Warmup Steps: {getattr(trainer_args, 'warmup_steps', 'N/A')}")
        logger.info(f"Seed: {getattr(trainer_args, 'seed', 'N/A')}")
        logger.info(f"Output Dir: {getattr(trainer_args, 'output_dir', 'N/A')}")
        logger.info(f"Gradient Accumulation Steps: {getattr(trainer_args, 'gradient_accumulation_steps', 'N/A')}")
        logger.info(f"Weight Decay: {getattr(trainer_args, 'weight_decay', 'N/A')}")
        logger.info(f"LR Scheduler Type: {getattr(trainer_args, 'lr_scheduler_type', 'N/A')}")
    logger.info(f"------------------------------")

    # Debugging information
    logger.info(f"Model Configuration: {model_cfg}")
    logger.info(f"Model Args: {model_cfg.model_args}")
    logger.info(f"Use LoRA: {model_cfg.model_args.get('use_lora', 'NOT_FOUND')}")
    logger.info(f"LoRA Configuration: {model_cfg.model_args.get('lora_config', 'NOT_FOUND')}")
    
    model, tokenizer = get_model(model_cfg)

    # PEFT information
    if hasattr(model, "peft_config"):
        adapter_names = list(model.peft_config.keys())
        logger.info(f"Loaded adapters: {adapter_names}")

    data_cfg = cfg.data
    data = get_data(data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args)
    logger.info(f"----------Initializing Data----------")
    logger.info(data)
    
    is_federated = cfg.trainer.handler in [
    "FederatedUnlearningTrainer",
    "FederatedFinetuneTrainer",
    "FederatedPermTrainer",  # ← 新增
]

    if is_federated:
        method_args = cfg.trainer.method_args
        num_clients = method_args.get("num_clients", 3)
        
        raw_sources = method_args.get("source_client_indices", [0])
        if isinstance(raw_sources, str):
            source_client_indices = [int(x.strip()) for x in raw_sources.split(",") if x.strip()]
        elif isinstance(raw_sources, int):
            source_client_indices = [raw_sources]
        else:
            source_client_indices = [int(x) for x in raw_sources]
            
        retain_split_name = method_args.get("retain_split_name", "retain")
        forget_split_name = method_args.get("forget_split_name", "forget")

        def _load_split(split_name, split_kind: str):
            if split_name is None:
                return None
            handler = method_args.get(f"{split_kind}_dataset_handler", "qa")
            hf_args_cfg = method_args.get(f"{split_kind}_dataset_hf_args", {})
            hf_args = {**hf_args_cfg} if hf_args_cfg else {}
            hf_args.setdefault("split", split_name)
            if handler == "qa":
                hf_args.setdefault("name", split_name)
                hf_args.setdefault("path", "locuslab/TOFU")
                # TOFU 数据集的 retain/forget/holdout 均在 train split 下，通过 name 区分
                tofu_like = split_name and (
                    split_name.startswith(("retain", "forget", "holdout"))
                    or split_name == "full"
                )
                if tofu_like and hf_args.get("split") == split_name:
                    hf_args["split"] = "train"
                return QADataset(
                    hf_args=hf_args,
                    template_args=template_args,
                    tokenizer=tokenizer,
                    max_length=model_cfg.data_args.get("max_length", 512)
                    if hasattr(model_cfg, "data_args")
                    else 512,
                    predict_with_generate=getattr(
                        model_cfg, "predict_with_generate", False
                    ),
                )
            if handler == "pretrain":
                hf_args.setdefault("path", "muse-bench/MUSE-News")
                max_length = (
                    model_cfg.data_args.get("max_length", 2048)
                    if hasattr(model_cfg, "data_args")
                    else 2048
                )
                text_key = method_args.get(f"{split_kind}_text_key", "text")
                return PretrainingDataset(
                    hf_args=hf_args,
                    template_args=template_args,
                    tokenizer=tokenizer,
                    text_key=text_key,
                    max_length=max_length,
                )
            raise ValueError(f"Unsupported dataset handler '{handler}' for {split_kind}")

        retain_dataset_obj = _load_split(retain_split_name, "retain")
        forget_dataset_obj = _load_split(forget_split_name, "forget")
        
        if retain_dataset_obj is None or forget_dataset_obj is None:
            raise ValueError("Both retain and forget datasets must be loaded for this partitioning logic.")
            
        data_seed = method_args.get("data_partition_seed", cfg.trainer.args.seed)
        rng = np.random.RandomState(data_seed)
        
        # (1) 数据划分方式：retain随机平分到各个客户端
        retain_len = len(retain_dataset_obj)
        retain_indices = rng.permutation(retain_len)
        retain_splits = np.array_split(retain_indices, num_clients)
        
        federated_train_data = {}
        federated_eval_data = {}
        train_on = method_args.get("train_on", "full")
        
        for client_idx in range(num_clients):
            client_retain_data = Subset(retain_dataset_obj, retain_splits[client_idx].tolist())
            
            if client_idx in source_client_indices and train_on != "retain":
                # 每个源客户端使用全量的 forget 数据
                client_train_data = ConcatDataset([client_retain_data, forget_dataset_obj])
            else:
                client_train_data = client_retain_data
                
            federated_train_data[client_idx] = client_train_data
            
            # (2) 评测时，各个客户端使用各自的本地retain数据集+全量的forget数据集测评
            federated_eval_data[f"client_{client_idx}"] = ConcatDataset([client_retain_data, forget_dataset_obj])
            
        data["train"] = federated_train_data
        data["eval"] = federated_eval_data
        print(f"Prepared federated data for {num_clients} clients. Source clients: {source_client_indices}")

    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)
    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, "Please set trainer"

    # 恢复加载自定义 evaluator 的逻辑，用于生成 TOFU_SUMMARY.json
    evaluator = None
    eval_cfgs = cfg.get("eval", None)
    selected_eval = None
    if cfg.get("hydra") and cfg.hydra.get("runtime") and cfg.hydra.runtime.get("choices"):
        selected_eval = cfg.hydra.runtime.choices.get("eval")
    if selected_eval is None:
        selected_eval = os.environ.get("EVAL_GROUP") or os.environ.get("EVAL_NAME")

    def has_handler(node):
        try:
            return node.get("handler", None) is not None
        except Exception:
            return False

    eval_candidates = {}
    if isinstance(eval_cfgs, (dict, DictConfig)):
        for name, maybe_cfg in eval_cfgs.items():
            if isinstance(maybe_cfg, (dict, DictConfig)) and has_handler(maybe_cfg):
                eval_candidates[name] = maybe_cfg
    if not eval_candidates and isinstance(eval_cfgs, (dict, DictConfig)) and has_handler(eval_cfgs):
        eval_candidates = {"default": eval_cfgs}

    data_train_cfg = cfg.data.get("train", {})
    train_keys = " ".join(list(getattr(data_train_cfg, "keys", lambda: [])()))
    train_keys_lower = train_keys.lower()

    if eval_candidates:
        chosen_eval = None
        if selected_eval and selected_eval in eval_candidates:
            chosen_eval = selected_eval
        elif "muse" in eval_candidates and "muse" in train_keys_lower:
            chosen_eval = "muse"
        elif "tofu" in eval_candidates and "tofu" in train_keys_lower:
            chosen_eval = "tofu"
        elif len(eval_candidates) == 1:
            chosen_eval = next(iter(eval_candidates))

        if chosen_eval is not None:
            evaluator = get_evaluator(
                chosen_eval, eval_candidates[chosen_eval], template_args=template_args, model=model, tokenizer=tokenizer
            )

    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        tokenizer=tokenizer,
        data_collator=collator,
        evaluator=evaluator,
        template_args=template_args,
    )

    if trainer_args.do_train:
        trainer.train()
        trainer.save_state()
        trainer.save_model(trainer_args.output_dir)

    if trainer_args.do_eval:
        trainer.evaluate()


if __name__ == "__main__":
    main()
