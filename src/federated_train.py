import hydra
from omegaconf import DictConfig
from data import (
    get_data,
    get_collators,
    get_federated_data,
    get_federated_full_retain_forget,
)
from data.qa import QADataset
from data.pretraining import PretrainingDataset
from model import get_model
from trainer import load_trainer
from evals import get_evaluator
from trainer.utils import seed_everything
from torch.utils.data import Subset
import numpy as np
import os
import logging

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
        num_clients = cfg.trainer.method_args.get("num_clients", 3)
        retain_ratio = cfg.trainer.method_args.get("retain_ratio", 0.99)
        save_client_splits = cfg.trainer.method_args.get(
            "save_client_splits_for_eval", False
        )
        assign_forget_to_revoker_only = cfg.trainer.method_args.get(
            "assign_forget_to_revoker_only", False
        )
        revoker_client_idx = cfg.trainer.method_args.get("revoker_client_idx", 0)
        retain_indices_path = cfg.trainer.method_args.get("retain_indices_path", None)
        forget_indices_path = cfg.trainer.method_args.get("forget_indices_path", None)
        global_retain_indices = cfg.trainer.method_args.get("global_retain_indices", None)
        global_forget_indices = cfg.trainer.method_args.get("global_forget_indices", None)
        retain_split_name = cfg.trainer.method_args.get("retain_split_name", None)
        forget_split_name = cfg.trainer.method_args.get("forget_split_name", None)
        split_save_dir = None
        if save_client_splits:
            split_save_dir = os.environ.get("FEDPERM_SPLIT_DIR", None)
            if not split_save_dir:
                split_save_dir = os.path.join(cfg.trainer.args.output_dir, "client_splits")
        if split_save_dir:
            os.makedirs(split_save_dir, exist_ok=True)

        def _load_split(split_name, split_kind: str):
            if split_name is None:
                return None
            method_args = cfg.trainer.method_args
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
        if mode == "unlearn":
            if "forget" not in data or "retain" not in data:
                raise ValueError("Both forget and retain data must be in data dictionary")
            federated_data = get_federated_data(
                data,
                num_clients=num_clients,
                target_client_idx=cfg.trainer.get("target_client_idx", 0),
            )

        else:  # mode == "train" for federated fine-tuning / FedPerm
            if "train" not in data:
                raise ValueError("Training data must be in data dictionary")
            if retain_dataset_obj is None or forget_dataset_obj is None:
                raise ValueError("retain_split_name and forget_split_name must be set for federated training")
            train_data = data["train"]
            federated_data = get_federated_full_retain_forget(
                full_dataset=train_data,
                num_clients=num_clients,
                retain_ratio=retain_ratio,
                seed=cfg.trainer.args.seed,
                assign_forget_to_revoker_only=assign_forget_to_revoker_only,
                revoker_client_idx=revoker_client_idx,
                split_save_dir=split_save_dir,
                retain_indices_path=retain_indices_path,
                forget_indices_path=forget_indices_path,
                global_retain_indices=global_retain_indices,
                global_forget_indices=global_forget_indices,
                retain_dataset=retain_dataset_obj,
                forget_dataset=forget_dataset_obj,
            )
        
        data["train"] = federated_data
        print(f"Prepared federated data for {num_clients} clients")

    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)
    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, "Please set trainer"

    evaluator = None
    eval_cfgs = cfg.get("eval", None)
    selected_eval = None
    if cfg.get("hydra") and cfg.hydra.get("runtime") and cfg.hydra.runtime.get("choices"):
        selected_eval = cfg.hydra.runtime.choices.get("eval")
    if selected_eval is None:
        # Fallback for shell wrappers that pin eval via env (e.g., EVAL_GROUP=muse).
        selected_eval = os.environ.get("EVAL_GROUP") or os.environ.get("EVAL_NAME")

    # Normalise eval configs to a {name: cfg} mapping; some configs package under eval.<name>,
    # while others inject directly. We only treat entries with a handler as evaluators.
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

    # Heuristic: align eval choice with dataset family when not explicitly provided.
    data_train_cfg = {}
    try:
        data_train_cfg = cfg.data.get("train", {})
    except Exception:
        data_train_cfg = {}
    train_keys = " ".join(list(getattr(data_train_cfg, "keys", lambda: [])()))
    train_keys_lower = train_keys.lower()

    if eval_candidates:
        chosen_eval = None
        if selected_eval:
            if selected_eval in eval_candidates:
                chosen_eval = selected_eval
            else:
                raise ValueError(
                    f"Eval choice '{selected_eval}' not found in available evaluators {list(eval_candidates.keys())}. "
                    "请通过 eval=<name> 或环境变量 EVAL_GROUP/EVAL_NAME 选择正确的评测配置。"
                )
        elif "muse" in eval_candidates and "muse" in train_keys_lower:
            chosen_eval = "muse"
        elif "tofu" in eval_candidates and "tofu" in train_keys_lower:
            chosen_eval = "tofu"
        elif len(eval_candidates) == 1:
            chosen_eval = next(iter(eval_candidates))

        if chosen_eval is None:
            raise ValueError(
                f"检测到多个评测配置 {list(eval_candidates.keys())}，未能确定应使用哪一个。"
                " 请显式指定 eval=<name>（Hydra 覆盖）或设置环境变量 EVAL_GROUP/EVAL_NAME。"
            )

        eval_name, eval_cfg = chosen_eval, eval_candidates[chosen_eval]
        evaluator = get_evaluator(
            eval_name,
            eval_cfg,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
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

    # trainer_args.do_eval = False

    # if trainer_args.do_eval:
    #     trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()
