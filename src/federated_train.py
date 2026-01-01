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
        split_save_dir = (
            os.path.join(cfg.trainer.args.output_dir, "client_splits")
            if save_client_splits
            else None
        )
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
    if eval_cfgs:
        assert len(eval_cfgs) <= 1, ValueError("Only one evaluation supported while training")
        eval_name, eval_cfg = next(iter(eval_cfgs.items()))
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
