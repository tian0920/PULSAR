import torch
import logging
import shutil
from copy import deepcopy
from typing import Dict, List, Any, Optional
import os
from trainer.base import FinetuneTrainer
from trainer.unlearn.base import UnlearnTrainer
from data.unlearn import ForgetRetainDataset
from torch.utils.data import Dataset
from trainer.federated.federated_utils import *
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from transformers.trainer_utils import IntervalStrategy, SaveStrategy


logger = logging.getLogger(__name__)

class FederatedUnlearningTrainer(FinetuneTrainer):

    
    def __init__(
        self,
        model,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        data_collator=None,
        args=None,
        evaluator=None,
        template_args=None,
        num_clients=3,
        target_client_idx=0,
        unlearn_trainer_cls=None,
        aggregation_strategy="average",
        global_rounds=1,
        unlearn_epochs=None,
        **kwargs
    ):
        # Pop FedPerm 专用的数据划分开关，避免透传到 HF Trainer 造成未知参数报错
        self.federated_dataset = train_dataset if isinstance(train_dataset, dict) else None
        dummy_train_dataset = None
        if self.federated_dataset and 0 in self.federated_dataset:
            dummy_train_dataset = self.federated_dataset[0].retain
            # dummy_train_dataset = self.federated_dataset[0].get("retain")
        
        super().__init__(
            model=model,
            train_dataset=dummy_train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            args=args,
            evaluator=evaluator,
            template_args=template_args,
        )
        
        # logger.info(f"Training args: {self.args}")
        self.num_clients = num_clients
        self.target_client_idx = target_client_idx
        self.unlearn_trainer_cls = unlearn_trainer_cls
        self.aggregation_strategy = aggregation_strategy
        self.global_rounds = global_rounds
        self.unlearn_epochs = unlearn_epochs  
        self.client_models = []
        self.client_data_sizes = [] 
        self.kwargs = kwargs
        self.enable_pre_eval = os.environ.get("ENABLE_PRE_EVAL", "false").lower() in (
            "1",
            "true",
            "yes",
        )
        
  
        self.is_peft = isinstance(self.model, PeftModel)
        self.args.save_strategy = self._normalize_save_strategy(getattr(self.args, "save_strategy", None))

        self.server_momentum = None  
        self.server_velocity = None  
        
        
        self.fedadam_momentum = None  
        self.fedadam_velocity = None   

        self.fedyogi_momentum = None  
        self.fedyogi_velocity = None   
        

        self.fedavgm_momentum = None   
        
 
        self.fedadagrad_velocity = None  
        

        self.federated_args = {
            'server_lr': kwargs.get('server_lr', 1.0),
            'beta1': kwargs.get('beta1', 0.9),
            'beta2': kwargs.get('beta2', 0.99),
            'epsilon': kwargs.get('epsilon', 1e-3),
            'tau': kwargs.get('tau', 1e-3),
            'mu': kwargs.get('mu', 0.01),
            'momentum_factor': kwargs.get('momentum_factor', 0.9),
            # 目标客户端在保留训练阶段的 proximal 约束系数，约束其靠近遗忘后参数（theta_unlearn）
            'prox_to_unlearn_mu': kwargs.get('prox_to_unlearn_mu', 0.01),
        }
        self.init_all_clients_from_unlearn = kwargs.get(
            "init_all_clients_from_unlearn", False
        )
        self.theta_unlearn_state = None  # 保存遗忘后参数，用于目标客户端的 prox 约束
        
        if not self.federated_dataset:
            logger.warning("No federated dataset provided!")
        else:
            for client_idx in range(num_clients):
                if client_idx not in self.federated_dataset:
                    logger.warning(f"Client {client_idx} has no data!")
        
        logger.info(f"Initialized with {num_clients} clients, target: {target_client_idx}, global rounds: {global_rounds}")
        logger.info(f"Model type - PEFT: {self.is_peft}")


    def train(self, resume_from_checkpoint=None, trial=None, **kwargs):
        if not self.federated_dataset:
            raise ValueError("Federated dataset is not properly set up")
        
        logger.info(f"Starting federated unlearning training with {self.global_rounds} global rounds")
        
        # Ensure model is in training mode before copying
        self.model.train()
        self.model = self.model.cpu()
        
        # For PEFT models, ensure adapter parameters are trainable before copying
        from peft import PeftModel
        if isinstance(self.model, PeftModel):
            # Enable gradients for all adapter parameters
            for name, param in self.model.named_parameters():
                if 'lora' in name.lower() or 'adapter' in name.lower():
                    param.requires_grad = True
            logger.info("Enabled gradients for PEFT adapter parameters before copying")
        
        # Create a copy for unlearning (preserve training state)
        unlearn_model = deepcopy(self.model)
        # Ensure the copied model is also in training mode
        unlearn_model.train()
        
        # After deepcopy, re-enable gradients for PEFT models
        if isinstance(unlearn_model, PeftModel):
            for name, param in unlearn_model.named_parameters():
                if 'lora' in name.lower() or 'adapter' in name.lower():
                    param.requires_grad = True
            logger.info("Re-enabled gradients for PEFT adapter parameters after copying")
        
        logger.info(f"Step 1: Unlearning on target client {self.target_client_idx}")
        unlearn_trainer = None
        if self.target_client_idx in self.federated_dataset:
            target_dataset = self.federated_dataset[self.target_client_idx]
            unlearn_trainer = self.unlearn_client_model(unlearn_model, target_dataset)
            logger.info("Completed unlearning on target client")
            if self.init_all_clients_from_unlearn:
                self._apply_theta_unlearn_state_to_global()
            # 在目标客户端保留训练前保存“纯遗忘后”模型，便于单独评测遗忘效果
            pre_retain_target_path = f"{self.args.output_dir}/pre_retain_target_client_{self.target_client_idx}"
            os.makedirs(pre_retain_target_path, exist_ok=True)
            tgt_model = unlearn_trainer.model.cpu()
            if self.is_peft:
                tgt_model.save_pretrained(pre_retain_target_path, safe_serialization=False)
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(pre_retain_target_path, safe_serialization=False)
            else:
                torch.save(
                    tgt_model.state_dict(),
                    os.path.join(pre_retain_target_path, "pytorch_model.bin"),
                )
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(pre_retain_target_path, safe_serialization=False)
            logger.info(
                f"Saved pre-retain target client {self.target_client_idx} model at {pre_retain_target_path}"
            )
        else:
            logger.warning(f"Target client {self.target_client_idx} has no data, skipping unlearning")
    
        for round_idx in range(self.global_rounds):
            logger.info(f"Starting global round {round_idx + 1}/{self.global_rounds}")
            
    
            logger.info("Step 2: All clients training on retain data with updated global model")

            # Server-side eval before broadcasting (聚合前评测)
            if self.enable_pre_eval:
                self._maybe_evaluate_model(
                    model=self.model,
                    round_idx=round_idx,
                    stage="server_pre_agg",
                )
            
         
            self.client_models = [deepcopy(self.model) for _ in range(self.num_clients)]
            self.client_data_sizes = [] 

           
            logger.info(f"Aggregation strategy: {self.aggregation_strategy}")


            for client_idx in range(self.num_clients):
                if client_idx not in self.federated_dataset:
                    logger.warning(f"Skipping client {client_idx} as no data is available")
                    self.client_data_sizes.append(0)
                    continue
                    
                client_model = self.client_models[client_idx]
                client_dataset = self.federated_dataset[client_idx]
                
                retain_data = self._resolve_retain_data(client_dataset.retain)
                client_data_size = len(retain_data)
                self.client_data_sizes.append(client_data_size)
                
                logger.info(f"Training client {client_idx} on retain data with {self.args.num_train_epochs} epochs")
                if self.enable_pre_eval:
                    stage_name = f"client{client_idx}_pre_train"
                    if round_idx == 0:
                        reused = self._reuse_server_pre_eval(round_idx, stage_name)
                    else:
                        reused = self._reuse_server_post_eval(round_idx, stage_name)
                    if not reused:
                        self._maybe_evaluate_model(
                            model=client_model,
                            round_idx=round_idx,
                            stage=stage_name,
                        )
                self.train_client_model(client_model, client_dataset, client_idx)
                
                self.client_models[client_idx] = client_model.cpu()

            for client_idx in range(self.num_clients):
                trained_client_model = (
                    self.client_models[client_idx]
                    if client_idx < len(self.client_models)
                    else None
                )
                if trained_client_model is None:
                    continue
                self._maybe_evaluate_model(
                    model=trained_client_model,
                    round_idx=round_idx,
                    stage=f"client{client_idx}_post_train",
                )

            logger.info("Step 3: Aggregating all client models")
            self.aggregate_models(round_idx)
            self._maybe_evaluate_model(
                model=self.model,
                round_idx=round_idx,
                stage="server_post_agg",
            )
            logger.info(f"Completed global round {round_idx + 1}/{self.global_rounds}")
            
            strategy = self._normalize_save_strategy(getattr(self.args, "save_strategy", None))
            is_epoch = strategy == SaveStrategy.EPOCH
            if is_epoch or (round_idx == self.global_rounds - 1):
                save_path = f"{self.args.output_dir}/round_{round_idx + 1}"
                self.model = self.model.cpu()
                self.save_model(save_path)
                logger.info(f"Model saved at {save_path}")
        
        # 保存最终的全局模型与目标客户端模型，便于后续测评
        # 不再额外生成 final_* 目录，依赖 round_N 检查点
        return None

    @staticmethod
    def _resolve_retain_data(retain_dataset):
        """Return underlying retain split when present; fallback to the dataset itself."""
        if hasattr(retain_dataset, "retain") and getattr(retain_dataset, "retain") is not None:
            return retain_dataset.retain
        return retain_dataset

    def _apply_theta_unlearn_state_to_global(self):
        """Initialize global model from theta_unlearn_state when available."""
        if self.theta_unlearn_state is None:
            return
        if self.is_peft:
            set_peft_model_state_dict(self.model, self.theta_unlearn_state)
        else:
            self.model.load_state_dict(self.theta_unlearn_state, strict=False)
        logger.info("Initialized global model from theta_unlearn_state")

    @staticmethod
    def _normalize_save_strategy(strategy):
        """Accept SaveStrategy/IntervalStrategy/str; default to EPOCH when None."""
        if isinstance(strategy, SaveStrategy):
            return strategy
        if isinstance(strategy, IntervalStrategy):
            try:
                return SaveStrategy(strategy.value)
            except Exception:
                return SaveStrategy.EPOCH
        if isinstance(strategy, str):
            try:
                return SaveStrategy(strategy.lower())
            except Exception:
                return SaveStrategy.EPOCH
        if strategy is None:
            return SaveStrategy.EPOCH
        return SaveStrategy.EPOCH

    def _maybe_evaluate_model(self, model, round_idx: int, stage: str):
        """
        Run evaluator (if provided) and save SUMMARY.json for the given stage.
        """
        if self.evaluator is None:
            return
        if (stage.startswith("server_pre") or stage.endswith("_pre_train")) and not self.enable_pre_eval:
            logger.info(
                f"Skipping pre eval for stage {stage} due to ENABLE_PRE_EVAL={self.enable_pre_eval}"
            )
            return
        eval_dir = os.path.join(
            self.args.output_dir,
            "eval_rounds",
            f"round_{round_idx + 1}",
            stage,
        )
        os.makedirs(eval_dir, exist_ok=True)

        summary_file = os.path.join(eval_dir, "TOFU_SUMMARY.json")
        summary_exists = os.path.isfile(summary_file) and os.path.getsize(summary_file) > 0

        force_eval = os.environ.get("FORCE_FEDPERM_EVAL", "0") in ("1", "true", "True")
        if summary_exists and not force_eval:
            logger.info(
                f"Skipping eval for stage {stage} at round {round_idx + 1}, existing summary: {summary_file}"
            )
            return

        model_was_on_cpu = next(model.parameters()).device.type == "cpu"
        if torch.cuda.is_available() and model_was_on_cpu:
            device = torch.device("cuda")
            model = model.to(device)
            logger.info(f"Model moved to GPU device {device} for evaluation at stage {stage}")

        overwrite = True
        if hasattr(self.evaluator, "eval_cfg"):
            overwrite = getattr(self.evaluator.eval_cfg, "overwrite", True)

        try:
            self.evaluator.evaluate(
                model=model,
                tokenizer=self.tokenizer,
                template_args=self.template_args,
                output_dir=eval_dir,
                overwrite=overwrite,
            )
        except Exception as exc:
            logger.warning(f"Evaluation failed at {stage} (round {round_idx}): {exc}")
        finally:
            if torch.cuda.is_available() and model_was_on_cpu:
                model = model.cpu()
                torch.cuda.empty_cache()

    def _reuse_server_post_eval(self, round_idx: int, target_stage: str) -> bool:
        """
        Reuse previous round server_post_agg eval as current pre_train eval.
        """
        if self.evaluator is None or round_idx == 0:
            return False
        prev_dir = os.path.join(
            self.args.output_dir,
            "eval_rounds",
            f"round_{round_idx}",
            "server_post_agg",
        )
        src_summary = os.path.join(prev_dir, "TOFU_SUMMARY.json")
        if not os.path.isfile(src_summary):
            return False

        dest_dir = os.path.join(
            self.args.output_dir,
            "eval_rounds",
            f"round_{round_idx + 1}",
            target_stage,
        )
        os.makedirs(dest_dir, exist_ok=True)

        copied = False
        for fname in os.listdir(prev_dir):
            src = os.path.join(prev_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dest_dir, fname))
                copied = True
        if copied:
            logger.info(
                f"Reused server_post_agg eval (round {round_idx}) for stage {target_stage}"
            )
        return copied

    def _reuse_server_pre_eval(self, round_idx: int, target_stage: str) -> bool:
        """
        Reuse current round server_pre_agg eval as client pre_train eval (first round).
        """
        if self.evaluator is None:
            return False
        src_dir = os.path.join(
            self.args.output_dir,
            "eval_rounds",
            f"round_{round_idx + 1}",
            "server_pre_agg",
        )
        src_summary = os.path.join(src_dir, "TOFU_SUMMARY.json")
        if not os.path.isfile(src_summary):
            return False

        dest_dir = os.path.join(
            self.args.output_dir,
            "eval_rounds",
            f"round_{round_idx + 1}",
            target_stage,
        )
        os.makedirs(dest_dir, exist_ok=True)

        copied = False
        for fname in os.listdir(src_dir):
            src = os.path.join(src_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dest_dir, fname))
                copied = True
        if copied:
            logger.info(
                f"Reused server_pre_agg eval (round {round_idx + 1}) for stage {target_stage}"
            )
        return copied

    @staticmethod
    def _filter_unlearn_trainer_kwargs(kwargs):
        """Drop unsupported args before constructing HF Trainer subclasses."""
        if not kwargs:
            return {}
        filtered = dict(kwargs)
        filtered.pop("retain_split_name", None)
        filtered.pop("forget_split_name", None)
        filtered.pop("save_client_splits_for_eval", None)
        filtered.pop("retain_indices_path", None)
        filtered.pop("forget_indices_path", None)
        filtered.pop("assign_forget_to_revoker_only", None)
        filtered.pop("revoker_client_idx", None)
        filtered.pop("init_all_clients_from_unlearn", None)
        return filtered

    def unlearn_client_model(self, client_model, client_dataset):
        from trainer import TRAINER_REGISTRY
        trainer_cls = TRAINER_REGISTRY.get(self.unlearn_trainer_cls, None)


        forget_dataset = client_dataset.forget
        retain_dataset = client_dataset.retain

        # 展开可能的嵌套 ForgetRetainDataset，拿到底层真实数据集
        from data.unlearn import ForgetRetainDataset
        if isinstance(forget_dataset, ForgetRetainDataset) and getattr(
            forget_dataset, "forget", None
        ) is not None:
            forget_dataset = forget_dataset.forget
        if isinstance(retain_dataset, ForgetRetainDataset) and getattr(
            retain_dataset, "retain", None
        ) is not None:
            retain_dataset = retain_dataset.retain
        retain_data = self._resolve_retain_data(retain_dataset)

        logger.info(f"forget_data length: {len(forget_dataset)}")
        logger.info(f"retain_data length: {len(retain_data)}")
        logger.info(f"Unlearning_cls_name: {self.unlearn_trainer_cls}")
        
        class CombinedDataset(Dataset):
            def __init__(self, forget_data, retain_data):
                self.forget_data = forget_data
                self.retain_data = retain_data
                self.retain_length = len(retain_data)

            def __len__(self):
                return len(self.forget_data)

            def __getitem__(self, idx):
                retain_idx = idx % self.retain_length
                forget_item = self.forget_data[idx]
                retain_item = self.retain_data[retain_idx]
                if not (isinstance(forget_item, dict) and "forget" in forget_item):
                    forget_item = {"forget": forget_item}
                if not (isinstance(retain_item, dict) and "retain" in retain_item):
                    retain_item = {"retain": retain_item}
                return {
                    "forget": forget_item,
                    "retain": retain_item,
                }
        
        combined_dataset = CombinedDataset(forget_dataset, retain_data)

        # Ensure model is in training mode and parameters are trainable.
        # Some PEFT checkpoints arrive with all params frozen, which would make loss.detach() with no grad_fn.
        client_model.train()
        client_model.requires_grad_(True)
        from peft import PeftModel
        if isinstance(client_model, PeftModel):
            for name, param in client_model.named_parameters():
                if ('lora' in name.lower() or 'adapter' in name.lower()):
                    param.requires_grad = True
            logger.info("Enabled gradients for PEFT adapter parameters")
        # Verify that at least some parameters have gradients enabled
        trainable_params = sum(p.numel() for p in client_model.parameters() if p.requires_grad)
        if trainable_params == 0:
            logger.warning("No trainable parameters found! Enabling gradients for all parameters.")
            for param in client_model.parameters():
                param.requires_grad = True
        else:
            logger.info(f"Found {trainable_params} trainable parameters")

        unlearn_args = deepcopy(self.args)
        if self.unlearn_epochs is not None:
            unlearn_args.num_train_epochs = self.unlearn_epochs
            logger.info(f"Using {self.unlearn_epochs} epochs for unlearn stage")
        else:
            logger.info(f"Using default {unlearn_args.num_train_epochs} epochs for unlearn stage")

        trainer_kwargs = self._filter_unlearn_trainer_kwargs(self.kwargs)
        if self.aggregation_strategy == "FedProx":
            class FedProxUnlearnTrainer(trainer_cls):
                def __init__(self, *args, global_model=None, mu=0.01, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.global_model = global_model.to(self.args.device) if global_model is not None else None
                    self.mu = mu

                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    return_values = super().compute_loss(model, inputs, return_outputs=return_outputs)
                    
                    if return_outputs:
                        loss, outputs = return_values
                    else:
                        loss = return_values

                    proximal_term = 0.0
                    if self.global_model is not None:
                        global_state = get_peft_model_state_dict(self.global_model)
                        for name, w in model.named_parameters():
                            if not w.requires_grad:
                                continue
                            name = name.replace(".default", "") 
                            if name not in global_state:
                                logger.warning(f"Parameter {name} not found in global_state, skipping")
                                continue
                            proximal_term += torch.norm(w - global_state[name].to(w.device).detach()) ** 2
                        loss += (self.mu / 2) * proximal_term
                    
                    return (loss, outputs) if return_outputs else loss
            
            unlearn_trainer = FedProxUnlearnTrainer(
                model=client_model,
                train_dataset=combined_dataset,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                args=unlearn_args, 
                evaluator=self.evaluator,
                template_args=self.template_args,
                global_model=self.model.cpu(),
                mu=self.federated_args['mu'],
                **trainer_kwargs
            )
        else:
            unlearn_trainer = trainer_cls(
                model=client_model,
                train_dataset=combined_dataset,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                args=unlearn_args, 
                evaluator=self.evaluator,
                template_args=self.template_args,
                **trainer_kwargs
            )
        
        # After Trainer initialization, ensure model is in training mode and parameters are trainable
        unlearn_trainer.model.train()
        unlearn_trainer.model.requires_grad_(True)
        trainable_count = sum(1 for p in unlearn_trainer.model.parameters() if p.requires_grad)
        if trainable_count == 0:
            logger.warning("No trainable parameters after Trainer initialization! Enabling all parameters.")
            for param in unlearn_trainer.model.parameters():
                param.requires_grad = True
        else:
            logger.info(f"Model has {trainable_count} trainable parameters after Trainer initialization")

        unlearn_trainer.train()

        # 记录遗忘后参数
        if self.is_peft:
            self.theta_unlearn_state = get_peft_model_state_dict(unlearn_trainer.model)
        else:
            self.theta_unlearn_state = unlearn_trainer.model.state_dict()
        
        return unlearn_trainer
    

    def train_client_model(self, client_model, client_dataset, client_idx):
        retain_dataset = client_dataset.retain
        retain_data = self._resolve_retain_data(retain_dataset)
        
        # Move client model to GPU if available (Trainer will handle device placement, but we ensure it's on GPU)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            client_model = client_model.to(device)
            logger.info(f"Client model moved to GPU device: {device}")
        
        if self.aggregation_strategy == "FedProx":
            class FedProxTrainer(FinetuneTrainer):
                def __init__(self, *args, global_model=None, mu=0.01, **kwargs):
                    super(FedProxTrainer, self).__init__(*args, **kwargs)
                    self.global_model = global_model
                    self.mu = mu
                    
                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    return_values = super().compute_loss(model, inputs, return_outputs=return_outputs)
                    
                    if return_outputs:
                        loss, outputs = return_values
                    else:
                        loss = return_values
                    proximal_term = 0.0
                    if self.global_model is not None:
                        global_state = get_peft_model_state_dict(self.global_model)
                        for name, w in model.named_parameters():
                            if not w.requires_grad:
                                continue
                            name = name.replace(".default", "") 
                            if name not in global_state:
                                logger.warning(f"Parameter {name} not found in global_state, skipping")
                                continue
                            proximal_term += torch.norm(w - global_state[name].to(w.device).detach()) ** 2
                        loss += (self.mu / 2) * proximal_term                    
                    return (loss, outputs) if return_outputs else loss
                
            trainer = FedProxTrainer(
                model=client_model,
                train_dataset=retain_data,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                args=self.args,
                evaluator=self.evaluator,
                template_args=self.template_args,
                global_model=self.model.cpu(),
                mu=self.federated_args['mu']
            )
        else:
            trainer = FinetuneTrainer(
                model=client_model,
                train_dataset=retain_data,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                args=self.args,
                evaluator=self.evaluator,
                template_args=self.template_args,
            )
        
        trainer.train()
        return trainer
    

    def aggregate_models(self, round_idx=0):
        logger.info("Aggregating client models")
        

        total_samples = sum(self.client_data_sizes)
        if total_samples == 0:
            logger.warning("No samples available for aggregation")
            return
            
        client_weights = [size / total_samples for size in self.client_data_sizes]
        logger.info(f"Client sample sizes: {self.client_data_sizes}")
        logger.info(f"Client weights: {[f'{w:.4f}' for w in client_weights]}")

        # Determine target device for aggregation (prefer GPU if available)
        use_gpu_for_agg = torch.cuda.is_available()
        agg_device = torch.device("cuda") if use_gpu_for_agg else torch.device("cpu")
        
        if self.is_peft:
            client_state_dicts = []
            for idx, model in enumerate(self.client_models):
                # Extract state dict without moving to CPU first
                peft_state_dict = get_peft_model_state_dict(model)
                # Move to aggregation device if needed
                if use_gpu_for_agg:
                    peft_state_dict = {k: v.to(agg_device) if isinstance(v, torch.Tensor) else v 
                                      for k, v in peft_state_dict.items()}
                client_state_dicts.append(peft_state_dict)
                logger.debug(f"Client {model} has PEFT state dict")
        else:

            client_state_dicts = []
            for idx, model in enumerate(self.client_models):
                # Extract state dict without moving to CPU first
                state = model.state_dict()
                # Move to aggregation device if needed
                if use_gpu_for_agg:
                    state = {k: v.to(agg_device) if isinstance(v, torch.Tensor) else v 
                            for k, v in state.items()}
                client_state_dicts.append(state)

        logger.info(f"Aggregation_strategy:{self.aggregation_strategy}, device: {agg_device}")
        
        # Extract global model state dict and move to aggregation device
        if self.is_peft:
            global_peft_state_dict = get_peft_model_state_dict(self.model)
            if use_gpu_for_agg:
                global_peft_state_dict = {k: v.to(agg_device) if isinstance(v, torch.Tensor) else v 
                                         for k, v in global_peft_state_dict.items()}
        else:
            global_peft_state_dict = self.model.state_dict()
            if use_gpu_for_agg:
                global_peft_state_dict = {k: v.to(agg_device) if isinstance(v, torch.Tensor) else v 
                                         for k, v in global_peft_state_dict.items()}

        if self.aggregation_strategy == "FedAvg":
            global_state_dict = FedAvg(
                client_state_dicts, 
                global_model_state_dict=global_peft_state_dict,
                client_weights=client_weights
            )
        elif self.aggregation_strategy == "FedAvgM":
            global_state_dict, self.fedavgm_momentum = FedAvgM(
                client_state_dicts,
                global_model_state_dict=global_peft_state_dict,
                proxy_dict=self.fedavgm_momentum,
                round_idx=round_idx,
                momentum_factor=self.federated_args['momentum_factor'],
                tau=self.federated_args['tau'],
                client_weights=client_weights
            )
        elif self.aggregation_strategy == "FedAdagrad":
            global_state_dict, self.fedadagrad_velocity = FedAdagrad(
                client_state_dicts,
                global_model_state_dict=global_peft_state_dict,
                server_velocity=self.fedadagrad_velocity,
                epsilon=self.federated_args['epsilon'],
                tau=self.federated_args['tau'],
                client_weights=client_weights
            )
        elif self.aggregation_strategy == "FedYogi":
            global_state_dict, self.fedyogi_momentum, self.fedyogi_velocity = FedYogi(
                client_state_dicts,
                global_model_state_dict=global_peft_state_dict,
                proxy_dict=self.fedyogi_momentum,
                opt_proxy_dict=self.fedyogi_velocity,
                round_idx=round_idx,
                beta1=self.federated_args['beta1'],
                beta2=self.federated_args['beta2'],
                epsilon=self.federated_args['epsilon'],
                tau=self.federated_args['tau'],
                client_weights=client_weights
            )
        elif self.aggregation_strategy == "FedAdam":
            global_state_dict, self.fedadam_momentum, self.fedadam_velocity = FedAdam(
                client_state_dicts,
                global_model_state_dict=global_peft_state_dict,
                proxy_dict=self.fedadam_momentum,
                opt_proxy_dict=self.fedadam_velocity,
                round_idx=round_idx,
                beta1=self.federated_args['beta1'],
                beta2=self.federated_args['beta2'],
                epsilon=self.federated_args['epsilon'],
                tau=self.federated_args['tau'],
                client_weights=client_weights
            )

        elif self.aggregation_strategy == "FedProx":
            global_state_dict = FedAvg(
                client_state_dicts,
                global_model_state_dict=global_peft_state_dict,
                client_weights=client_weights
            )
        
        # Move aggregated state dict back to CPU if model is on CPU (to match model device)
        model_device = next(self.model.parameters()).device
        if model_device.type == "cpu" and use_gpu_for_agg:
            global_state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                                for k, v in global_state_dict.items()}
            if self.fedavgm_momentum is not None:
                self.fedavgm_momentum = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                                        for k, v in self.fedavgm_momentum.items()}
            if self.fedadagrad_velocity is not None:
                self.fedadagrad_velocity = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                                           for k, v in self.fedadagrad_velocity.items()}
            if self.fedyogi_momentum is not None:
                self.fedyogi_momentum = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                                        for k, v in self.fedyogi_momentum.items()}
            if self.fedyogi_velocity is not None:
                self.fedyogi_velocity = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                                        for k, v in self.fedyogi_velocity.items()}
            if self.fedadam_momentum is not None:
                self.fedadam_momentum = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                                        for k, v in self.fedadam_momentum.items()}
            if self.fedadam_velocity is not None:
                self.fedadam_velocity = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                                        for k, v in self.fedadam_velocity.items()}
            
        if self.is_peft:
            set_peft_model_state_dict(self.model, global_state_dict)
        else:
            self.model.load_state_dict(global_state_dict)

        logger.info("Global model updated")
