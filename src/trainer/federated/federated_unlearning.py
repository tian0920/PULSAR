import torch
import logging
from copy import deepcopy
from typing import Dict, List, Any, Optional
import os
from trainer.base import FinetuneTrainer
from trainer.unlearn.base import UnlearnTrainer
from data.unlearn import ForgetRetainDataset
from torch.utils.data import Dataset
from trainer.federated.federated_utils import *
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict


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
        
  
        self.is_peft = isinstance(self.model, PeftModel)

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
            # 在目标客户端保留训练前保存“纯遗忘后”模型，便于单独评测遗忘效果
            pre_retain_target_path = f"{self.args.output_dir}/pre_retain_target_client_{self.target_client_idx}"
            os.makedirs(pre_retain_target_path, exist_ok=True)
            tgt_model = unlearn_trainer.model.cpu()
            if self.is_peft:
                tgt_model.save_pretrained(pre_retain_target_path)
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(pre_retain_target_path)
            else:
                torch.save(
                    tgt_model.state_dict(),
                    os.path.join(pre_retain_target_path, "pytorch_model.bin"),
                )
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(pre_retain_target_path)
            logger.info(
                f"Saved pre-retain target client {self.target_client_idx} model at {pre_retain_target_path}"
            )
        else:
            logger.warning(f"Target client {self.target_client_idx} has no data, skipping unlearning")
    
        for round_idx in range(self.global_rounds):
            logger.info(f"Starting global round {round_idx + 1}/{self.global_rounds}")
            
    
            logger.info("Step 2: All clients training on retain data with updated global model")
            
         
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
                
                retain_data = client_dataset.retain.retain
                client_data_size = len(retain_data)
                self.client_data_sizes.append(client_data_size)
                
                logger.info(f"Training client {client_idx} on retain data with {self.args.num_train_epochs} epochs")
                self.train_client_model(client_model, client_dataset, client_idx)
                
                self.client_models[client_idx] = client_model.cpu()

            logger.info("Step 3: Aggregating all client models")
            self.aggregate_models(round_idx)
            logger.info(f"Completed global round {round_idx + 1}/{self.global_rounds}")
            
            if self.args.save_strategy == "epoch" or (round_idx == self.global_rounds - 1):
                save_path = f"{self.args.output_dir}/round_{round_idx + 1}"
                self.model = self.model.cpu()
                self.save_model(save_path)
                logger.info(f"Model saved at {save_path}")
        
        # 保存最终的全局模型与目标客户端模型，便于后续测评
        self.save_state()
        final_global_path = f"{self.args.output_dir}/final_global"
        self.model = self.model.cpu()
        self.save_model(final_global_path)
        logger.info(f"Final global model saved at {final_global_path}")

        target_model = None
        if (
            0 <= self.target_client_idx < len(self.client_models)
            and self.client_models[self.target_client_idx] is not None
        ):
            target_model = self.client_models[self.target_client_idx]
        elif unlearn_trainer is not None and hasattr(unlearn_trainer, "model"):
            target_model = unlearn_trainer.model

        if target_model is not None:
            final_target_client_path = f"{self.args.output_dir}/final_target_client_{self.target_client_idx}"
            os.makedirs(final_target_client_path, exist_ok=True)
            target_model = target_model.cpu()
            if self.is_peft:
                logger.info(
                    f"Saving final target client {self.target_client_idx} PEFT model to {final_target_client_path}"
                )
                target_model.save_pretrained(final_target_client_path)
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(final_target_client_path)
            else:
                torch.save(
                    target_model.state_dict(),
                    os.path.join(final_target_client_path, "pytorch_model.bin"),
                )
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(final_target_client_path)
            logger.info(
                f"Final target client {self.target_client_idx} model saved at {final_target_client_path}"
            )
        else:
            logger.warning("Target client model not found, skipping target client save")

        return None

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

        # 记录当前全局模型状态，作为构造 filter_F_A 的基线
        # Extract state dict without moving to CPU first (can be done on GPU if model is on GPU)
        if self.is_peft:
            base_state = get_peft_model_state_dict(self.model)
        else:
            base_state = self.model.state_dict()
        
        # Use GPU for filter computation if available (faster)
        use_gpu_for_filter = torch.cuda.is_available()
        filter_device = torch.device("cuda") if use_gpu_for_filter else torch.device("cpu")
        if use_gpu_for_filter:
            base_state = {k: v.to(filter_device) if isinstance(v, torch.Tensor) else v 
                         for k, v in base_state.items()}

        logger.info(f"forget_data length: {len(forget_dataset)}")
        logger.info(f"retain_data length: {len(retain_dataset.retain)}")
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
                return {
                    "forget": self.forget_data[idx],
                    "retain": self.retain_data[retain_idx]
                }
        
        combined_dataset = CombinedDataset(forget_dataset, retain_dataset)

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
                **self.kwargs
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
                **self.kwargs
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

        # 记录遗忘后参数，用于后续目标客户端保留训练的 proximal 约束
        # Extract state dict without moving to CPU first (can be done on GPU if model is on GPU)
        if self.is_peft:
            self.theta_unlearn_state = get_peft_model_state_dict(unlearn_trainer.model)
        else:
            self.theta_unlearn_state = unlearn_trainer.model.state_dict()
        
        # Move theta_unlearn_state to filter device if needed
        if use_gpu_for_filter:
            self.theta_unlearn_state = {k: v.to(filter_device) if isinstance(v, torch.Tensor) else v 
                                       for k, v in self.theta_unlearn_state.items()}

        # 构造 filter_F_A = base_state - theta_unlearn_state (on GPU if available)
        self.filter_F_A = {}
        missing_keys = 0
        for name, tensor in base_state.items():
            ref = self.theta_unlearn_state.get(name) if self.theta_unlearn_state else None
            if ref is None:
                self.filter_F_A[name] = None
                missing_keys += 1
                continue
            self.filter_F_A[name] = (tensor - ref).detach().clone()
        if missing_keys > 0:
            logger.warning(f"{missing_keys} keys in base_state not found in theta_unlearn_state or not Tensor.")
        
        return unlearn_trainer
    

    def train_client_model(self, client_model, client_dataset, client_idx):
        retain_dataset = client_dataset.retain
        retain_data = retain_dataset.retain
        
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
        elif (
            client_idx == self.target_client_idx
            and self.theta_unlearn_state is not None
        ):
            # 仅对目标客户端应用 prox 约束，保持靠近遗忘后参数 theta_unlearn_state
            class ProxToUnlearnTrainer(FinetuneTrainer):
                def __init__(self, *args, ref_state=None, mu=0.01, **kwargs):
                    super().__init__(*args, **kwargs)
                    self.ref_state = ref_state
                    self.mu = mu

                def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                    return_values = super().compute_loss(model, inputs, return_outputs=return_outputs)
                    if return_outputs:
                        loss, outputs = return_values
                    else:
                        loss = return_values

                    proximal_term = 0.0
                    if self.ref_state is not None:
                        for name, w in model.named_parameters():
                            if not w.requires_grad:
                                continue
                            key = name.replace(".default", "")
                            ref = self.ref_state.get(key)
                            if ref is None:
                                continue
                            proximal_term += torch.norm(w - ref.to(w.device).detach()) ** 2
                        loss = loss + (self.mu / 2) * proximal_term
                    return (loss, outputs) if return_outputs else loss

            trainer = ProxToUnlearnTrainer(
                model=client_model,
                train_dataset=retain_data,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                args=self.args,
                evaluator=self.evaluator,
                template_args=self.template_args,
                ref_state=self.theta_unlearn_state,
                mu=self.federated_args["prox_to_unlearn_mu"],
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
        
        # Move filter_F_A to aggregation device if it exists
        if hasattr(self, "filter_F_A") and self.filter_F_A is not None and use_gpu_for_agg:
            self.filter_F_A = {k: v.to(agg_device) if isinstance(v, torch.Tensor) else v 
                              for k, v in self.filter_F_A.items()}
        
        if self.is_peft:
            client_state_dicts = []
            for idx, model in enumerate(self.client_models):
                # Extract state dict without moving to CPU first
                peft_state_dict = get_peft_model_state_dict(model)
                # Move to aggregation device if needed
                if use_gpu_for_agg:
                    peft_state_dict = {k: v.to(agg_device) if isinstance(v, torch.Tensor) else v 
                                      for k, v in peft_state_dict.items()}
                # 对目标客户端的更新加回遗忘过滤器，避免遗忘方向被带入全局聚合
                if (
                    idx == self.target_client_idx
                    and hasattr(self, "filter_F_A")
                    and self.filter_F_A is not None
                ):
                    restored_state = {}
                    for name, tensor in peft_state_dict.items():
                        filt = self.filter_F_A.get(name) if self.filter_F_A else None
                        if filt is None:
                            restored_state[name] = tensor
                        else:
                            restored_state[name] = tensor + filt.to(tensor.device)
                    peft_state_dict = restored_state
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
                if (
                    idx == self.target_client_idx
                    and hasattr(self, "filter_F_A")
                    and self.filter_F_A is not None
                ):
                    restored_state = {}
                    for name, tensor in state.items():
                        filt = self.filter_F_A.get(name) if self.filter_F_A else None
                        if filt is None:
                            restored_state[name] = tensor
                        else:
                            restored_state[name] = tensor + filt.to(tensor.device)
                    state = restored_state
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