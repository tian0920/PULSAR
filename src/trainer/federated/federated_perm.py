import torch
import logging
from copy import deepcopy
from typing import Dict, List, Any, Optional
import os
from trainer.base import FinetuneTrainer
from torch.utils.data import Dataset
from trainer.federated.federated_utils import (
    FedAvg,
    generate_permutation_key,
    apply_permutation,
    apply_inverse_permutation,
    apply_adversarial_filter,
)
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from trainer.unlearn.grad_diff import GradDiff

logger = logging.getLogger(__name__)


class FederatedPermTrainer(FinetuneTrainer):
    """
    FedPerm: Dynamic Authorization with Server-side Filtering and Obfuscation
    """
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
        target_client_idx=1,  # Client B
        revoker_client_idx=0,  # Client A
        aggregation_strategy="FedAvg",
        global_rounds=1,
        filter_alpha=1.0,
        aggregation_lr=1.0,
        enable_revocation=True,
        revocation_round=None,
        unlearn_epochs: Optional[int] = None,
        **kwargs
    ):
        self.federated_dataset = train_dataset if isinstance(train_dataset, dict) else None
        dummy_train_dataset = None
        if self.federated_dataset and 0 in self.federated_dataset:
            dummy_train_dataset = self.federated_dataset[0]

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

        self.num_clients = num_clients
        self.target_client_idx = target_client_idx
        self.revoker_client_idx = revoker_client_idx
        self.aggregation_strategy = aggregation_strategy
        self.global_rounds = global_rounds
        self.filter_alpha = filter_alpha
        self.aggregation_lr = aggregation_lr
        self.enable_revocation = enable_revocation
        self.revocation_round = revocation_round
        self.unlearn_epochs = unlearn_epochs
        self.kwargs = kwargs

        self.revocation_status = False
        self.filter_F_A = None
        self.permutation_keys = {}
        self.client_models = []
        self.client_data_sizes = []

        self.is_peft = isinstance(self.model, PeftModel)

        if not self.federated_dataset:
            logger.warning("No federated dataset provided!")
        else:
            for client_idx in range(num_clients):
                if client_idx not in self.federated_dataset:
                    logger.warning(f"Client {client_idx} has no data!")

        logger.info(f"FedPerm initialized with {num_clients} clients")
        logger.info(f"Target client (B): {self.target_client_idx}, Revoker client (A): {self.revoker_client_idx}")
        logger.info(f"Model type - PEFT: {self.is_peft}")
        logger.info(f"Revocation enabled: {self.enable_revocation}")

    def train(self, **kwargs):
        """Main training loop for FedPerm"""
        if not self.federated_dataset:
            raise ValueError("Federated dataset is not properly set up")
        
        self.model = self.model.cpu()
        self.model = deepcopy(self.model)
        
        self._should_trigger_revocation = False

        for round_idx in range(self.global_rounds):
            logger.info(f"Starting global round {round_idx + 1}/{self.global_rounds}")
            
            # Check for revocation trigger
            if self.enable_revocation and not self.revocation_status:
                if self.revocation_round is not None and round_idx == self.revocation_round:
                    self._trigger_revocation(round_idx)
                if getattr(self, "_should_trigger_revocation", False):
                    self._trigger_revocation(round_idx)
                    self._should_trigger_revocation = False
                # Note: Manual revocation can be triggered via trigger_revocation() method
            
            # Phase 1: Client Selection & Distribution
            logger.info("Phase 1: Client Selection & Distribution")
            self.client_models = [deepcopy(self.model) for _ in range(self.num_clients)]
            self.client_data_sizes = []
            
            # Get global model state
            if self.is_peft:
                global_state = get_peft_model_state_dict(self.model.cpu())
            else:
                global_state = self.model.cpu().state_dict()
            
            # Distribute models to clients
            for client_idx in range(self.num_clients):
                if client_idx not in self.federated_dataset:
                    logger.warning(f"Skipping client {client_idx} as no data is available")
                    self.client_data_sizes.append(0)
                    continue
                
                client_model = self.client_models[client_idx]
                
                # Apply filtering and obfuscation for target client B if revocation is active
                if self.revocation_status and client_idx == self.target_client_idx:
                    logger.info(f"Applying adversarial filter and obfuscation for target client {client_idx}")
                    
                    # Step 1: Apply adversarial filter
                    if self.filter_F_A is not None:
                        filtered_state = apply_adversarial_filter(global_state, self.filter_F_A, self.filter_alpha)
                    else:
                        logger.warning("Filter F_A is None, skipping filter application")
                        filtered_state = global_state.copy()
                    
                    # Step 2: Apply structural obfuscation
                    perm_key = generate_permutation_key(
                        filtered_state,
                        seed=round_idx * 1000 + client_idx,
                        predicate=self._is_adapter_parameter,
                    )
                    self.permutation_keys[round_idx] = {client_idx: perm_key}
                    
                    # Apply permutation
                    obfuscated_state = apply_permutation(filtered_state, perm_key)
                    
                    # Load obfuscated model to client
                    if self.is_peft:
                        set_peft_model_state_dict(client_model, obfuscated_state)
                    else:
                        client_model.load_state_dict(obfuscated_state)
                    
                    logger.info(f"Client {client_idx} received obfuscated model")
                else:
                    # Authorized clients get normal model
                    if self.is_peft:
                        set_peft_model_state_dict(client_model, global_state)
                    else:
                        client_model.load_state_dict(global_state)
                    logger.info(f"Client {client_idx} received normal model")
                
                client_data_size = len(self.federated_dataset[client_idx])
                self.client_data_sizes.append(client_data_size)
            
            # Phase 2: Local Training (Parallel)
            logger.info("Phase 2: Local Training")
            client_initial_states = []  # Store initial states for update computation
            client_updated_states = []  # Store updated states
            
            for client_idx in range(self.num_clients):
                if client_idx not in self.federated_dataset:
                    client_initial_states.append(None)
                    client_updated_states.append(None)
                    continue
                
                client_model = self.client_models[client_idx]
                client_dataset = self.federated_dataset[client_idx]
                
                # Store initial state
                if self.is_peft:
                    initial_state = get_peft_model_state_dict(client_model.cpu())
                else:
                    initial_state = client_model.cpu().state_dict()
                client_initial_states.append(initial_state)
                
                logger.info(f"Training client {client_idx}")
                trained_model = self.train_client_model(client_model, client_dataset)
                
                # Store updated state
                if self.is_peft:
                    updated_state = get_peft_model_state_dict(trained_model.cpu())
                else:
                    updated_state = trained_model.cpu().state_dict()
                client_updated_states.append(updated_state)
                
                self.client_models[client_idx] = trained_model
            
            # Phase 3: Secure Aggregation
            logger.info("Phase 3: Secure Aggregation")
            self._aggregate_with_alignment(
                round_idx, 
                client_initial_states, 
                client_updated_states,
                global_state
            )
            
            logger.info(f"Completed global round {round_idx + 1}/{self.global_rounds}")
            
            # Save model
            if self.args.save_strategy == "epoch" or (round_idx == self.global_rounds - 1):
                save_path = f"{self.args.output_dir}/round_{round_idx + 1}"
                self.model = self.model.cpu()
                self.save_model(save_path)
                logger.info(f"Model saved at {save_path}")
        
        self.save_state()
        return None

    def _trigger_revocation(self, round_idx: int):
        """
        触发授权撤销：在 Client A 的遗忘数据上训练出一个“已遗忘”的模型，
        再用 (Theta^t - Theta_unlearn) 构造过滤器 F_A。
        """
        logger.info(f"=== Triggering Revocation at Round {round_idx} ===")
        logger.info(f"Client A (idx {self.revoker_client_idx}) optimizing filter F_A")

        if self.revoker_client_idx not in self.federated_dataset:
            logger.warning(
                f"Revoker client {self.revoker_client_idx} has no data, skipping revocation"
            )
            return

        client_A_dataset = self.federated_dataset[self.revoker_client_idx]

        # 1. 准备“忘记 + 保留”数据
        # 如果是 unlearn 模式，client_A_dataset 通常是 ForgetRetainDataset，包含 .forget / .retain
        if hasattr(client_A_dataset, "forget") and hasattr(client_A_dataset, "retain"):
            forget_dataset = client_A_dataset.forget
            retain_dataset = client_A_dataset.retain
            retain_inner = getattr(retain_dataset, "retain", retain_dataset)
        else:
            # 否则退化为：全量数据当作 forget，retain 用同一份数据
            forget_dataset = client_A_dataset
            retain_inner = client_A_dataset

        class CombinedDataset(Dataset):
            """
            简化版 Combine：每个样本同时提供 forget 和 retain，
            方便直接复用 GradDiff 的 compute_loss 逻辑。
            """
            def __init__(self, forget_data, retain_data):
                self.forget_data = forget_data
                self.retain_data = retain_data
                self.retain_len = len(retain_data)

            def __len__(self):
                return len(self.forget_data)

            def __getitem__(self, idx):
                retain_idx = idx % self.retain_len
                return {
                    "forget": {"forget": self.forget_data[idx]},
                    "retain": {"retain": self.retain_data[retain_idx]},
                }

        combined_dataset = CombinedDataset(forget_dataset, retain_inner)

        # 2. 构造一个用于遗忘阶段的 Trainer（GradDiff）
        unlearn_args = deepcopy(self.args)
        if self.unlearn_epochs is not None:
            unlearn_args.num_train_epochs = self.unlearn_epochs
            logger.info(f"Using {self.unlearn_epochs} epochs for unlearn stage")
        else:
            logger.info(f"Using default {unlearn_args.num_train_epochs} epochs for unlearn stage")

        # 使用当前全局模型的一个拷贝作为遗忘起点
        unlearn_model = deepcopy(self.model)

        unlearn_trainer = GradDiff(
            model=unlearn_model,
            train_dataset=combined_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            args=unlearn_args,
            evaluator=self.evaluator,
            template_args=self.template_args,
            **self.kwargs,
        )

        logger.info("Start unlearning training on Client A's data (GradDiff)...")
        unlearn_trainer.train()
        logger.info("Unlearning training finished.")

        # 3. 计算当前全局参数与遗忘后模型之间的差值 → 过滤器 F_A
        if self.is_peft:
            base_state = get_peft_model_state_dict(self.model.cpu())
            unlearn_state = get_peft_model_state_dict(unlearn_trainer.model.cpu())
        else:
            base_state = self.model.cpu().state_dict()
            unlearn_state = unlearn_trainer.model.cpu().state_dict()

        self.filter_F_A = {}
        missing_keys = 0
        for key, param in base_state.items():
            if key in unlearn_state and isinstance(param, torch.Tensor):
                # F_A[key] = Theta^t[key] - Theta_unlearn[key]
                self.filter_F_A[key] = (param - unlearn_state[key]).detach().clone()
            else:
                self.filter_F_A[key] = None
                missing_keys += 1

        if missing_keys > 0:
            logger.warning(f"{missing_keys} keys in base_state not found in unlearn_state or not Tensor.")

        logger.info("Filter F_A constructed from (Theta^t - Theta_unlearn).")
        self.revocation_status = True
        logger.info("RevocationStatus set to True")

    def train_client_model(self, client_model, client_dataset):
        """Train client model locally"""
        trainer = FinetuneTrainer(
            model=client_model,
            train_dataset=client_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            args=self.args,
            evaluator=self.evaluator,
            template_args=self.template_args,
        )

        trainer.train()
        trained_model = trainer.model.cpu()
        return trained_model

    def _aggregate_with_alignment(self, round_idx: int, initial_states: List, updated_states: List, global_state: Dict):
        """
        Secure aggregation with re-alignment for target client B
        """
        logger.info("Aggregating client models with alignment")

        total_samples = sum(self.client_data_sizes)
        if total_samples == 0:
            logger.warning("No samples available for aggregation")
            return

        client_weights = [size / total_samples for size in self.client_data_sizes]
        logger.info(f"Client sample sizes: {self.client_data_sizes}")
        logger.info(f"Client weights: {[f'{w:.4f}' for w in client_weights]}")

        # Collect aligned client parameters (not parameter deltas)
        aligned_client_states = []
        for client_idx in range(self.num_clients):
            if updated_states[client_idx] is None:
                # No data for this client; contribute current global state (weight should be ~0)
                aligned_client_states.append(global_state)
                continue

            client_state = updated_states[client_idx]

            # Re-align if this is target client B and revocation is active
            if self.revocation_status and client_idx == self.target_client_idx:
                if round_idx in self.permutation_keys and client_idx in self.permutation_keys[round_idx]:
                    logger.info(f"Re-aligning parameters from target client {client_idx}")
                    perm_key = self.permutation_keys[round_idx][client_idx]
                    client_state = apply_inverse_permutation(client_state, perm_key)
                else:
                    logger.warning(f"No permutation key found for client {client_idx} at round {round_idx}")

                # Add back filter so the forget effect is not propagated
                if self.filter_F_A is not None:
                    restored_state = {}
                    for key, tensor in client_state.items():
                        if not isinstance(tensor, torch.Tensor):
                            restored_state[key] = tensor
                            continue
                        filter_param = self.filter_F_A.get(key)
                        if filter_param is None:
                            restored_state[key] = tensor.clone()
                        else:
                            restored_state[key] = tensor + self.filter_alpha * filter_param.to(tensor.device)
                    client_state = restored_state

            aligned_client_states.append(client_state)

        # Aggregate full parameters (FedAvg-style) using client weights
        aggregated_state = FedAvg(
            aligned_client_states,
            global_model_state_dict=global_state,
            client_weights=client_weights,
        )

        # Load updated global state
        if self.is_peft:
            set_peft_model_state_dict(self.model, aggregated_state)
        else:
            self.model.load_state_dict(aggregated_state)

        logger.info("Global model updated with aligned aggregations")

    def trigger_revocation(self):
        """
        Public method to manually trigger revocation
        Can be called externally to trigger revocation at any time
        """
        if self.revocation_status:
            logger.warning("Revocation already active")
            return

        self._should_trigger_revocation = True

    def save_model(self, output_dir=None):
        """Save model with FedPerm state"""
        if output_dir is None:
            output_dir = self.args.output_dir

        os.makedirs(output_dir, exist_ok=True)

        if self.is_peft:
            logger.info(f"Saving PEFT model to {output_dir}")
            self.model.save_pretrained(output_dir)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir)
        else:
            super().save_model(output_dir)

        # Save FedPerm specific state
        perm_state = {
            'revocation_status': self.revocation_status,
            'filter_F_A': self.filter_F_A,
            'permutation_keys': self.permutation_keys,
        }
        torch.save(perm_state, os.path.join(output_dir, 'fedperm_state.pt'))
        logger.info(f"FedPerm state saved to {output_dir}")

    def _is_adapter_parameter(self, name: str, param: torch.Tensor) -> bool:
        """
        Determine whether a parameter belongs to adapter modules (e.g., LoRA)
        and is eligible for permutation.
        """
        if not self.is_peft or not isinstance(param, torch.Tensor):
            return False
        return "lora" in name.lower()
