import torch
import logging
from copy import deepcopy
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from trainer.base import FinetuneTrainer
from torch.utils.data import Dataset
from trainer.federated.federated_utils import *
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
import os
import shutil

logger = logging.getLogger(__name__)

class FederatedFinetuneTrainer(FinetuneTrainer):
    
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
        aggregation_strategy="FedAvg",
        global_rounds=1,
        retain_ratio: float = 0.99,
        train_on: str = "full",
        **kwargs
    ):
        self.federated_dataset = train_dataset if isinstance(train_dataset, dict) else None
        self.train_on = train_on
        self.retain_ratio = retain_ratio
        dummy_train_dataset = None
        if self.federated_dataset and 0 in self.federated_dataset:
            dummy_train_dataset = self._select_train_dataset(self.federated_dataset[0])

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
        self.aggregation_strategy = aggregation_strategy
        self.global_rounds = global_rounds
        self.client_models = []
        self.client_data_sizes = []  
        self.kwargs = kwargs

        # 并行配置：默认使用可用 GPU 数，可通过 method_args.client_parallelism / FEDPERM_CLIENT_DEVICES 覆盖
        default_parallelism = torch.cuda.device_count() or 1
        self.client_parallelism = max(1, int(kwargs.get("client_parallelism", default_parallelism)))
        device_override = kwargs.get("client_devices", None) or os.environ.get("FEDPERM_CLIENT_DEVICES", None)
        self.client_devices = self._parse_client_devices(device_override)
        

        self.is_peft = isinstance(self.model, PeftModel)



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
            'momentum_factor': kwargs.get('momentum_factor', 0.9)
        }
        
        if not self.federated_dataset:
            logger.warning("No federated dataset provided!")
        else:
            for client_idx in range(num_clients):
                if client_idx not in self.federated_dataset:
                    logger.warning(f"Client {client_idx} has no data!")
        
        logger.info(f"Initialized with {num_clients} clients, global rounds: {global_rounds}")
        logger.info(f"Model type - PEFT: {self.is_peft}")

    def train(self, **kwargs):

                

        self.model = self.model.cpu()
        self.model = deepcopy(self.model)
        
        for round_idx in range(self.global_rounds):
            logger.info(f"Starting global round {round_idx + 1}/{self.global_rounds}")
            
            # Server-side eval before broadcasting (聚合前评测)
            self._maybe_evaluate_model(
                model=self.model,
                round_idx=round_idx,
                stage="server_pre_agg",
            )

            self.client_models = [deepcopy(self.model) for _ in range(self.num_clients)]
            self.client_data_sizes = []  


            logger.info(f"Aggregation strategy: {self.aggregation_strategy}")

            training_tasks = []
            for client_idx in range(self.num_clients):
                if client_idx not in self.federated_dataset:
                    logger.warning(f"Skipping client {client_idx} as no data is available")
                    self.client_data_sizes.append(0)  
                    continue
                    
                client_model = self.client_models[client_idx]
                client_container = self.federated_dataset[client_idx]
                client_dataset = self._select_train_dataset(client_container)

                client_data_size = len(client_dataset) if client_dataset is not None else 0
                self.client_data_sizes.append(client_data_size)
                
                # Client-side eval before local training：第 1 轮直接复用 server_pre_agg，其余轮次复用上一轮 server_post_agg
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

                # 收集并行训练任务
                training_tasks.append(
                    (
                        client_idx,
                        client_model,
                        client_dataset,
                        self._device_for_client(client_idx),
                    )
                )

            max_workers = max(1, min(self.client_parallelism, len(self.client_devices)))
            logger.info(f"Launching client training with up to {max_workers} workers over devices {self.client_devices}")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self._train_client_task,
                        idx,
                        model,
                        dataset,
                        device,
                    )
                    for (idx, model, dataset, device) in training_tasks
                ]
                for future in as_completed(futures):
                    client_idx, trained_client_model = future.result()
                    self.client_models[client_idx] = trained_client_model

            # Client-side eval after local training（串行以避免 evaluator 并发冲突）
            for client_idx in range(self.num_clients):
                trained_client_model = self.client_models[client_idx] if client_idx < len(self.client_models) else None
                if trained_client_model is None:
                    continue
                self._maybe_evaluate_model(
                    model=trained_client_model,
                    round_idx=round_idx,
                    stage=f"client{client_idx}_post_train",
                )
            
 
            self.aggregate_models(round_idx)

            # Server-side eval after aggregation
            self._maybe_evaluate_model(
                model=self.model,
                round_idx=round_idx,
                stage="server_post_agg",
            )
            logger.info(f"Completed global round {round_idx + 1}/{self.global_rounds}")
            
      
            if self._should_save_round(round_idx):
                save_path = f"{self.args.output_dir}/round_{round_idx + 1}"
             
                self.model = self.model.cpu()
                self.save_model(save_path)
                logger.info(f"Model saved at {save_path}")
        
        self.save_state()
        return None

    def train_client_model(self, client_model, client_dataset, device=None):
        # Move client model to GPU if available (Trainer will handle device placement, but we ensure it's on GPU)
        target_device = None
        if device is not None:
            target_device = device if isinstance(device, torch.device) else torch.device(device)
        elif torch.cuda.is_available():
            target_device = torch.device("cuda")
        if target_device is not None and target_device.type == "cuda":
            torch.cuda.set_device(target_device)
            client_model = client_model.to(target_device)
            logger.info(f"Client model moved to device: {target_device}")

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
                        # logger.info(f"FedProx loss: {loss.item()}, proximal_term: {proximal_term.item()}")
                    
                    return (loss, outputs) if return_outputs else loss
                
            trainer = FedProxTrainer(
                model=client_model,
                train_dataset=client_dataset,
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
                train_dataset=client_dataset,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                args=self.args,
                evaluator=self.evaluator,
                template_args=self.template_args,
            )
        
        trainer.train()
        # return trainer
        

        trained_model = trainer.model.cpu()
        if target_device is not None and target_device.type == "cuda":
            torch.cuda.empty_cache()
        return trained_model

    def _parse_client_devices(self, device_spec: Optional[str]) -> List[torch.device]:
        """
        返回可用设备列表；默认使用所有 GPU，若无 GPU 则仅 CPU。
        device_spec 可为逗号分隔的 GPU 编号（如 "0,1"）。
        """
        if not torch.cuda.is_available():
            return [torch.device("cpu")]
        if device_spec:
            devices = []
            for part in str(device_spec).split(","):
                part = part.strip()
                if part == "":
                    continue
                try:
                    idx = int(part)
                    if idx < torch.cuda.device_count():
                        devices.append(torch.device(f"cuda:{idx}"))
                except ValueError:
                    logger.warning(f"Invalid device id in FEDPERM_CLIENT_DEVICES: {part}")
            if devices:
                return devices
        return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]

    def _device_for_client(self, client_idx: int) -> torch.device:
        pool = self.client_devices or [torch.device("cpu")]
        return pool[client_idx % len(pool)]

    def _train_client_task(
        self,
        client_idx: int,
        client_model,
        client_dataset,
        device: torch.device,
    ):
        logger.info(f"Training on client {client_idx}")
        logger.info(f"Client {client_idx} dataset size: {len(client_dataset) if client_dataset is not None else 0} samples")
        logger.info(f"Training args - Batch size: {self.args.per_device_train_batch_size}, Epochs: {self.args.num_train_epochs}")
        if hasattr(self.args, 'max_steps') and self.args.max_steps > 0:
            logger.info(f"Max steps limit: {self.args.max_steps}")
        trained_client_model = self.train_client_model(
            client_model,
            client_dataset,
            device=device,
        )
        return client_idx, trained_client_model

    def _maybe_evaluate_model(self, model, round_idx: int, stage: str):
        """
        Run evaluator (if provided) and save SUMMARY.json for the given stage.
        """
        if self.evaluator is None:
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

        # If summary exists and no force flag, skip to save time
        force_eval = os.environ.get("FORCE_FEDPERM_EVAL", "0") in ("1", "true", "True")
        if summary_exists and not force_eval:
            logger.info(
                f"Skipping eval for stage {stage} at round {round_idx + 1}, existing summary: {summary_file}"
            )
            return

        # Move model to GPU if available for faster evaluation
        model_was_on_cpu = next(model.parameters()).device.type == "cpu"
        if torch.cuda.is_available() and model_was_on_cpu:
            device = torch.device("cuda")
            model = model.to(device)
            logger.info(f"Model moved to GPU device {device} for evaluation at stage {stage}")

        # Respect eval overwrite flag for downstream evaluator (only matters if we actually run)
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
            # Move model back to CPU if it was originally on CPU (to save GPU memory)
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
            for model in self.client_models:
                # Extract state dict (can be done on GPU if model is on GPU)
                peft_state_dict = get_peft_model_state_dict(model)
                
                if not peft_state_dict:
                    logger.warning(f"Client {model} has no PEFT state dict")
                    continue
                
                # Move tensors to aggregation device if needed
                if use_gpu_for_agg:
                    peft_state_dict = {k: v.to(agg_device) if isinstance(v, torch.Tensor) else v 
                                      for k, v in peft_state_dict.items()}
            
                client_state_dicts.append(peft_state_dict)
                logger.debug(f"Client {model} has PEFT state dict")
        else:
            # Extract state dicts and move to aggregation device
            client_state_dicts = []
            for model in self.client_models:
                state_dict = model.state_dict()
                if use_gpu_for_agg:
                    state_dict = {k: v.to(agg_device) if isinstance(v, torch.Tensor) else v 
                                 for k, v in state_dict.items()}
                client_state_dicts.append(state_dict)
        
        logger.info(f"Aggregation strategy: {self.aggregation_strategy}, device: {agg_device}")
        
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
            
            # self.model.load_state_dict(global_state_dict, strict=False)
            set_peft_model_state_dict(self.model, global_state_dict)
        else:
            
            self.model.load_state_dict(global_state_dict)
            
        logger.info("Global model updated")


    def save_model(self, output_dir=None):

        if output_dir is None:
            output_dir = self.args.output_dir

        os.makedirs(output_dir, exist_ok=True)
        
        if self.is_peft:
            logger.info(f"Saving PEFT model to {output_dir}")
            self.model.save_pretrained(output_dir, safe_serialization=False)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir, safe_serialization=False)
        else:
            super().save_model(output_dir)

    def _should_save_round(self, round_idx: int) -> bool:
        """Respect FEDPERM_SKIP_SAVE_ROUNDS env for per-round checkpoints."""
        if self.args.save_strategy != "epoch" and round_idx != self.global_rounds - 1:
            return False
        skip_env = os.environ.get("FEDPERM_SKIP_SAVE_ROUNDS", "")
        if skip_env:
            try:
                skip_rounds = {int(x) for x in skip_env.split(",") if x.strip() != ""}
                if round_idx in skip_rounds and round_idx != self.global_rounds - 1:
                    logger.info(
                        f"Skipping save at round {round_idx + 1} due to FEDPERM_SKIP_SAVE_ROUNDS"
                    )
                    return False
            except Exception as exc:
                logger.warning(f"Failed to parse FEDPERM_SKIP_SAVE_ROUNDS='{skip_env}': {exc}")
        return True

    def _select_train_dataset(self, client_data):
        """
        Choose which split to train on for the client.
        If client_data carries .full/.retain/.forget, pick by train_on; else return as-is.
        """
        if client_data is None:
            return None
        if hasattr(client_data, "full") or hasattr(client_data, "retain") or hasattr(client_data, "forget"):
            if self.train_on == "retain" and getattr(client_data, "retain", None) is not None:
                return client_data.retain
            if self.train_on == "forget" and getattr(client_data, "forget", None) is not None:
                return client_data.forget
            return getattr(client_data, "full", client_data)
        return client_data
