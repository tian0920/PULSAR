import torch
import logging
from copy import deepcopy
from typing import Dict, List, Any, Optional

from trainer.base import FinetuneTrainer
from torch.utils.data import Dataset
from trainer.federated.federated_utils import *
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from transformers.trainer_utils import IntervalStrategy, SaveStrategy
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

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
        target_client_idx: Optional[int] = None,
        save_target_client_checkpoint: bool = False,
        aggregation_strategy="FedAvg",
        global_rounds=1,
        retain_ratio: float = 0.99,
        train_on: str = "full",
        **kwargs
    ):
        self.federated_dataset = train_dataset if isinstance(train_dataset, dict) else None
        self.train_on = train_on
        self.retain_ratio = retain_ratio
        self.enable_pre_eval = os.environ.get("ENABLE_PRE_EVAL", "false").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
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
        self.target_client_idx = target_client_idx
        self.revoker_client_idx = int(kwargs.get("revoker_client_idx", 0))
        self.source_to_targets = self._resolve_source_to_targets(
            kwargs.get("source_to_targets", None),
            target_client_idx=target_client_idx,
        )
        self.source_client_indices = sorted(self.source_to_targets.keys())
        self.target_client_indices = self._resolve_target_client_indices(
            kwargs.get("target_client_indices", None),
            target_client_idx=target_client_idx,
            source_to_targets=self.source_to_targets,
        )
        self.save_target_client_checkpoint = save_target_client_checkpoint
        self.aggregation_strategy = aggregation_strategy
        self.global_rounds = global_rounds
        self.client_models = []
        self.client_data_sizes = []  
        self.kwargs = kwargs

        # 单卡：客户端与服务器均使用同一设备
        self.train_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.server_device = self.train_device
        self.client_parallel_devices = self._resolve_device_pool(
            kwargs.get("client_parallel_devices", None),
            fallback_device=self.train_device,
        )
        self.parallel_client_workers = int(kwargs.get("parallel_client_workers", 0))
        

        self.is_peft = isinstance(self.model, PeftModel)
        self.args.save_strategy = self._normalize_save_strategy(getattr(self.args, "save_strategy", None))



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
        logger.info(f"Client parallel devices: {self.client_parallel_devices}, workers={self.parallel_client_workers}")
        logger.info(f"Source clients for eval: {self.source_client_indices}")
        if self.save_target_client_checkpoint:
            logger.info(f"Save target client checkpoints: {self.target_client_indices}")

    def _normalize_client_indices(self, client_indices):
        normalized = []
        for idx in client_indices or []:
            client_idx = int(idx)
            if client_idx < 0 or client_idx >= self.num_clients:
                raise ValueError(
                    f"Client index {client_idx} out of range for num_clients={self.num_clients}"
                )
            normalized.append(client_idx)
        return sorted(dict.fromkeys(normalized))

    def _resolve_source_to_targets(self, source_to_targets, target_client_idx: Optional[int]):
        if source_to_targets is None:
            return {}
        normalized = {}
        for raw_src, raw_targets in source_to_targets.items():
            src_idx = int(raw_src)
            if src_idx < 0 or src_idx >= self.num_clients:
                raise ValueError(
                    f"Source client index {src_idx} out of range for num_clients={self.num_clients}"
                )
            target_indices = self._normalize_client_indices(raw_targets)
            if src_idx in target_indices:
                raise ValueError(
                    f"Source client {src_idx} cannot target itself in source_to_targets"
                )
            normalized[src_idx] = target_indices
        return normalized

    def _resolve_target_client_indices(
        self,
        configured_target_indices,
        target_client_idx: Optional[int],
        source_to_targets: Dict[int, List[int]],
    ) -> List[int]:
        resolved = self._normalize_client_indices(configured_target_indices)
        if resolved:
            return resolved
        if source_to_targets:
            target_indices = []
            for values in source_to_targets.values():
                target_indices.extend(values)
            return self._normalize_client_indices(target_indices)
        if target_client_idx is not None:
            return [int(target_client_idx)]
        return []

    def _resolve_device_pool(self, configured_devices, fallback_device: torch.device) -> List[torch.device]:
        if configured_devices:
            devices = []
            for device_spec in configured_devices:
                device = device_spec if isinstance(device_spec, torch.device) else torch.device(str(device_spec))
                if device.type == "cuda" and not torch.cuda.is_available():
                    logger.warning(f"CUDA device {device} requested but CUDA is unavailable; skipping it")
                    continue
                if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
                    logger.warning(f"CUDA device {device} requested but only {torch.cuda.device_count()} devices are available; skipping it")
                    continue
                devices.append(device)
            if devices:
                return devices

        if torch.cuda.is_available():
            return [torch.device(f"cuda:{idx}") for idx in range(torch.cuda.device_count())]
        return [fallback_device]

    def _resolve_worker_count(self, requested_workers: int, task_count: int, device_pool: List[torch.device]) -> int:
        if task_count <= 1:
            return 1
        if not device_pool:
            return 1
        has_cuda = any(device.type == "cuda" for device in device_pool)
        if requested_workers <= 0:
            if has_cuda:
                requested_workers = len(device_pool)
            else:
                requested_workers = 1
        if requested_workers <= 1:
            return 1
        if has_cuda:
            return max(1, min(requested_workers, len(device_pool), task_count))
        return max(1, min(requested_workers, task_count))

    def _run_parallel_tasks(self, task_specs, worker_fn, device_pool, requested_workers, task_label: str):
        task_count = len(task_specs)
        if task_count == 0:
            return []
        devices = device_pool or [torch.device("cpu")]
        worker_count = self._resolve_worker_count(requested_workers, task_count, devices)
        if worker_count <= 1:
            logger.info(f"Running {task_count} {task_label} tasks sequentially")
            return [worker_fn(task_spec, devices[idx % len(devices)]) for idx, task_spec in enumerate(task_specs)]

        logger.info(
            "Running %s %s tasks in parallel with %s workers on devices %s",
            task_count,
            task_label,
            worker_count,
            devices,
        )
        futures = {}
        results = [None] * task_count
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for idx, task_spec in enumerate(task_specs):
                device = devices[idx % len(devices)]
                future = executor.submit(worker_fn, task_spec, device)
                futures[future] = idx
            for future in as_completed(futures):
                result_idx = futures[future]
                results[result_idx] = future.result()
        return results

    def train(self, **kwargs):

                

        self.model = self.model.cpu()
        self.model = deepcopy(self.model)
        start_round = self._resume_from_existing_rounds()
        if start_round >= self.global_rounds:
            logger.info("All required rounds already exist; skip federated finetune loop.")
            return None
        if start_round > 0:
            logger.info(f"Resuming federated finetune from round {start_round + 1}")

        for round_idx in range(start_round, self.global_rounds):
            logger.info(f"Starting global round {round_idx + 1}/{self.global_rounds}")
            
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
            client_task_specs = []
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

                client_task_specs.append(
                    {
                        "client_idx": client_idx,
                        "client_model": deepcopy(client_model).cpu(),
                        "client_dataset": client_dataset,
                    }
                )

            def _train_worker(task_spec, device):
                return self._train_client_task(
                    task_spec["client_idx"],
                    task_spec["client_model"],
                    task_spec["client_dataset"],
                    device=device,
                )

            for client_idx, trained_client_model in self._run_parallel_tasks(
                client_task_specs,
                _train_worker,
                self.client_parallel_devices,
                self.parallel_client_workers,
                task_label="client-train",
            ):
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
                self._maybe_save_target_client(round_idx)
        
        self.save_state()
        return None

    def _maybe_save_target_client(self, round_idx: int):
        """Optionally persist target clients' post-train checkpoints."""
        if not self.save_target_client_checkpoint:
            return
        if not self.target_client_indices:
            logger.warning("save_target_client_checkpoint enabled but no target clients are resolved; skip.")
            return
        for target_client_idx in self.target_client_indices:
            if target_client_idx >= len(self.client_models):
                logger.warning(
                    f"target_client_idx {target_client_idx} out of range for {len(self.client_models)} clients; skip."
                )
                continue
            target_client_model = self.client_models[target_client_idx]
            if target_client_model is None:
                logger.warning(f"Target client {target_client_idx} model missing; skip.")
                continue
            target_dir = f"{self.args.output_dir}/round_{round_idx + 1}_target_client_{target_client_idx}"
            os.makedirs(target_dir, exist_ok=True)
            if self.is_peft:
                logger.info(
                    f"Saving target client {target_client_idx} PEFT model to {target_dir} (model only)"
                )
                target_client_model = target_client_model.cpu()
                target_client_model.save_pretrained(target_dir, safe_serialization=False)
            else:
                target_client_model = target_client_model.cpu()
                torch.save(target_client_model.state_dict(), os.path.join(target_dir, "pytorch_model.bin"))
            logger.info(f"Target client {target_client_idx} model saved at {target_dir}")

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

    def _clear_evaluator_cache(self):
        if hasattr(self.evaluator, "metrics"):
            for metric in self.evaluator.metrics.values():
                if hasattr(metric, "data"):
                    metric.data = None
                if hasattr(metric, "collators"):
                    metric.collators = None

    def _resolve_eval_retain_indices_path(self, stage: str):
        split_dir = os.environ.get(
            "FEDPERM_SPLIT_DIR", os.path.join(self.args.output_dir, "client_splits")
        )
        retain_idx_path = None
        m = re.match(r"client(\d+)_", stage)
        if stage.startswith("server"):
            retain_idx_path = os.path.join(split_dir, "global_retain_indices.json")
        elif m:
            cidx = int(m.group(1))
            retain_idx_path = os.path.join(split_dir, f"client{cidx}_retain_indices.json")
        return split_dir, retain_idx_path

    def _set_eval_indices(self, stage: str, forget_source_idx: int):
        split_dir, retain_idx_path = self._resolve_eval_retain_indices_path(stage)
        forget_idx_path = os.path.join(
            split_dir, f"client{forget_source_idx}_forget_indices.json"
        )

        if hasattr(self.evaluator, "eval_cfg"):
            if hasattr(self.evaluator.eval_cfg, "tofu"):
                eval_tofu = self.evaluator.eval_cfg.tofu
                if not os.path.exists(forget_idx_path):
                    raise FileNotFoundError(
                        f"Forget indices missing for stage {stage}: {forget_idx_path}"
                    )
                eval_tofu.forget_indices_path = forget_idx_path
                if retain_idx_path:
                    if not os.path.exists(retain_idx_path):
                        raise FileNotFoundError(
                            f"Retain indices missing for stage {stage}: {retain_idx_path}"
                        )
                    eval_tofu.retain_indices_path = retain_idx_path

            if hasattr(self.evaluator.eval_cfg, "muse"):
                eval_muse = self.evaluator.eval_cfg.muse
                if not os.path.exists(forget_idx_path):
                    raise FileNotFoundError(
                        f"MUSE forget indices missing for stage {stage}: {forget_idx_path}"
                    )
                eval_muse.forget_indices_path = forget_idx_path

        self._clear_evaluator_cache()

    def _forget_eval_sources(self):
        return self.source_client_indices or [self.revoker_client_idx]

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
        force_eval = os.environ.get("FORCE_FEDPERM_EVAL", "0") in ("1", "true", "True")

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
            forget_sources = self._forget_eval_sources()
            for forget_source_idx in forget_sources:
                eval_stage = (
                    stage
                    if len(forget_sources) == 1
                    else f"{stage}__forget_src_{forget_source_idx}"
                )
                eval_dir = os.path.join(
                    self.args.output_dir,
                    "eval_rounds",
                    f"round_{round_idx + 1}",
                    eval_stage,
                )
                os.makedirs(eval_dir, exist_ok=True)

                summary_file = os.path.join(eval_dir, "TOFU_SUMMARY.json")
                summary_exists = os.path.isfile(summary_file) and os.path.getsize(summary_file) > 0
                if summary_exists and not force_eval:
                    logger.info(
                        f"Skipping eval for stage {eval_stage} at round {round_idx + 1}, existing summary: {summary_file}"
                    )
                    continue

                try:
                    self._set_eval_indices(stage, forget_source_idx)
                except Exception as exc:
                    logger.warning(
                        f"Failed to set eval indices for stage {eval_stage}: {exc}"
                    )
                    continue

                try:
                    self.evaluator.evaluate(
                        model=model,
                        tokenizer=self.tokenizer,
                        template_args=self.template_args,
                        output_dir=eval_dir,
                        overwrite=overwrite,
                    )
                except Exception as exc:
                    logger.warning(f"Evaluation failed at {eval_stage} (round {round_idx}): {exc}")
        finally:
            # Move model back to CPU if it was originally on CPU (to save GPU memory)
            if torch.cuda.is_available() and model_was_on_cpu:
                model = model.cpu()
                torch.cuda.empty_cache()

    def _copy_eval_stage_outputs(self, src_dir: str, dest_dir: str) -> bool:
        if not os.path.isdir(src_dir):
            return False
        src_summary = os.path.join(src_dir, "TOFU_SUMMARY.json")
        if not os.path.isfile(src_summary):
            return False

        os.makedirs(dest_dir, exist_ok=True)
        copied = False
        for fname in os.listdir(src_dir):
            src = os.path.join(src_dir, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dest_dir, fname))
                copied = True
        return copied

    def _reuse_server_post_eval(self, round_idx: int, target_stage: str) -> bool:
        """
        Reuse previous round server_post_agg eval as current pre_train eval.
        """
        if self.evaluator is None or round_idx == 0:
            return False
        copied = False
        forget_sources = self._forget_eval_sources()
        for forget_source_idx in forget_sources:
            src_stage = (
                "server_post_agg"
                if len(forget_sources) == 1
                else f"server_post_agg__forget_src_{forget_source_idx}"
            )
            dest_stage = (
                target_stage
                if len(forget_sources) == 1
                else f"{target_stage}__forget_src_{forget_source_idx}"
            )
            prev_dir = os.path.join(
                self.args.output_dir,
                "eval_rounds",
                f"round_{round_idx}",
                src_stage,
            )
            dest_dir = os.path.join(
                self.args.output_dir,
                "eval_rounds",
                f"round_{round_idx + 1}",
                dest_stage,
            )
            copied = self._copy_eval_stage_outputs(prev_dir, dest_dir) or copied
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
        copied = False
        forget_sources = self._forget_eval_sources()
        for forget_source_idx in forget_sources:
            src_stage = (
                "server_pre_agg"
                if len(forget_sources) == 1
                else f"server_pre_agg__forget_src_{forget_source_idx}"
            )
            dest_stage = (
                target_stage
                if len(forget_sources) == 1
                else f"{target_stage}__forget_src_{forget_source_idx}"
            )
            src_dir = os.path.join(
                self.args.output_dir,
                "eval_rounds",
                f"round_{round_idx + 1}",
                src_stage,
            )
            dest_dir = os.path.join(
                self.args.output_dir,
                "eval_rounds",
                f"round_{round_idx + 1}",
                dest_stage,
            )
            copied = self._copy_eval_stage_outputs(src_dir, dest_dir) or copied
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
        use_gpu_for_agg = self.server_device.type == "cuda" and torch.cuda.is_available()
        agg_device = self.server_device if use_gpu_for_agg else torch.device("cpu")
        
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
        strategy = self._normalize_save_strategy(getattr(self.args, "save_strategy", None))
        is_epoch = strategy == SaveStrategy.EPOCH
        if not is_epoch and round_idx != self.global_rounds - 1:
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

    def _resume_from_existing_rounds(self) -> int:
        """
        Locate the latest non-empty round_* under output_dir and warm start from it.
        Returns the number of completed rounds (0-index for loop start).
        """
        latest_round = self._find_latest_round(self.args.output_dir, self.global_rounds)
        if latest_round is None:
            return 0

        try:
            self._load_round_model(latest_round)
            logger.info(f"Loaded checkpoint from round_{latest_round} as resume baseline")
            return latest_round
        except Exception as exc:
            logger.warning(f"Failed to load round_{latest_round} checkpoint, restart from scratch: {exc}")
            return 0

    @staticmethod
    def _find_latest_round(base_dir: str, max_rounds: int) -> Optional[int]:
        for r in range(max_rounds, 0, -1):
            rdir = os.path.join(base_dir, f"round_{r}")
            if os.path.isdir(rdir) and os.listdir(rdir):
                return r
        return None

    def _load_round_model(self, round_idx: int):
        round_dir = os.path.join(self.args.output_dir, f"round_{round_idx}")
        if self.is_peft:
            state_dict = self._load_peft_state(round_dir, f"round_{round_idx}")
            set_peft_model_state_dict(self.model, state_dict)
        else:
            weight_path = os.path.join(round_dir, "pytorch_model.bin")
            if not os.path.isfile(weight_path):
                raise FileNotFoundError(f"pytorch_model.bin not found in {round_dir}")
            state_dict = torch.load(weight_path, map_location="cpu")
            self.model.load_state_dict(state_dict)
        self.model = self.model.cpu()

    def _load_peft_state(self, base_path: str, context: str):
        """
        Load a PEFT adapter state; prefer adapter_model.safetensors, fallback to adapter_model.bin.
        """
        if not os.path.isdir(base_path):
            raise FileNotFoundError(f"{context} path not found: {base_path}")

        bin_path = os.path.join(base_path, "adapter_model.bin")
        safetensors_path = os.path.join(base_path, "adapter_model.safetensors")

        if os.path.isfile(safetensors_path):
            try:
                from safetensors.torch import load_file
            except Exception as exc:
                raise ImportError(f"Failed to import safetensors for {context}: {exc}") from exc
            try:
                return load_file(safetensors_path, device="cpu")
            except Exception as exc:
                raise RuntimeError(f"Failed to load adapter_model.safetensors for {context}: {exc}") from exc

        if os.path.isfile(bin_path):
            try:
                return torch.load(bin_path, map_location="cpu")
            except Exception as exc:
                raise RuntimeError(f"Failed to load adapter_model.bin for {context}: {exc}") from exc

        raise FileNotFoundError(
            f"adapter_model.[safetensors|bin] not found in {base_path} for {context}"
        )
