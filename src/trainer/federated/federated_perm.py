import torch
import logging
from copy import deepcopy
from typing import Dict, List, Any, Optional
import os
import shutil
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from trainer.base import FinetuneTrainer
from torch.utils.data import Dataset
from trainer.federated.federated_utils import (
    FedAvg,
    FedAvgM,
    FedAdagrad,
    FedYogi,
    FedAdam,
    apply_inverse_permutation,
    apply_adversarial_filter,
    apply_permutation,
    generate_permutation_key,
)
from transformers.trainer_utils import IntervalStrategy, SaveStrategy
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from trainer.unlearn.grad_diff import GradDiff
from transformers import default_data_collator
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

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
        filter_alpha=0.5,
        aggregation_lr=1.0,
        enable_permutation=True,
        enable_revocation=True,
        revocation_round=None,
        unlearn_epochs: Optional[int] = None,
        train_on: str = "full",
        **kwargs
    ):
        self.federated_dataset = train_dataset if isinstance(train_dataset, dict) else None
        self.train_on = train_on
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
        self.filter_alpha = filter_alpha
        self.aggregation_lr = aggregation_lr
        self.enable_permutation = enable_permutation
        self.enable_revocation = enable_revocation
        self.revocation_round = revocation_round
        self.unlearn_epochs = unlearn_epochs
        self.unlearn_trainer_cls = kwargs.get("unlearn_trainer_cls")
        self.enable_pre_eval = os.environ.get("ENABLE_PRE_EVAL", "false").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.base_learning_rate = getattr(self.args, "learning_rate", None)
        self.recovery_learning_rate = kwargs.get("recovery_lr", None)
        self.kwargs = kwargs

        # 轮次编排：默认前 4 轮正常训练，3 轮定向遗忘，剩余轮次恢复正常
        self.normal_rounds_before_forget = kwargs.get("normal_rounds_before_forget", 4)
        self.forget_round_count = kwargs.get("forget_round_count", 3)
        provided_forget_rounds = kwargs.get("forget_rounds", None)
        if provided_forget_rounds is not None:
            self.forget_rounds = set(provided_forget_rounds)
        else:
            start = self.normal_rounds_before_forget
            end = min(self.global_rounds, start + self.forget_round_count)
            self.forget_rounds = set(range(start, end))
        self.recovery_round_start = kwargs.get(
            "recovery_round_start",
            (max(self.forget_rounds) + 1) if self.forget_rounds else None,
        )
        # C 类客户端中用于评测的一个索引（可指定），默认取第一个非 A/B
        self.eval_client_c_idx = kwargs.get("eval_client_c_idx", None)
        # 目标客户端遗忘阶段：loss = CE_current + lambda * CE_prev_post(B)
        self.target_prev_ce_enabled = kwargs.get("target_prev_ce_enabled", False)
        self.target_prev_ce_weight = float(kwargs.get("target_prev_ce_weight", 1.0))
        self.prev_target_model_path = kwargs.get("prev_target_model_path", None)
        self.prev_target_model_paths = kwargs.get("prev_target_model_paths", None)
        self.prev_target_models_post = {}
        self.unlearn_base_full_rounds = kwargs.get("unlearn_base_full_rounds", None)
        self.unlearn_base_offset = kwargs.get("unlearn_base_offset", None)
        self.unlearn_base_full_dir = kwargs.get("unlearn_base_full_dir", None)
        self.source_to_targets = self._resolve_source_to_targets(
            kwargs.get("source_to_targets", None),
            revoker_client_idx=revoker_client_idx,
            target_client_idx=target_client_idx,
        )
        self.source_client_indices = sorted(self.source_to_targets.keys())
        self.target_to_sources = self._invert_source_to_targets(self.source_to_targets)
        self.target_client_indices = sorted(self.target_to_sources.keys())
        self.revoker_client_idx = (
            self.source_client_indices[0] if self.source_client_indices else revoker_client_idx
        )
        self.target_client_idx = (
            self.target_client_indices[0] if self.target_client_indices else target_client_idx
        )

        self.revocation_status = False
        self.filter_F_A = None
        self.source_filters = {}
        self.target_filters = {}
        self.permutation_keys = {}
        self.client_models = []
        self.client_data_sizes = []

        self.is_peft = isinstance(self.model, PeftModel)
        # 保存策略归一化：支持枚举直接传入，缺省时默认逐轮保存
        self.args.save_strategy = self._normalize_save_strategy(getattr(self.args, "save_strategy", None))
        # 单卡模式：训练/聚合/过滤全部使用同一设备
        self.train_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.server_device = self.train_device
        self.filter_device = self.train_device
        self.client_parallel_devices = self._resolve_device_pool(
            kwargs.get("client_parallel_devices", None),
            fallback_device=self.train_device,
        )
        self.filter_parallel_devices = self._resolve_device_pool(
            kwargs.get("filter_parallel_devices", None),
            fallback_device=self.filter_device,
        )
        self.parallel_client_workers = int(kwargs.get("parallel_client_workers", 0))
        self.parallel_filter_workers = int(kwargs.get("parallel_filter_workers", 0))
        self.fedadam_momentum = None
        self.fedadam_velocity = None
        self.fedyogi_momentum = None
        self.fedyogi_velocity = None
        self.fedavgm_momentum = None
        self.fedadagrad_velocity = None
        self.federated_args = {
            "beta1": kwargs.get("beta1", 0.9),
            "beta2": kwargs.get("beta2", 0.99),
            "epsilon": kwargs.get(
                "epsilon",
                self.aggregation_lr if self.aggregation_lr is not None else 1e-3,
            ),
            "tau": kwargs.get("tau", 1e-3),
            "momentum_factor": kwargs.get("momentum_factor", 0.9),
        }

        if not self.federated_dataset:
            logger.warning("No federated dataset provided!")
        else:
            for client_idx in range(num_clients):
                if client_idx not in self.federated_dataset:
                    logger.warning(f"Client {client_idx} has no data!")

        logger.info(f"FedPerm initialized with {num_clients} clients")
        logger.info(f"Source clients: {self.source_client_indices}")
        logger.info(f"Source to targets mapping: {self.source_to_targets}")
        logger.info(f"Target to sources mapping: {self.target_to_sources}")
        logger.info(f"Model type - PEFT: {self.is_peft}")
        logger.info(f"Aggregation strategy: {self.aggregation_strategy}")
        logger.info(f"Revocation enabled: {self.enable_revocation}, permutation enabled: {self.enable_permutation}")
        logger.info(f"Client parallel devices: {self.client_parallel_devices}, workers={self.parallel_client_workers}")
        logger.info(f"Filter parallel devices: {self.filter_parallel_devices}, workers={self.parallel_filter_workers}")
        logger.info(f"Forget rounds: {sorted(self.forget_rounds)}, recovery starts at: {self.recovery_round_start}")
        if self.prev_target_model_path:
            logger.info(f"Prev target model path: {self.prev_target_model_path}")
        if self.unlearn_base_full_rounds is not None and self.unlearn_base_offset is not None:
            logger.info(
                "Unlearn base schedule: full_rounds=%s, offset=%s, full_dir=%s",
                self.unlearn_base_full_rounds,
                self.unlearn_base_offset,
                self.unlearn_base_full_dir,
            )

    def _load_topology_metadata(self):
        split_dir = os.environ.get(
            "FEDPERM_SPLIT_DIR", os.path.join(self.args.output_dir, "client_splits")
        )
        topology_path = os.path.join(split_dir, "fedperm_topology.json")
        if not os.path.isfile(topology_path):
            return None
        try:
            with open(topology_path, "r") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning(f"Failed to load topology metadata from {topology_path}: {exc}")
            return None

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

    def _resolve_source_to_targets(self, source_to_targets, revoker_client_idx: int, target_client_idx: int):
        topology = source_to_targets
        if topology is None:
            metadata = self._load_topology_metadata()
            if metadata:
                topology = metadata.get("source_to_targets", None)
        if topology is None:
            return {int(revoker_client_idx): [int(target_client_idx)]}

        normalized = {}
        for raw_src, raw_targets in topology.items():
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

    @staticmethod
    def _invert_source_to_targets(source_to_targets):
        target_to_sources = {}
        for src_idx, target_indices in source_to_targets.items():
            for tgt_idx in target_indices:
                target_to_sources.setdefault(int(tgt_idx), []).append(int(src_idx))
        for tgt_idx in list(target_to_sources.keys()):
            target_to_sources[tgt_idx] = sorted(dict.fromkeys(target_to_sources[tgt_idx]))
        return target_to_sources

    def _resolve_prev_target_model_paths(self):
        resolved = {}
        if hasattr(self.prev_target_model_paths, "items"):
            for key, value in self.prev_target_model_paths.items():
                resolved[int(key)] = value
        elif self.prev_target_model_path and len(self.target_client_indices) == 1:
            resolved[self.target_client_indices[0]] = self.prev_target_model_path
        return resolved

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

    def _resolve_unlearn_base_path(self, round_idx: int):
        """
        Map unlearn round index to a checkpoint path across full+unlearn rounds.
        target_round = full_rounds + (t) - offset - 1, where t = round_idx + 1.
        If target_round <= full_rounds: use full_dir/round_{target_round}
        else use unlearn output_dir/round_{target_round - full_rounds}.
        """
        if (
            self.unlearn_base_full_rounds is None
            or self.unlearn_base_offset is None
            or not self.unlearn_base_full_dir
        ):
            raise ValueError(
                "unlearn base schedule is not configured; require "
                "unlearn_base_full_rounds, unlearn_base_offset, unlearn_base_full_dir"
            )
        t = round_idx + 1
        target_round = self.unlearn_base_full_rounds + t - self.unlearn_base_offset - 1
        if target_round < 1:
            raise ValueError(
                f"unlearn base target_round < 1 (round_idx={round_idx}, "
                f"full_rounds={self.unlearn_base_full_rounds}, offset={self.unlearn_base_offset})"
            )
        if target_round <= self.unlearn_base_full_rounds:
            return os.path.join(self.unlearn_base_full_dir, f"round_{target_round}")
        unlearn_round = target_round - self.unlearn_base_full_rounds
        return os.path.join(self.args.output_dir, f"round_{unlearn_round}")

    def train(self, **kwargs):
        """Main training loop for FedPerm"""
        if not self.federated_dataset:
            raise ValueError("Federated dataset is not properly set up")
        
        self.model = self.model.cpu()
        self.model = deepcopy(self.model)
        start_round = self._resume_from_existing_rounds()
        if start_round >= self.global_rounds:
            logger.info("All required rounds already exist; skip FedPerm loop.")
            return None
        if start_round > 0:
            logger.info(f"Resuming FedPerm from round {start_round + 1}")
        # Try to preload target client's previous post model for prev-CE
        if not self.prev_target_models_post:
            self._preload_prev_target_models()
        
        self._should_trigger_revocation = False

        for round_idx in range(start_round, self.global_rounds):
            logger.info(f"Starting global round {round_idx + 1}/{self.global_rounds}")

            # Decide current阶段：遗忘/恢复/正常
            is_forget_round = self.enable_revocation and (round_idx in self.forget_rounds)
            recovery_due = (
                self.recovery_round_start is not None
                and round_idx >= self.recovery_round_start
                and not is_forget_round
            )
            # 动态切换恢复阶段学习率（若指定）
            if recovery_due and self.recovery_learning_rate is not None:
                if getattr(self.args, "learning_rate", None) != self.recovery_learning_rate:
                    logger.info(f"Switch learning_rate to recovery_lr={self.recovery_learning_rate}")
                self.args.learning_rate = self.recovery_learning_rate
            else:
                self.args.learning_rate = self.base_learning_rate

            # Activate / deactivate filter as needed
            if is_forget_round:
                self._trigger_revocation(round_idx)
            elif recovery_due and self.revocation_status:
                self._deactivate_revocation(round_idx)

            # Phase 1: Client Selection & Distribution
            logger.info("Phase 1: Client Selection & Distribution")
            self.client_models = [deepcopy(self.model) for _ in range(self.num_clients)]
            self.client_data_sizes = []
            
            # Get global model state (extract without moving to CPU first)
            if self.is_peft:
                global_state = get_peft_model_state_dict(self.model)
            else:
                global_state = self.model.state_dict()

            # Server-side eval before broadcasting (聚合前评测)
            if self.enable_pre_eval:
                self._maybe_evaluate_model(
                    model=self.model,
                    round_idx=round_idx,
                    stage="server_pre_agg",
                )
            
            # Distribute models to clients
            self.client_data_sizes = []
            for client_idx in range(self.num_clients):
                if client_idx not in self.federated_dataset:
                    logger.warning(f"Skipping client {client_idx} as no data is available")
                    self.client_data_sizes.append(0)
                    continue
                
                client_model = self.client_models[client_idx]
                client_container = self.federated_dataset[client_idx]
                client_dataset = self._select_train_dataset(client_container)
                
                # Apply filtering for target client B if遗忘激活；其他客户端直接用全局模型
                filtered_state = global_state
                if self.revocation_status and client_idx in self.target_filters:
                    logger.info(f"Applying adversarial filter for target client {client_idx}")
                    target_filter = self.target_filters.get(client_idx)
                    if target_filter is not None:
                        filtered_state = apply_adversarial_filter(
                            global_state, target_filter, alpha=1.0
                        )
                        if self.enable_permutation:
                            predicate = self._is_permutable_parameter
                            perm_key = generate_permutation_key(
                                filtered_state, predicate=predicate
                            )
                            self.permutation_keys.setdefault(round_idx, {})[client_idx] = perm_key
                            filtered_state = apply_permutation(filtered_state, perm_key)
                    else:
                        logger.warning("Filter F_A is None, skipping filter application")

                if self.is_peft:
                    set_peft_model_state_dict(client_model, filtered_state)
                else:
                    client_model.load_state_dict(filtered_state)

                if self.revocation_status and client_idx in self.target_filters:
                    logger.info(f"Client {client_idx} received filtered model")
                else:
                    logger.info(f"Client {client_idx} received normal model")
                
                client_data_size = len(client_dataset) if client_dataset is not None else 0
                self.client_data_sizes.append(client_data_size)

                # 训练前评测（逐客户端）：不再复用 server 评测结果
                if self.enable_pre_eval and self._should_eval_client(client_idx):
                    self._maybe_evaluate_model(
                        model=client_model,
                        round_idx=round_idx,
                        stage=f"client{client_idx}_pre_train",
                    )
            
            # Phase 2: Local Training (Parallel)
            logger.info("Phase 2: Local Training")
            client_initial_states = []  # Store initial states for update computation
            client_updated_states = []  # Store updated states
            
            # 收集每个客户端的训练任务，后续并行执行
            client_initial_states = [None] * self.num_clients
            client_updated_states = [None] * self.num_clients
            client_task_specs = []
            for client_idx in range(self.num_clients):
                if client_idx not in self.federated_dataset:
                    continue

                client_model = self.client_models[client_idx]
                client_container = self.federated_dataset[client_idx]
                client_dataset = self._select_train_dataset(client_container)

                # Store initial state (extract without moving to CPU first)
                if self.is_peft:
                    initial_state = get_peft_model_state_dict(client_model)
                else:
                    initial_state = client_model.state_dict()
                client_initial_states[client_idx] = initial_state

                logger.info(f"Queue training for client {client_idx}")
                prev_teacher = None
                if (
                    self.revocation_status
                    and self.target_prev_ce_enabled
                    and client_idx in self.target_filters
                ):
                    prev_teacher = self.prev_target_models_post.get(client_idx)
                    if prev_teacher is None:
                        logger.warning(
                            "Prev target model missing for target client %s; skipping prev CE this round.",
                            client_idx,
                        )
                        prev_teacher = None

                client_task_specs.append(
                    {
                        "client_idx": client_idx,
                        "client_model": deepcopy(client_model).cpu(),
                        "client_dataset": client_dataset,
                        "prev_teacher": deepcopy(prev_teacher).cpu() if prev_teacher is not None else None,
                    }
                )

            def _train_worker(task_spec, device):
                return self._train_client_task(
                    task_spec["client_idx"],
                    task_spec["client_model"],
                    task_spec["client_dataset"],
                    task_spec["prev_teacher"],
                    device,
                )

            for client_idx, trained_model, updated_state in self._run_parallel_tasks(
                client_task_specs,
                _train_worker,
                self.client_parallel_devices,
                self.parallel_client_workers,
                task_label="client-train",
            ):
                client_updated_states[client_idx] = updated_state
                self.client_models[client_idx] = trained_model

            # 训练后评测（串行，避免共享 evaluator 的并发问题）
            for client_idx in range(self.num_clients):
                trained_model = self.client_models[client_idx] if client_idx < len(self.client_models) else None
                if trained_model is None or not self._should_eval_client(client_idx):
                    continue
                self._maybe_evaluate_model(
                    model=trained_model,
                    round_idx=round_idx,
                    stage=f"client{client_idx}_post_train",
                )

            # Phase 3: Secure Aggregation
            logger.info("Phase 3: Secure Aggregation")
            self._aggregate_with_alignment(
                round_idx, 
                client_initial_states, 
                client_updated_states,
                global_state
            )

            # 服务器聚合后评测
            self._maybe_evaluate_model(
                model=self.model,
                round_idx=round_idx,
                stage="server_post_agg",
            )
            
            logger.info(f"Completed global round {round_idx + 1}/{self.global_rounds}")

            self._save_round_outputs(round_idx)
        
        # 不再额外生成 final_* 目录，依赖 round_N 检查点
        return None

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
        match = re.match(r"client(\d+)_", stage)
        if stage.startswith("server"):
            retain_idx_path = os.path.join(split_dir, "global_retain_indices.json")
        elif match:
            client_idx = int(match.group(1))
            retain_idx_path = os.path.join(split_dir, f"client{client_idx}_retain_indices.json")
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

    def _maybe_evaluate_model(self, model, round_idx: int, stage: str):
        """
        Run evaluator (if provided) and save SUMMARY.json for the given stage.
        Forget metrics are evaluated separately for each source client.
        """
        if self.evaluator is None:
            return
        if (stage.startswith("server_pre") or stage.endswith("_pre_train")) and not self.enable_pre_eval:
            logger.info(
                f"Skipping pre eval for stage {stage} due to ENABLE_PRE_EVAL={self.enable_pre_eval}"
            )
            return

        skip_rounds_env = os.environ.get("FEDPERM_SKIP_EVAL_ROUNDS", "")
        if skip_rounds_env:
            try:
                skip_rounds = {int(x) for x in skip_rounds_env.split(",") if x.strip() != ""}
                if round_idx in skip_rounds:
                    logger.info(
                        f"Skipping evaluation at round {round_idx + 1} ({stage}) due to FEDPERM_SKIP_EVAL_ROUNDS"
                    )
                    return
            except Exception as exc:
                logger.warning(f"Failed to parse FEDPERM_SKIP_EVAL_ROUNDS='{skip_rounds_env}': {exc}")

        forget_sources = self.source_client_indices or [self.revoker_client_idx]
        force_eval = os.environ.get("FORCE_FEDPERM_EVAL", "0") in ("1", "true", "True")
        overwrite = True
        if hasattr(self.evaluator, "eval_cfg"):
            overwrite = getattr(self.evaluator.eval_cfg, "overwrite", True)

        model_was_on_cpu = next(model.parameters()).device.type == "cpu"
        if torch.cuda.is_available() and model_was_on_cpu:
            device = torch.device("cuda")
            model = model.to(device)
            logger.info(f"Model moved to GPU device {device} for evaluation at stage {stage}")

        try:
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
                summary_exists = (
                    os.path.isfile(summary_file) and os.path.getsize(summary_file) > 0
                )
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
                    logger.warning(
                        f"Evaluation failed at {eval_stage} (round {round_idx}): {exc}"
                    )
        finally:
            if torch.cuda.is_available() and model_was_on_cpu:
                model = model.cpu()
                torch.cuda.empty_cache()

    def _reuse_server_post_eval(self, round_idx: int, target_stage: str, client_idx: int) -> bool:
        """
        Reuse previous round server_post_agg eval as current client pre_train eval
        when the broadcast model is identical (i.e., non-target or no filter applied).
        """
        if self.evaluator is None or round_idx == 0:
            return False
        # 遗忘阶段或已持有过滤器时，目标客户端需要重新评测
        if client_idx in self.target_to_sources and (
            self.revocation_status or self.target_filters.get(client_idx) is not None
        ):
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

    def _should_eval_client(self, client_idx: int) -> bool:
        """
        评测所有客户端，确保 A/B/C 的 pre/post 都被记录。
        """
        return self.evaluator is not None

    def _trigger_revocation(self, round_idx: int):
        """
        触发授权撤销：在每个源客户端的遗忘数据上训练出一个“已遗忘”的模型，
        再组合成目标客户端各自的过滤器。
        """
        logger.info(f"=== Triggering Revocation at Round {round_idx} ===")
        logger.info(f"Optimizing source filters for source clients {self.source_client_indices}")

        class ForgetOnlyDataset(Dataset):
            def __init__(self, base):
                self.base = base

            def __len__(self):
                return len(self.base)

            def __getitem__(self, idx):
                sample = self.base[idx]
                return {
                    "forget": {"forget": sample},
                    "retain": {"retain": sample},  # 占位以兼容 alpha>0 的情况
                }

        unlearn_args = deepcopy(self.args)
        if self.unlearn_epochs is not None:
            unlearn_args.num_train_epochs = self.unlearn_epochs
            logger.info(f"Using {self.unlearn_epochs} epochs for unlearn stage")
        else:
            logger.info(f"Using default {unlearn_args.num_train_epochs} epochs for unlearn stage")
        # 遗忘阶段固定单卡
        if self.filter_device.type == "cuda":
            object.__setattr__(unlearn_args, "_n_gpu", 1)
        else:
            object.__setattr__(unlearn_args, "_n_gpu", 0)

        base_path = self._resolve_unlearn_base_path(round_idx)
        if not os.path.isdir(base_path):
            raise FileNotFoundError(
                f"unlearn base path does not exist: {base_path}"
            )
        if self.is_peft:
            checkpoint_state = self._load_peft_state(base_path, "unlearn init model")
        else:
            weight_path = os.path.join(base_path, "pytorch_model.bin")
            if not os.path.isfile(weight_path):
                raise FileNotFoundError(
                    f"pytorch_model.bin not found at unlearn base path={base_path}"
                )
            checkpoint_state = torch.load(weight_path, map_location="cpu")

        base_state = {
            k: (v.detach().clone().cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in checkpoint_state.items()
        }

        from trainer import TRAINER_REGISTRY
        trainer_cls = TRAINER_REGISTRY.get(self.unlearn_trainer_cls, None)
        if trainer_cls is None:
            logger.warning(
                f"Unrecognized unlearn_trainer_cls={self.unlearn_trainer_cls}; skip revocation for this round"
            )
            return

        self.source_filters = {}
        source_task_specs = []
        for source_client_idx in self.source_client_indices:
            if source_client_idx not in self.federated_dataset:
                logger.warning(
                    f"Source client {source_client_idx} has no data, skipping source filter construction"
                )
                continue

            source_dataset = self.federated_dataset[source_client_idx]
            forget_dataset = (
                source_dataset.forget
                if hasattr(source_dataset, "forget")
                else source_dataset
            )
            if forget_dataset is None or len(forget_dataset) == 0:
                logger.warning(
                    f"Source client {source_client_idx} has empty forget dataset, skipping source filter construction"
                )
                continue

            source_task_specs.append(
                {
                    "source_client_idx": source_client_idx,
                    "forget_dataset": forget_dataset,
                    "checkpoint_state": checkpoint_state,
                    "base_state": base_state,
                    "base_path": base_path,
                    "trainer_cls": trainer_cls,
                    "unlearn_args": deepcopy(unlearn_args),
                    "forget_dataset_cls": ForgetOnlyDataset,
                }
            )

        def _source_filter_worker(task_spec, device):
            source_client_idx = task_spec["source_client_idx"]
            forget_only_dataset = task_spec["forget_dataset_cls"](task_spec["forget_dataset"])
            local_unlearn_args = deepcopy(task_spec["unlearn_args"])
            if device.type == "cuda":
                object.__setattr__(local_unlearn_args, "_n_gpu", 1)
            else:
                object.__setattr__(local_unlearn_args, "_n_gpu", 0)

            unlearn_init_model = deepcopy(self.model)
            checkpoint_state_local = task_spec["checkpoint_state"]
            if self.is_peft:
                set_peft_model_state_dict(unlearn_init_model, checkpoint_state_local)
            else:
                unlearn_init_model.load_state_dict(checkpoint_state_local)
            logger.info(
                "Initialized unlearn model for source client %s from %s on device %s",
                source_client_idx,
                task_spec["base_path"],
                device,
            )
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.set_device(device)
                unlearn_init_model = unlearn_init_model.to(device)

            trainer_kwargs = dict(
                model=unlearn_init_model,
                train_dataset=forget_only_dataset,
                tokenizer=self.tokenizer,
                data_collator=self.data_collator,
                args=local_unlearn_args,
                evaluator=None,
                template_args=self.template_args,
            )
            if task_spec["trainer_cls"] is GradDiff:
                trainer_kwargs["alpha"] = 0.0

            unlearn_trainer = task_spec["trainer_cls"](**trainer_kwargs)
            logger.info(
                "Start unlearning training on source client %s data (%s) on device %s...",
                source_client_idx,
                task_spec["trainer_cls"].__name__,
                device,
            )
            unlearn_trainer.train()
            logger.info("Unlearning training finished for source client %s.", source_client_idx)
            unlearn_model = unlearn_trainer.model.cpu()

            if self.is_peft:
                unlearn_state = get_peft_model_state_dict(unlearn_model)
            else:
                unlearn_state = unlearn_model.state_dict()
            device_base_state = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in task_spec["base_state"].items()
            }
            unlearn_state = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in unlearn_state.items()
            }

            source_filter = {}
            missing_keys = 0
            for key, param in device_base_state.items():
                if key in unlearn_state and isinstance(param, torch.Tensor):
                    source_filter[key] = (
                        self.filter_alpha * (
                        param - unlearn_state[key]
                        )
                    ).detach().clone().cpu()
                else:
                    source_filter[key] = None
                    missing_keys += 1
            if missing_keys > 0:
                logger.warning(
                    "%s keys in base_state not found in unlearn_state or not Tensor for source client %s.",
                    missing_keys,
                    source_client_idx,
                )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return source_client_idx, source_filter

        for source_client_idx, source_filter in self._run_parallel_tasks(
            source_task_specs,
            _source_filter_worker,
            self.filter_parallel_devices,
            self.parallel_filter_workers,
            task_label="source-filter",
        ):
            self.source_filters[source_client_idx] = source_filter

        self.target_filters = {}
        for target_client_idx, source_indices in self.target_to_sources.items():
            combined_filter = {}
            for src_idx in source_indices:
                source_filter = self.source_filters.get(src_idx)
                if source_filter is None:
                    continue
                for key, value in source_filter.items():
                    if value is None:
                        combined_filter.setdefault(key, None)
                        continue
                    if combined_filter.get(key) is None:
                        combined_filter[key] = value.detach().clone()
                    else:
                        combined_filter[key] = combined_filter[key] + value
            if combined_filter:
                self.target_filters[target_client_idx] = combined_filter

        if not self.target_filters:
            logger.warning("No target filters were constructed; revocation remains inactive")
            self.revocation_status = False
            self.filter_F_A = None
            return

        self.filter_F_A = self.target_filters.get(self.target_client_idx)
        logger.info("Constructed source filters for clients: %s", sorted(self.source_filters.keys()))
        logger.info("Constructed target filters for clients: %s", sorted(self.target_filters.keys()))
        self.revocation_status = True
        logger.info("RevocationStatus set to True")

    def _deactivate_revocation(self, round_idx: int):
        """
        撤销遗忘请求，恢复正常联邦训练。
        """
        logger.info(f"=== Deactivating revocation at Round {round_idx} ===")
        self.revocation_status = False
        self.filter_F_A = None
        self.source_filters = {}
        self.target_filters = {}
        self.permutation_keys = {}
        logger.info("RevocationStatus set to False, filter cleared")

    def _select_train_dataset(self, client_data):
        """
        根据 train_on 选择客户端使用的数据 split；若无 split 属性则直接返回。
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

    def train_client_model(
        self,
        client_model,
        client_dataset,
        prev_teacher=None,
        device=None,
    ):
        """Train client model locally"""
        # Move client model到目标设备
        if device is not None:
            target_device = device if isinstance(device, torch.device) else torch.device(device)
        else:
            target_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        if target_device is not None and target_device.type == "cuda":
            torch.cuda.set_device(target_device)
            client_model = client_model.to(target_device)
            logger.info(f"Client model moved to device: {target_device}")
        
        trainer = FinetuneTrainer(
            model=client_model,
            train_dataset=client_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            args=self.args,
            evaluator=self.evaluator,
            template_args=self.template_args,
        )

        trained_model = None
        if (
            prev_teacher is not None
            and self.target_prev_ce_enabled
            and float(self.target_prev_ce_weight) != 0
        ):
            trained_model = self._train_target_with_prev_teacher(
                trainer=trainer,
                student_model=client_model,
                teacher_model=deepcopy(prev_teacher),
                device=self.train_device if device is None else device,
            )
        if trained_model is None:
            trainer.train()
            trained_model = trainer.model.cpu()
        if target_device is not None and target_device.type == "cuda":
            torch.cuda.empty_cache()
        return trained_model

    def _train_client_task(
        self,
        client_idx: int,
        client_model,
        client_dataset,
        prev_teacher,
        device: torch.device,
    ):
        trained_model = self.train_client_model(
            client_model,
            client_dataset,
            prev_teacher=prev_teacher,
            device=device,
        )
        if self.is_peft:
            updated_state = get_peft_model_state_dict(trained_model)
        else:
            updated_state = trained_model.state_dict()
        return client_idx, trained_model, updated_state

    def _move_batch_to_device(self, batch, device):
        """Recursively move tensors in a batch to the specified device."""
        if torch.is_tensor(batch):
            return batch.to(device)
        if isinstance(batch, dict):
            return {k: self._move_batch_to_device(v, device) for k, v in batch.items()}
        if isinstance(batch, list):
            return [self._move_batch_to_device(v, device) for v in batch]
        return batch

    def _train_target_with_prev_teacher(self, trainer, student_model, teacher_model, device):
        """
        使用“先更新 teacher，再更新 student”的交替方式训练目标客户端。
        teacher 基于真实标签优化，随后固定 teacher，仅更新 student。
        """
        student_model = student_model.to(device)
        teacher_model = teacher_model.to(device)
        student_model.train()
        teacher_model.train()

        dataloader = trainer.get_train_dataloader()
        args = trainer.args
        grad_accum = max(1, getattr(args, "gradient_accumulation_steps", 1))
        num_epochs = int(getattr(args, "num_train_epochs", 1))
        max_grad_norm = getattr(args, "max_grad_norm", None)
        teacher_weight = float(self.target_prev_ce_weight)

        teacher_opt = AdamW(
            teacher_model.parameters(),
            lr=args.learning_rate,
            weight_decay=getattr(args, "weight_decay", 0.0),
        )
        student_opt = AdamW(
            student_model.parameters(),
            lr=args.learning_rate,
            weight_decay=getattr(args, "weight_decay", 0.0),
        )

        teacher_opt.zero_grad(set_to_none=True)
        student_opt.zero_grad(set_to_none=True)

        for _ in range(num_epochs):
            step = -1
            for step, batch in enumerate(dataloader):
                batch = self._move_batch_to_device(batch, device)

                # Step 1: 更新 teacher（CE + 真实标签）
                teacher_loss = teacher_weight * teacher_model(**batch).loss / grad_accum
                teacher_loss.backward()
                if (step + 1) % grad_accum == 0:
                    if max_grad_norm is not None:
                        clip_grad_norm_(teacher_model.parameters(), max_grad_norm)
                    teacher_opt.step()
                    teacher_opt.zero_grad(set_to_none=True)

                # Step 2: 固定 teacher，更新 student
                teacher_model.eval()
                with torch.no_grad():
                    teacher_const_loss = teacher_weight * teacher_model(**batch).loss
                teacher_model.train()

                student_loss = student_model(**batch).loss
                # 加上常数项以保持 loss 形式一致（对梯度无影响）
                student_loss = student_loss + teacher_const_loss.detach()
                student_loss = student_loss / grad_accum
                student_loss.backward()
                if (step + 1) % grad_accum == 0:
                    if max_grad_norm is not None:
                        clip_grad_norm_(student_model.parameters(), max_grad_norm)
                    student_opt.step()
                    student_opt.zero_grad(set_to_none=True)

            # flush 残余梯度（处理非整除 grad_accum 的情况）
            if step >= 0 and (step + 1) % grad_accum != 0:
                if max_grad_norm is not None:
                    clip_grad_norm_(teacher_model.parameters(), max_grad_norm)
                    clip_grad_norm_(student_model.parameters(), max_grad_norm)
                teacher_opt.step()
                student_opt.step()
                teacher_opt.zero_grad(set_to_none=True)
                student_opt.zero_grad(set_to_none=True)

        student_model = student_model.cpu()
        teacher_model = teacher_model.cpu()
        return student_model

    def _save_round_outputs(self, round_idx: int):
        """Persist server and target-client checkpoints for a given round."""
        skip_env = os.environ.get("FEDPERM_SKIP_SAVE_ROUNDS", "")
        skip_rounds = set()
        if skip_env:
            try:
                skip_rounds = {int(x) for x in skip_env.split(",") if x.strip() != ""}
            except Exception as exc:
                logger.warning(f"Failed to parse FEDPERM_SKIP_SAVE_ROUNDS='{skip_env}': {exc}")
        if round_idx in skip_rounds and round_idx != self.global_rounds - 1:
            logger.info(f"Skipping save at round {round_idx + 1} due to FEDPERM_SKIP_SAVE_ROUNDS")
            return

        save_path = f"{self.args.output_dir}/round_{round_idx + 1}"
        # 仅保存模型权重（PEFT 保存 adapter），不重复写 tokenizer，避免后续误用 round_* 目录作为 tokenizer 路径
        self.model = self.model.cpu()
        self._save_model_only(self.model, save_path)
        logger.info(f"Model saved at {save_path} (model only, tokenizer not duplicated)")

        self.prev_target_models_post = {}
        for target_client_idx in self.target_client_indices:
            if not (0 <= target_client_idx < len(self.client_models)):
                continue
            target_client_model = self.client_models[target_client_idx]
            if target_client_model is None:
                continue

            self.prev_target_models_post[target_client_idx] = deepcopy(target_client_model).cpu()
            target_client_save_path = (
                f"{self.args.output_dir}/round_{round_idx + 1}_target_client_{target_client_idx}"
            )
            os.makedirs(target_client_save_path, exist_ok=True)

            if self.is_peft:
                logger.info(
                    f"Saving target client {target_client_idx} PEFT model to {target_client_save_path} (model only)"
                )
                target_client_model = target_client_model.cpu()
                target_client_model.save_pretrained(
                    target_client_save_path, safe_serialization=False
                )
            else:
                target_client_model = target_client_model.cpu()
                torch.save(
                    target_client_model.state_dict(),
                    os.path.join(target_client_save_path, "pytorch_model.bin"),
                )

            logger.info(
                f"Target client {target_client_idx} model saved at {target_client_save_path}"
            )

    def _infer_batch_size(self, batch) -> int:
        """Infer batch size from the first tensor in batch."""
        if isinstance(batch, dict):
            for v in batch.values():
                size = self._infer_batch_size(v)
                if size is not None:
                    return size
        if isinstance(batch, list):
            for v in batch:
                size = self._infer_batch_size(v)
                if size is not None:
                    return size
        if torch.is_tensor(batch):
            return batch.size(0)
        return 1

    def _load_prev_target_model(self, model_path: str):
        """Load target client's previous post model for prev-CE warm start."""
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"prev_target_model_path does not exist: {model_path}")
        teacher = deepcopy(self.model).cpu()
        if self.is_peft:
            state_dict = self._load_peft_state(model_path, "prev target model")
            set_peft_model_state_dict(teacher, state_dict)
        else:
            weight_path = os.path.join(model_path, "pytorch_model.bin")
            if not os.path.isfile(weight_path):
                raise FileNotFoundError(f"pytorch_model.bin not found in {model_path}")
            state_dict = torch.load(weight_path, map_location="cpu")
            teacher.load_state_dict(state_dict)
        teacher.eval()
        logger.info(f"Loaded prev target model from {model_path}")
        return teacher

    def _preload_prev_target_models(self):
        for client_idx, model_path in self._resolve_prev_target_model_paths().items():
            try:
                self.prev_target_models_post[client_idx] = self._load_prev_target_model(model_path)
            except Exception as exc:
                logger.warning(
                    f"Failed to preload target client model for client {client_idx} from {model_path}: {exc}"
                )

    def _load_peft_state(self, base_path: str, context: str):
        """
        Load a PEFT adapter state; prefer adapter_model.safetensors, fallback to adapter_model.bin.
        Missing files or load failures raise to surface issues early.
        """
        if not os.path.isdir(base_path):
            raise FileNotFoundError(f"{context} path not found: {base_path}")

        bin_path = os.path.join(base_path, "adapter_model.bin")
        safetensors_path = os.path.join(base_path, "adapter_model.safetensors")

        # Prefer safetensors if present
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

    def _save_model_only(self, model, output_dir: str):
        """
        Save only model weights (adapter for PEFT), skipping tokenizer to avoid
        accidentally reusing round_* dirs as tokenizer path.
        """
        os.makedirs(output_dir, exist_ok=True)
        if self.is_peft:
            model.save_pretrained(output_dir, safe_serialization=False)
        else:
            torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

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
            self._load_prev_target_from_round(latest_round)
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

    def _load_prev_target_from_round(self, round_idx: int):
        """
        Try to hydrate previous target models from round-specific target checkpoints.
        """
        loaded_models = {}
        candidate_targets = (
            self.target_client_indices if self.target_client_indices else list(range(self.num_clients))
        )
        for client_idx in candidate_targets:
            target_dir = os.path.join(
                self.args.output_dir,
                f"round_{round_idx}_target_client_{client_idx}",
            )
            if not os.path.isdir(target_dir):
                continue
            try:
                loaded_models[client_idx] = self._load_prev_target_model(target_dir)
            except Exception as exc:
                logger.warning(f"Failed to load target client model from {target_dir}: {exc}")
        self.prev_target_models_post = loaded_models


    def _aggregate_with_alignment(self, round_idx: int, initial_states: List, updated_states: List, global_state: Dict):
        """
        Secure aggregation with re-alignment for filtered target clients.
        """
        logger.info("Aggregating client models with alignment")

        total_samples = sum(self.client_data_sizes)
        if total_samples == 0:
            logger.warning("No samples available for aggregation")
            return

        client_weights = [size / total_samples for size in self.client_data_sizes]
        logger.info(f"Client sample sizes: {self.client_data_sizes}")
        logger.info(f"Client weights: {[f'{w:.4f}' for w in client_weights]}")
        # Align weight vector length with expected client count to avoid index issues
        if len(client_weights) != self.num_clients:
            logger.warning(
                f"client_weights len {len(client_weights)} != num_clients {self.num_clients}; padding/truncating."
            )
            if len(client_weights) < self.num_clients:
                client_weights = client_weights + [0.0] * (self.num_clients - len(client_weights))
            else:
                client_weights = client_weights[: self.num_clients]

        # Determine target device for aggregation (prefer GPU if available)
        use_gpu_for_agg = self.server_device.type == "cuda" and torch.cuda.is_available()
        agg_device = self.server_device if use_gpu_for_agg else torch.device("cpu")
        
        # Move global_state to aggregation device if needed
        if use_gpu_for_agg:
            global_state = {
                k: v.to(agg_device) if isinstance(v, torch.Tensor) else v
                for k, v in global_state.items()
            }

        # Collect aligned client parameters (not parameter deltas)
        aligned_client_states = []
        for client_idx in range(self.num_clients):
            if updated_states[client_idx] is None:
                # No data for this client; contribute current global state (weight should be ~0)
                aligned_client_states.append(global_state)
                continue

            client_state = updated_states[client_idx]
            # Move to aggregation device if needed
            if use_gpu_for_agg:
                client_state = {
                    k: v.to(agg_device) if isinstance(v, torch.Tensor) else v
                    for k, v in client_state.items()
                }

            if (
                self.revocation_status
                and client_idx in self.target_filters
                and self.enable_permutation
                and round_idx in self.permutation_keys
                and client_idx in self.permutation_keys.get(round_idx, {})
            ):
                logger.info(f"Re-aligning parameters from target client {client_idx}")
                perm_key = self.permutation_keys[round_idx][client_idx]
                if use_gpu_for_agg:
                    perm_key = {
                        k: v.to(agg_device) if isinstance(v, torch.Tensor) else v
                        for k, v in perm_key.items()
                    }
                client_state = apply_inverse_permutation(client_state, perm_key)

            aligned_client_states.append(client_state)

        # Ensure weights and aligned states have the same length before FedAvg
        if len(client_weights) != len(aligned_client_states):
            logger.warning(
                f"Aligned client states ({len(aligned_client_states)}) and weights ({len(client_weights)}) length mismatch; fixing."
            )
            if len(client_weights) < len(aligned_client_states):
                client_weights = client_weights + [0.0] * (len(aligned_client_states) - len(client_weights))
            else:
                client_weights = client_weights[: len(aligned_client_states)]
        weight_sum = sum(client_weights)
        if weight_sum <= 0:
            client_weights = [1.0 / len(aligned_client_states)] * len(aligned_client_states)
        else:
            client_weights = [w / weight_sum for w in client_weights]

        if self.aggregation_strategy == "FedAvg":
            aggregated_state = FedAvg(
                aligned_client_states,
                global_model_state_dict=global_state,
                client_weights=client_weights,
            )
        elif self.aggregation_strategy == "FedAvgM":
            aggregated_state, self.fedavgm_momentum = FedAvgM(
                aligned_client_states,
                global_model_state_dict=global_state,
                proxy_dict=self.fedavgm_momentum,
                round_idx=round_idx,
                momentum_factor=self.federated_args["momentum_factor"],
                tau=self.federated_args["tau"],
                client_weights=client_weights,
            )
        elif self.aggregation_strategy == "FedAdagrad":
            aggregated_state, self.fedadagrad_velocity = FedAdagrad(
                aligned_client_states,
                global_model_state_dict=global_state,
                server_velocity=self.fedadagrad_velocity,
                epsilon=self.federated_args["epsilon"],
                tau=self.federated_args["tau"],
                client_weights=client_weights,
            )
        elif self.aggregation_strategy == "FedYogi":
            aggregated_state, self.fedyogi_momentum, self.fedyogi_velocity = FedYogi(
                aligned_client_states,
                global_model_state_dict=global_state,
                proxy_dict=self.fedyogi_momentum,
                opt_proxy_dict=self.fedyogi_velocity,
                round_idx=round_idx,
                beta1=self.federated_args["beta1"],
                beta2=self.federated_args["beta2"],
                epsilon=self.federated_args["epsilon"],
                tau=self.federated_args["tau"],
                client_weights=client_weights,
            )
        elif self.aggregation_strategy == "FedAdam":
            aggregated_state, self.fedadam_momentum, self.fedadam_velocity = FedAdam(
                aligned_client_states,
                global_model_state_dict=global_state,
                proxy_dict=self.fedadam_momentum,
                opt_proxy_dict=self.fedadam_velocity,
                round_idx=round_idx,
                beta1=self.federated_args["beta1"],
                beta2=self.federated_args["beta2"],
                epsilon=self.federated_args["epsilon"],
                tau=self.federated_args["tau"],
                client_weights=client_weights,
            )
        elif self.aggregation_strategy == "FedProx":
            aggregated_state = FedAvg(
                aligned_client_states,
                global_model_state_dict=global_state,
                client_weights=client_weights,
            )
        else:
            raise ValueError(f"Unsupported aggregation_strategy: {self.aggregation_strategy}")

        # Move aggregated state dict back to CPU if model is on CPU (to match model device)
        model_device = next(self.model.parameters()).device
        if model_device.type == "cpu" and use_gpu_for_agg:
            aggregated_state = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                               for k, v in aggregated_state.items()}

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
            self.model.save_pretrained(output_dir, safe_serialization=False)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(output_dir, safe_serialization=False)
        else:
            super().save_model(output_dir)

        # Save FedPerm specific state
        perm_state = {
            'revocation_status': self.revocation_status,
            'filter_F_A': self.filter_F_A,
            'source_filters': self.source_filters,
            'target_filters': self.target_filters,
            'source_to_targets': self.source_to_targets,
            'target_to_sources': self.target_to_sources,
            'permutation_keys': self.permutation_keys,
        }
        torch.save(perm_state, os.path.join(output_dir, 'fedperm_state.pt'))
        logger.info(f"FedPerm state saved to {output_dir}")

    def _should_save_round(self, round_idx: int) -> bool:
        """Respect FEDPERM_SKIP_SAVE_ROUNDS env; otherwise默认每轮保存。"""
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
            # IntervalStrategy shares values with SaveStrategy ("no","steps","epoch")
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

    def _is_adapter_parameter(self, name: str, param: torch.Tensor) -> bool:
        """
        Determine whether a parameter belongs to adapter modules (e.g., LoRA)
        and is eligible for permutation.
        """
        if not self.is_peft or not isinstance(param, torch.Tensor):
            return False
        return "lora" in name.lower()

    def _is_permutable_parameter(self, name: str, param: torch.Tensor) -> bool:
        """
        仅允许同层神经元置换：
        - 限 LoRA/adapter 参数
        - 跳过输入/输出层（embed/embedding/lm_head/output）
        """
        if not self._is_adapter_parameter(name, param):
            return False
        lowered = name.lower()
        if any(tag in lowered for tag in ["embed", "embedding", "lm_head", "output"]):
            return False
        # 标量/空张量不置换
        return param.dim() > 0 and param.numel() > 0
