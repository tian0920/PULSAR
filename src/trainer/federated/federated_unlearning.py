import torch
import logging
import json
import shutil
from copy import deepcopy
from typing import Dict, List, Any, Optional
import os
import random
from trainer.base import FinetuneTrainer
from trainer.unlearn.base import UnlearnTrainer
from data.unlearn import ForgetRetainDataset
from torch.utils.data import Dataset
from trainer.federated.federated_utils import *
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from transformers.trainer_utils import IntervalStrategy, SaveStrategy
from trainer.federated.eval_routing import (
    ROUND_FORGET_MERGE_STAGE_SUFFIX,
    build_eval_variants,
    resolve_forget_indices_path,
    should_run_target_round_forget_merge_eval,
)


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
        self.selected_clients = set()
        self.kwargs = kwargs
        self.source_to_targets = self._resolve_source_to_targets(
            kwargs.get("source_to_targets", None)
        )
        self.source_client_indices = sorted(self.source_to_targets.keys())
        self.target_client_indices = sorted(
            {
                int(target_idx)
                for target_indices in self.source_to_targets.values()
                for target_idx in (target_indices or [])
            }
        )
        explicit_source_clients = kwargs.get("source_client_indices", None)
        if explicit_source_clients:
            explicit_source_clients = sorted(
                dict.fromkeys(int(idx) for idx in explicit_source_clients)
            )
            for src_idx in explicit_source_clients:
                self.source_to_targets.setdefault(src_idx, [])
            self.source_client_indices = sorted(self.source_to_targets.keys())
        if not self.source_client_indices:
            self.source_client_indices = [int(self.target_client_idx)]
        self.participation_ratio = float(kwargs.get("participation_ratio", 1.0))
        self.min_participating_clients = int(kwargs.get("min_participating_clients", 1))
        participation_seed = kwargs.get("participation_seed", None)
        if participation_seed is None:
            participation_seed = getattr(self.args, "seed", 0)
        self.participation_seed = int(participation_seed)
        self.min_source_clients_per_round = int(kwargs.get("min_source_clients_per_round", 2))
        self.targets_per_source_per_round = int(kwargs.get("targets_per_source_per_round", 2))
        self.round_topology_seed = int(kwargs.get("round_topology_seed", self.participation_seed))
        self.current_round_source_to_targets: Dict[int, List[int]] = {}
        self.current_round_source_clients: List[int] = []
        _etm = kwargs.get("eval_target_round_forget_merge", None)
        if _etm is None:
            self.eval_target_round_forget_merge = os.environ.get(
                "EVAL_TARGET_ROUND_FORGET_MERGE", ""
            ).lower() in ("1", "true", "yes", "on")
        else:
            self.eval_target_round_forget_merge = bool(_etm)
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
        logger.info(f"Source clients for eval: {self.source_client_indices}")
        logger.info(f"Source to targets mapping: {self.source_to_targets}")
        logger.info(
            "Client participation ratio: %s, min clients: %s, seed: %s",
            self.participation_ratio,
            self.min_participating_clients,
            self.participation_seed,
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

    def _resolve_source_to_targets(self, source_to_targets):
        topology = source_to_targets
        if topology is None:
            metadata = self._load_topology_metadata()
            if metadata:
                topology = metadata.get("source_to_targets", None)
        if topology is None:
            return {}

        normalized = {}
        for raw_src, raw_targets in topology.items():
            src_idx = int(raw_src)
            if src_idx < 0 or src_idx >= self.num_clients:
                raise ValueError(
                    f"Source client index {src_idx} out of range for num_clients={self.num_clients}"
                )
            target_indices = []
            for raw_tgt in raw_targets or []:
                tgt_idx = int(raw_tgt)
                if tgt_idx < 0 or tgt_idx >= self.num_clients:
                    raise ValueError(
                        f"Target client index {tgt_idx} out of range for num_clients={self.num_clients}"
                    )
                if tgt_idx == src_idx:
                    raise ValueError(
                        f"Source client {src_idx} cannot target itself in source_to_targets"
                    )
                target_indices.append(tgt_idx)
            normalized[src_idx] = sorted(dict.fromkeys(target_indices))
        return normalized

    def _eligible_clients(self) -> List[int]:
        return [
            client_idx for client_idx in range(self.num_clients) if client_idx in self.federated_dataset
        ]

    def _required_participating_clients(self, round_idx: int) -> List[int]:
        eligible_clients = self._eligible_clients()
        eligible_set = set(eligible_clients)
        source_candidates = [
            idx for idx in self.source_client_indices if idx in eligible_set
        ]
        target_candidates = [
            idx for idx in self.target_client_indices if idx in eligible_set
        ]
        if self.source_client_indices and not source_candidates:
            raise ValueError("No eligible source clients available for participation sampling")
        if self.target_client_indices and not target_candidates:
            raise ValueError("No eligible target clients available for participation sampling")
        if source_candidates:
            rng = random.Random(self.participation_seed + round_idx)
            required_sources = min(
                len(source_candidates),
                max(1, self.min_source_clients_per_round),
            )
            forced_source_clients = sorted(rng.sample(source_candidates, required_sources))
        else:
            forced_source_clients = []
        return sorted(
            dict.fromkeys(
                forced_source_clients
                + select_forced_group_clients(
                    eligible_clients=self._eligible_clients(),
                    client_groups=[source_candidates, target_candidates],
                    round_idx=round_idx,
                    seed=self.participation_seed,
                )
            )
        )

    def _active_source_to_targets(self) -> Dict[int, List[int]]:
        return self.current_round_source_to_targets or self.source_to_targets

    def _round_topology_cache_path(self) -> str:
        split_dir = os.environ.get(
            "FEDPERM_SPLIT_DIR", os.path.join(self.args.output_dir, "client_splits")
        )
        os.makedirs(split_dir, exist_ok=True)
        return os.path.join(split_dir, "fedperm_round_topology.json")

    def _load_round_topology(self, round_idx: int) -> Optional[Dict[str, Any]]:
        cache_path = self._round_topology_cache_path()
        if not os.path.isfile(cache_path):
            return None
        try:
            with open(cache_path, "r") as f:
                payload = json.load(f)
            rounds = payload.get("rounds", {})
            entry = rounds.get(str(round_idx + 1))
            if not entry:
                return None
            return {
                "source_clients": [int(idx) for idx in entry.get("source_clients", [])],
                "source_to_targets": {
                    int(src): [int(tgt) for tgt in tgts]
                    for src, tgts in (entry.get("source_to_targets", {}) or {}).items()
                },
            }
        except Exception as exc:
            logger.warning("Failed to load round topology cache: %s", exc)
            return None

    def _save_round_topology(self, round_idx: int):
        cache_path = self._round_topology_cache_path()
        payload = {"rounds": {}}
        if os.path.isfile(cache_path):
            try:
                with open(cache_path, "r") as f:
                    payload = json.load(f)
            except Exception:
                payload = {"rounds": {}}
        payload.setdefault("rounds", {})
        payload["rounds"][str(round_idx + 1)] = {
            "source_clients": [int(idx) for idx in self.current_round_source_clients],
            "source_to_targets": {
                str(src): [int(tgt) for tgt in tgts]
                for src, tgts in self.current_round_source_to_targets.items()
            },
            "seed": int(self.round_topology_seed),
        }
        with open(cache_path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def _assign_round_source_targets(self, round_idx: int):
        cached = self._load_round_topology(round_idx)
        if cached:
            self.current_round_source_clients = cached["source_clients"]
            self.current_round_source_to_targets = cached["source_to_targets"]
            return
        selected = sorted(int(idx) for idx in self.selected_clients)
        selected_set = set(selected)
        source_pool = [idx for idx in self.source_client_indices if idx in selected_set]
        if len(source_pool) < self.min_source_clients_per_round:
            source_pool = selected
        required_sources = min(len(source_pool), max(1, self.min_source_clients_per_round))
        rng = random.Random(self.round_topology_seed + round_idx)
        round_sources = sorted(rng.sample(source_pool, required_sources)) if required_sources > 0 else []
        candidate_targets = [idx for idx in selected if idx not in round_sources]
        mapping: Dict[int, List[int]] = {}
        for offset, src_idx in enumerate(round_sources):
            if not candidate_targets:
                mapping[src_idx] = []
                continue
            local_rng = random.Random(self.round_topology_seed + round_idx * 9973 + offset + 1)
            sample_n = min(len(candidate_targets), max(1, self.targets_per_source_per_round))
            mapping[src_idx] = sorted(local_rng.sample(candidate_targets, sample_n))
        self.current_round_source_clients = round_sources
        self.current_round_source_to_targets = mapping
        self._save_round_topology(round_idx)

    def _select_participating_clients(self, round_idx: int) -> List[int]:
        required_clients = self._required_participating_clients(round_idx)
        selected = sample_participating_clients(
            eligible_clients=self._eligible_clients(),
            participation_ratio=self.participation_ratio,
            min_clients=self.min_participating_clients,
            round_idx=round_idx,
            seed=self.participation_seed,
            forced_clients=required_clients,
        )
        logger.info(
            "Round %s selected clients for retain training: %s%s",
            round_idx + 1,
            selected,
            f" (forced={required_clients})" if required_clients else "",
        )
        return selected

    def _save_participation_metadata(
        self,
        round_idx: int,
        selected_clients: List[int],
        forced_clients: Optional[List[int]] = None,
    ):
        round_dir = os.path.join(
            self.args.output_dir,
            "eval_rounds",
            f"round_{round_idx + 1}",
        )
        os.makedirs(round_dir, exist_ok=True)
        payload = {
            "round": round_idx + 1,
            "selected_clients": [int(idx) for idx in selected_clients],
            "forced_clients": [int(idx) for idx in (forced_clients or [])],
            "source_clients": [int(idx) for idx in self.current_round_source_clients],
            "source_to_targets": {
                str(src): [int(tgt) for tgt in tgts]
                for src, tgts in self.current_round_source_to_targets.items()
            },
            "participation_ratio": self.participation_ratio,
            "min_participating_clients": self.min_participating_clients,
        }
        with open(os.path.join(round_dir, "participation.json"), "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def _forget_eval_sources(self) -> List[int]:
        if self.current_round_source_clients:
            return list(self.current_round_source_clients)
        return sorted(
            idx for idx in self.source_client_indices if idx in self.selected_clients
        )


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
        
        for round_idx in range(self.global_rounds):
            logger.info(f"Starting global round {round_idx + 1}/{self.global_rounds}")
            logger.info("Step 1: Unlearning on source clients %s with current global model", self.source_client_indices)
            selected_clients = self._select_participating_clients(round_idx)
            self.selected_clients = set(selected_clients)
            self._assign_round_source_targets(round_idx)
            self._save_participation_metadata(
                round_idx,
                selected_clients,
                forced_clients=self._required_participating_clients(round_idx),
            )
            unlearn_source_clients = self._source_clients_for_unlearning()
            unlearn_state_dicts = []
            unlearn_weights = []
            if not unlearn_source_clients:
                logger.warning("No source client with forget data available, skipping source unlearning")
            for source_client_idx in unlearn_source_clients:
                source_dataset = self.federated_dataset[source_client_idx]
                forget_data = self._resolve_forget_data(source_dataset.forget)
                if forget_data is None or len(forget_data) == 0:
                    logger.warning(
                        "Source client %s has no forget data, skipping unlearning",
                        source_client_idx,
                    )
                    continue
                logger.info("Running unlearning on source client %s", source_client_idx)
                unlearn_model = deepcopy(self.model)
                unlearn_trainer = self.unlearn_client_model(unlearn_model, source_dataset)
                logger.info("Completed unlearning on source client %s", source_client_idx)
                unlearn_state_dicts.append(self._extract_model_state(unlearn_trainer.model))
                unlearn_weights.append(len(forget_data))

            if unlearn_state_dicts:
                self.theta_unlearn_state = self._aggregate_unlearn_states(
                    unlearn_state_dicts,
                    unlearn_weights,
                )
                self._apply_theta_unlearn_state_to_global()

            logger.info("Step 2: All clients training on retain data with updated global model")

            # Server-side eval before broadcasting (聚合前评测)
            if self.enable_pre_eval:
                self._maybe_evaluate_model(
                    model=self.model,
                    round_idx=round_idx,
                    stage="server_pre_agg",
                )
            
         
            self.client_models = [None] * self.num_clients
            self.client_data_sizes = [0] * self.num_clients

           
            logger.info(f"Aggregation strategy: {self.aggregation_strategy}")


            for client_idx in range(self.num_clients):
                if client_idx not in self.federated_dataset:
                    logger.warning(f"Skipping client {client_idx} as no data is available")
                    continue
                if client_idx not in self.selected_clients:
                    logger.info(f"Skipping client {client_idx} this round due to participation sampling")
                    continue
                    
                client_model = deepcopy(self.model).cpu()
                if self.init_all_clients_from_unlearn:
                    self._apply_theta_unlearn_state_to_model(
                        client_model, scope=f"client{client_idx}"
                    )
                self.client_models[client_idx] = client_model
                client_dataset = self.federated_dataset[client_idx]
                
                retain_data = self._resolve_retain_data(client_dataset.retain)
                client_data_size = len(retain_data)
                self.client_data_sizes[client_idx] = client_data_size
                
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

            for client_idx in selected_clients:
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
                if should_run_target_round_forget_merge_eval(
                    self.eval_target_round_forget_merge,
                    client_idx,
                    self._forget_eval_sources(),
                    self._active_source_to_targets(),
                ):
                    self._maybe_evaluate_model(
                        model=trained_client_model,
                        round_idx=round_idx,
                        stage=f"client{client_idx}_post_train{ROUND_FORGET_MERGE_STAGE_SUFFIX}",
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

    @staticmethod
    def _resolve_forget_data(forget_dataset):
        """Return underlying forget split when present; fallback to the dataset itself."""
        if forget_dataset is None:
            return None
        if hasattr(forget_dataset, "forget") and getattr(forget_dataset, "forget") is not None:
            return forget_dataset.forget
        return forget_dataset

    def _source_clients_for_unlearning(self) -> List[int]:
        if self.current_round_source_clients:
            return [
                client_idx
                for client_idx in self.current_round_source_clients
                if client_idx in self.federated_dataset
            ]
        return [
            client_idx
            for client_idx in self.source_client_indices
            if client_idx in self.federated_dataset and client_idx in self.selected_clients
        ]

    def _extract_model_state(self, model):
        if self.is_peft:
            state = get_peft_model_state_dict(model)
        else:
            state = model.state_dict()
        return {
            key: value.detach().clone().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in state.items()
        }

    def _aggregate_unlearn_states(self, state_dicts, weights):
        if not state_dicts:
            return None
        if len(state_dicts) == 1:
            return state_dicts[0]
        return FedAvg(
            state_dicts,
            global_model_state_dict=state_dicts[0],
            client_weights=weights,
        )

    def _apply_theta_unlearn_state_to_global(self):
        """Initialize global model from theta_unlearn_state when available."""
        self._apply_theta_unlearn_state_to_model(self.model, scope="global")

    def _apply_theta_unlearn_state_to_model(self, model, scope: str = "model"):
        """Initialize a model from theta_unlearn_state when available."""
        if self.theta_unlearn_state is None:
            logger.warning(
                f"theta_unlearn_state is missing; skip applying to {scope}"
            )
            return
        if self.is_peft:
            set_peft_model_state_dict(model, self.theta_unlearn_state)
        else:
            model.load_state_dict(self.theta_unlearn_state, strict=False)
        logger.info(f"Initialized {scope} from theta_unlearn_state")

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
        force_eval = os.environ.get("FORCE_FEDPERM_EVAL", "0") in ("1", "true", "True")
        model_was_on_cpu = next(model.parameters()).device.type == "cpu"
        if torch.cuda.is_available() and model_was_on_cpu:
            device = torch.device("cuda")
            model = model.to(device)
            logger.info(f"Model moved to GPU device {device} for evaluation at stage {stage}")

        overwrite = True
        if hasattr(self.evaluator, "eval_cfg"):
            overwrite = getattr(self.evaluator.eval_cfg, "overwrite", True)

        try:
            forget_sources = self._forget_eval_sources()
            if self.source_client_indices and not forget_sources:
                logger.info(
                    "No sampled source clients selected at round %s; skip eval for stage %s",
                    round_idx + 1,
                    stage,
                )
                return
            active_map = self._active_source_to_targets()
            for eval_stage, forget_source_idx in build_eval_variants(
                stage,
                forget_sources,
                source_to_targets=active_map,
                source_client_indices=sorted(active_map.keys()),
            ):
                try:
                    import re

                    split_dir = os.environ.get(
                        "FEDPERM_SPLIT_DIR", os.path.join(self.args.output_dir, "client_splits")
                    )
                    retain_idx_path = None
                    m = re.match(r"client(\d+)_", eval_stage)
                    if eval_stage.startswith("server"):
                        retain_idx_path = os.path.join(split_dir, "global_retain_indices.json")
                    elif m:
                        cidx = int(m.group(1))
                        retain_idx_path = os.path.join(split_dir, f"client{cidx}_retain_indices.json")

                    forget_idx_path = resolve_forget_indices_path(
                        split_dir=split_dir,
                        stage=eval_stage,
                        source_to_targets=active_map,
                        source_client_indices=sorted(active_map.keys()),
                        forget_source_idx=forget_source_idx,
                        active_source_client_indices=forget_sources,
                    )

                    if hasattr(self.evaluator, "eval_cfg"):
                        if hasattr(self.evaluator.eval_cfg, "tofu"):
                            eval_tofu = self.evaluator.eval_cfg.tofu
                            if not forget_idx_path or not os.path.exists(forget_idx_path):
                                logger.error(
                                    f"Forget indices missing for stage {eval_stage}: {forget_idx_path}. "
                                    "Abort eval to avoid falling back to full dataset."
                                )
                                continue
                            eval_tofu.forget_indices_path = forget_idx_path
                            if retain_idx_path:
                                if not os.path.exists(retain_idx_path):
                                    logger.error(
                                        f"Retain indices missing for stage {eval_stage}: {retain_idx_path}. "
                                        "Abort eval to avoid falling back to full dataset."
                                    )
                                    continue
                                eval_tofu.retain_indices_path = retain_idx_path
                            self._update_eval_dataset_indices(eval_tofu)

                        if hasattr(self.evaluator.eval_cfg, "muse"):
                            eval_muse = self.evaluator.eval_cfg.muse
                            if not forget_idx_path or not os.path.exists(forget_idx_path):
                                logger.error(
                                    f"MUSE forget indices missing for stage {eval_stage}: {forget_idx_path}. "
                                    "Abort eval to avoid falling back to full dataset."
                                )
                                continue
                            eval_muse.forget_indices_path = forget_idx_path
                            self._update_eval_dataset_indices(eval_muse)

                        if hasattr(self.evaluator, "metrics"):
                            for metric in self.evaluator.metrics.values():
                                if hasattr(metric, "data"):
                                    metric.data = None
                                if hasattr(metric, "collators"):
                                    metric.collators = None
                except Exception as exc:
                    logger.warning(f"Failed to set eval indices dynamically for {eval_stage}: {exc}")
                    continue

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
            if torch.cuda.is_available() and model_was_on_cpu:
                model = model.cpu()
                torch.cuda.empty_cache()

    @staticmethod
    def _update_eval_dataset_indices(eval_cfg):
        """Propagate retain/forget indices into metric dataset configs."""
        if not hasattr(eval_cfg, "metrics"):
            return
        metrics_cfg = eval_cfg.metrics

        def _update_datasets(datasets_cfg):
            if not datasets_cfg:
                return
            for ds_name, ds_cfg in datasets_cfg.items():
                args = ds_cfg.get("args", None)
                if not args or "indices_path" not in args:
                    continue
                access_key = str(ds_cfg.get("access_key", "")).lower()
                name_hint = str(ds_name).lower()
                retain_path = getattr(eval_cfg, "retain_indices_path", None)
                forget_path = getattr(eval_cfg, "forget_indices_path", None)
                if "forget" in access_key or "forget" in name_hint:
                    if forget_path:
                        args["indices_path"] = forget_path
                elif "retain" in access_key or "retain" in name_hint:
                    if retain_path:
                        args["indices_path"] = retain_path

        for _, metric_cfg in metrics_cfg.items():
            _update_datasets(metric_cfg.get("datasets", None))
            pre_compute_cfg = metric_cfg.get("pre_compute", None)
            if pre_compute_cfg:
                for _, pre_cfg in pre_compute_cfg.items():
                    _update_datasets(pre_cfg.get("datasets", None))

    def _reuse_server_post_eval(self, round_idx: int, target_stage: str) -> bool:
        """
        Reuse previous round server_post_agg eval as current pre_train eval.
        """
        if self.evaluator is None or round_idx == 0:
            return False
        if self.source_client_indices and not self._forget_eval_sources():
            return False
        copied = False
        active_map = self._active_source_to_targets()
        for eval_stage, forget_source_idx in build_eval_variants(
            target_stage,
            self._forget_eval_sources(),
            source_to_targets=active_map,
            source_client_indices=sorted(active_map.keys()),
        ):
            src_stage = (
                "server_post_agg"
                if forget_source_idx is None
                else f"server_post_agg__forget_src_{forget_source_idx}"
            )
            prev_dir = os.path.join(
                self.args.output_dir,
                "eval_rounds",
                f"round_{round_idx}",
                src_stage,
            )
            src_summary = os.path.join(prev_dir, "TOFU_SUMMARY.json")
            if not os.path.isfile(src_summary):
                continue

            dest_dir = os.path.join(
                self.args.output_dir,
                "eval_rounds",
                f"round_{round_idx + 1}",
                eval_stage,
            )
            os.makedirs(dest_dir, exist_ok=True)

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
        if self.source_client_indices and not self._forget_eval_sources():
            return False
        copied = False
        active_map = self._active_source_to_targets()
        for eval_stage, forget_source_idx in build_eval_variants(
            target_stage,
            self._forget_eval_sources(),
            source_to_targets=active_map,
            source_client_indices=sorted(active_map.keys()),
        ):
            src_stage = (
                "server_pre_agg"
                if forget_source_idx is None
                else f"server_pre_agg__forget_src_{forget_source_idx}"
            )
            src_dir = os.path.join(
                self.args.output_dir,
                "eval_rounds",
                f"round_{round_idx + 1}",
                src_stage,
            )
            src_summary = os.path.join(src_dir, "TOFU_SUMMARY.json")
            if not os.path.isfile(src_summary):
                continue

            dest_dir = os.path.join(
                self.args.output_dir,
                "eval_rounds",
                f"round_{round_idx + 1}",
                eval_stage,
            )
            os.makedirs(dest_dir, exist_ok=True)

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
        filtered.pop("init_all_clients_from_unlearn", None)
        return filtered

    def unlearn_client_model(self, client_model, client_dataset):
        from trainer import TRAINER_REGISTRY
        trainer_cls = TRAINER_REGISTRY.get(self.unlearn_trainer_cls, None)


        forget_dataset = client_dataset.forget

        # 展开可能的嵌套 ForgetRetainDataset，拿到底层真实数据集
        from data.unlearn import ForgetRetainDataset
        if isinstance(forget_dataset, ForgetRetainDataset) and getattr(
            forget_dataset, "forget", None
        ) is not None:
            forget_dataset = forget_dataset.forget

        logger.info(f"forget_data length: {len(forget_dataset)}")
        logger.info(f"Unlearning_cls_name: {self.unlearn_trainer_cls}")
        
        class ForgetOnlyDataset(Dataset):
            def __init__(self, forget_data):
                self.forget_data = forget_data

            def __len__(self):
                return len(self.forget_data)

            def __getitem__(self, idx):
                forget_item = self.forget_data[idx]
                if not (isinstance(forget_item, dict) and "forget" in forget_item):
                    forget_item = {"forget": forget_item}
                retain_item = forget_item
                if not (isinstance(retain_item, dict) and "retain" in retain_item):
                    retain_item = {"retain": retain_item}
                return {
                    "forget": forget_item,
                    "retain": retain_item,
                }
        
        combined_dataset = ForgetOnlyDataset(forget_dataset)

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
        

        active_client_indices = [
            client_idx
            for client_idx, model in enumerate(self.client_models)
            if model is not None and self.client_data_sizes[client_idx] > 0
        ]
        total_samples = sum(self.client_data_sizes[client_idx] for client_idx in active_client_indices)
        if total_samples == 0 or not active_client_indices:
            logger.warning("No samples available for aggregation")
            return
            
        client_weights = [
            self.client_data_sizes[client_idx] / total_samples
            for client_idx in active_client_indices
        ]
        logger.info(f"Client sample sizes: {self.client_data_sizes}")
        logger.info(
            "Aggregating active clients: %s with weights: %s",
            active_client_indices,
            [f"{w:.4f}" for w in client_weights],
        )

        # Determine target device for aggregation (prefer GPU if available)
        use_gpu_for_agg = torch.cuda.is_available()
        agg_device = torch.device("cuda") if use_gpu_for_agg else torch.device("cpu")
        
        if self.is_peft:
            client_state_dicts = []
            for idx in active_client_indices:
                model = self.client_models[idx]
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
            for idx in active_client_indices:
                model = self.client_models[idx]
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
