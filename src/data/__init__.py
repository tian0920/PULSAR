# Update: Xinyu Yan, 2025-06-21
# TL;DR: This file is used to register the datasets and collators

from omegaconf import DictConfig
from typing import Dict, Any, Union, List, Optional
from torch.utils.data import Subset, ConcatDataset
import numpy as np
import logging

from data.collators import DataCollatorForSupervisedDataset
from data.pretraining import PretrainingDataset, CompletionDataset
from data.qa import QADataset, QAwithIdkDataset
from data.unlearn import ForgetRetainDataset


DATASET_REGISTRY: Dict[str, Any] = {}
COLLATOR_REGISTRY: Dict[str, Any] = {}
logger = logging.getLogger(__name__)


class ClientFullRetainForget:
    """Container for per-client full / retain / forget subsets."""

    def __init__(self, full, retain, forget):
        self.full = full
        self.retain = retain
        self.forget = forget

    def __len__(self):
        return len(self.full)


def _normalize_client_index_list(client_indices, num_clients: int) -> List[int]:
    if client_indices is None:
        return []
    normalized = []
    for idx in client_indices:
        client_idx = int(idx)
        if client_idx < 0 or client_idx >= num_clients:
            raise ValueError(
                f"Client index {client_idx} out of range for num_clients={num_clients}"
            )
        normalized.append(client_idx)
    return sorted(dict.fromkeys(normalized))


def _normalize_weight_vector(
    weights,
    num_clients: int,
    active_client_indices: Optional[List[int]] = None,
) -> List[float]:
    if active_client_indices is None:
        active_client_indices = list(range(num_clients))
    if not active_client_indices:
        return [0.0] * num_clients

    resolved = [0.0] * num_clients
    if weights is None:
        uniform = 1.0 / len(active_client_indices)
        for client_idx in active_client_indices:
            resolved[client_idx] = uniform
        return resolved

    if hasattr(weights, "items"):
        for key, value in weights.items():
            client_idx = int(key)
            if client_idx < 0 or client_idx >= num_clients:
                raise ValueError(
                    f"Weight key client index {client_idx} out of range for num_clients={num_clients}"
                )
            resolved[client_idx] = float(value)
    else:
        weights_list = [float(v) for v in weights]
        if len(weights_list) == num_clients:
            resolved = weights_list
        elif len(weights_list) == len(active_client_indices):
            for client_idx, value in zip(active_client_indices, weights_list):
                resolved[client_idx] = value
        else:
            raise ValueError(
                f"Weight vector length {len(weights_list)} does not match "
                f"num_clients={num_clients} or active_client_count={len(active_client_indices)}"
            )

    for client_idx, value in enumerate(resolved):
        if value < 0:
            raise ValueError(f"Weight for client {client_idx} must be non-negative, got {value}")

    active_set = set(active_client_indices)
    for client_idx, value in enumerate(resolved):
        if client_idx not in active_set and value != 0:
            raise ValueError(
                f"Weight for inactive client {client_idx} must be 0, got {value}"
            )

    total = sum(resolved[client_idx] for client_idx in active_client_indices)
    if total <= 0:
        raise ValueError("Weights on active clients must sum to a positive value")

    normalized = [0.0] * num_clients
    for client_idx in active_client_indices:
        normalized[client_idx] = resolved[client_idx] / total
    return normalized


def _allocate_counts(total_size: int, weights: List[float]) -> List[int]:
    if total_size < 0:
        raise ValueError(f"total_size must be non-negative, got {total_size}")
    if total_size == 0:
        return [0] * len(weights)

    weight_sum = sum(weights)
    if weight_sum <= 0:
        raise ValueError("weights must sum to a positive value")

    raw_counts = [total_size * (weight / weight_sum) for weight in weights]
    counts = [int(np.floor(value)) for value in raw_counts]
    remainder = total_size - sum(counts)
    if remainder > 0:
        ranked = sorted(
            range(len(weights)),
            key=lambda idx: (raw_counts[idx] - counts[idx], weights[idx], -idx),
            reverse=True,
        )
        for idx in ranked[:remainder]:
            counts[idx] += 1
    return counts


def _register_data(data_class):
    """Register a dataset class to the global registry"""
    DATASET_REGISTRY[data_class.__name__] = data_class


def _register_collator(collator_class):
    """Register a collator class to the global registry"""
    COLLATOR_REGISTRY[collator_class.__name__] = collator_class


def _load_single_dataset(dataset_name, dataset_cfg: DictConfig, **kwargs):
    """Load a single dataset based on configuration"""
    dataset_handler_name = dataset_cfg.get("handler")
    assert dataset_handler_name is not None, ValueError(
        f"{dataset_name} handler not set"
    )
    dataset_handler = DATASET_REGISTRY.get(dataset_handler_name)
    if dataset_handler is None:
        raise NotImplementedError(
            f"{dataset_handler_name} not implemented or not registered"
        )
    dataset_args = dataset_cfg.args
    return dataset_handler(**dataset_args, **kwargs)


def get_datasets(dataset_cfgs: Union[Dict, DictConfig], **kwargs):
    """Load multiple datasets based on configuration"""
    dataset = {}
    for dataset_name, dataset_cfg in dataset_cfgs.items():
        access_name = dataset_cfg.get("access_key", dataset_name)
        dataset[access_name] = _load_single_dataset(dataset_name, dataset_cfg, **kwargs)
    if len(dataset) == 1:
        # Return a single dataset
        return list(dataset.values())[0]
    # Return mapping to multiple datasets
    return dataset


def get_data(data_cfg: DictConfig, mode="train", **kwargs):
    """Get data based on configuration and mode (train or unlearn)"""
    data = {}
    data_cfg = dict(data_cfg)
    anchor = data_cfg.pop("anchor", "forget")
    for split, dataset_cfgs in data_cfg.items():
        data[split] = get_datasets(dataset_cfgs, **kwargs)

    if mode == "train":
        return data
    elif mode == "unlearn":
        unlearn_splits = {k: v for k, v in data.items() if k not in ("eval", "test")}

        # Create unlearn and retain datasets
        unlearn_dataset = ForgetRetainDataset(**unlearn_splits, anchor=anchor)
        retain_dataset = ForgetRetainDataset(**unlearn_splits, anchor="retain")
        print(f"Unlearn dataset data structure: {unlearn_dataset}")
        print(f"Retain dataset data structure: {retain_dataset}")

        # Assign datasets to data
        data["forget"] = unlearn_dataset
        data["retain"] = retain_dataset

        # Remove only the original splits, preserving "forget" and "retain"
        for split in list(unlearn_splits.keys()):
            if split not in ("forget", "retain"):  # Protect "forget" and "retain"
                data.pop(split, None)  # Use None to avoid KeyError if key doesn't exist

        print(f"11111 data structure: {data.keys()}")
    return data


def _get_single_collator(collator_name: str, collator_cfg: DictConfig, **kwargs):
    """Load a single collator based on configuration"""
    collator_handler_name = collator_cfg.get("handler")
    assert collator_handler_name is not None, ValueError(
        f"{collator_name} handler not set"
    )
    collator_handler = COLLATOR_REGISTRY.get(collator_handler_name)
    if collator_handler is None:
        raise NotImplementedError(
            f"{collator_handler_name} not implemented or not registered"
        )
    collator_args = collator_cfg.args
    return collator_handler(**collator_args, **kwargs)


def get_collators(collator_cfgs, **kwargs):
    """Load multiple collators based on configuration"""
    collators = {}
    for collator_name, collator_cfg in collator_cfgs.items():
        collators[collator_name] = _get_single_collator(
            collator_name, collator_cfg, **kwargs
        )
    if len(collators) == 1:
        # return a single collator
        return list(collators.values())[0]
    # return collators in a dict
    return collators


def get_federated_full_retain_forget(
    full_dataset,
    num_clients: int = 3,
    retain_ratio: float = 0.9,
    seed: int = 0,
    assign_forget_to_revoker_only: bool = False,
    revoker_client_idx: int = 0,
    source_client_indices=None,
    source_to_targets=None,
    retain_client_weights=None,
    forget_source_weights=None,
    split_save_dir: str = None,
    retain_indices_path: str = None,
    forget_indices_path: str = None,
    global_retain_indices=None,
    global_forget_indices=None,
    retain_dataset=None,
    forget_dataset=None,
):
    """
    Split data to clients.
    - 必须提供 retain_dataset/forget_dataset 或显式全局索引，才进行划分。
    - retain_dataset 按 retain_client_weights 划分到所有客户端，full 为 retain/forget concat。
    - forget_dataset 仅分配给 source_client_indices，按 forget_source_weights 划分。
    - 兼容旧配置：assign_forget_to_revoker_only=True 时，forget 全部分配给 revoker_client_idx。
    """
    import os
    import json

    dataset_size = len(full_dataset)

    rng = np.random.RandomState(seed)
    use_explicit_splits = retain_dataset is not None and forget_dataset is not None
    source_client_indices = _normalize_client_index_list(source_client_indices, num_clients)
    if source_to_targets is not None and not source_client_indices:
        source_client_indices = _normalize_client_index_list(
            source_to_targets.keys(), num_clients
        )
    if not source_client_indices:
        if assign_forget_to_revoker_only:
            source_client_indices = [int(revoker_client_idx)]
        else:
            source_client_indices = list(range(num_clients))

    retain_weights = _normalize_weight_vector(retain_client_weights, num_clients)
    forget_weights = _normalize_weight_vector(
        forget_source_weights,
        num_clients,
        active_client_indices=source_client_indices,
    )

    def _payload_to_lists(payload, fname="<payload>"):
        if isinstance(payload, dict):
            r = payload.get("retain_indices", [])
            f = payload.get("forget_indices", [])
        elif isinstance(payload, list):
            r, f = payload, []
        else:
            raise ValueError(f"Unsupported split payload type for {fname}: {type(payload)}")
        return list(r), list(f)

    def _maybe_load_saved_splits(split_dir: str):
        """Load precomputed client splits if present (avoid re-splitting)."""
        if not split_dir or not os.path.isdir(split_dir):
            return None
        payloads = {}
        for client_idx in range(num_clients):
            idx_path = os.path.join(split_dir, f"client{client_idx}_indices.json")
            if not os.path.isfile(idx_path):
                return None
            with open(idx_path, "r") as f:
                payload = json.load(f)
            payloads[client_idx] = payload
        return payloads if len(payloads) == num_clients else None

    # 优先使用环境变量指定的主目录，其次使用当前保存目录
    saved_split_payloads = None
    for candidate_dir in [os.environ.get("FEDPERM_SPLIT_DIR"), split_save_dir]:
        saved_split_payloads = _maybe_load_saved_splits(candidate_dir)
        if saved_split_payloads:
            split_save_dir = candidate_dir or split_save_dir
            logger.info(f"Reusing existing client splits from {candidate_dir}")
            break

    # Optional: use precomputed retain/forget indices (from original splits)
    if not use_explicit_splits:
        if global_retain_indices is None and retain_indices_path is not None:
            with open(retain_indices_path, "r") as f:
                global_retain_indices = json.load(f)
        if global_forget_indices is None and forget_indices_path is not None:
            with open(forget_indices_path, "r") as f:
                global_forget_indices = json.load(f)

    use_saved_splits = saved_split_payloads is not None
    if use_saved_splits:
        retain_indices = []
        forget_indices = []
        for client_idx in range(num_clients):
            r_list, f_list = _payload_to_lists(
                saved_split_payloads[client_idx], f"client{client_idx}_indices.json"
            )
            retain_indices.extend(r_list)
            forget_indices.extend(f_list)
        indices = None if use_explicit_splits else np.arange(dataset_size)
    elif use_explicit_splits:
        retain_indices = np.arange(len(retain_dataset))
        forget_indices = np.arange(len(forget_dataset))
        rng.shuffle(retain_indices)
        rng.shuffle(forget_indices)
        indices = None
    elif global_retain_indices is not None and global_forget_indices is not None:
        retain_indices = list(global_retain_indices)
        forget_indices = list(global_forget_indices)
        rng.shuffle(retain_indices)
        rng.shuffle(forget_indices)
        indices = np.arange(dataset_size)
    else:
        raise ValueError("Must provide retain/forget splits or explicit indices; fallback by retain_ratio is disabled.")

    retain_counts = _allocate_counts(len(retain_indices), retain_weights)
    forget_counts = _allocate_counts(len(forget_indices), forget_weights)

    client_datasets = {}
    retain_all_for_global = []
    forget_all_for_global = []
    retain_offsets = np.cumsum([0] + retain_counts[:-1]).tolist()
    forget_offsets = np.cumsum([0] + forget_counts[:-1]).tolist()
    for client_idx in range(num_clients):
        if use_saved_splits:
            payload = saved_split_payloads[client_idx]
            retain_indices_local, forget_indices_local = _payload_to_lists(
                payload, f"client{client_idx}_indices.json"
            )
            if not use_explicit_splits:
                # prefer stored full indices if present, otherwise union retain/forget
                if isinstance(payload, dict) and payload.get("full_indices"):
                    client_indices = np.array(payload["full_indices"])
                else:
                    client_indices = np.array(retain_indices_local + forget_indices_local)
            else:
                client_indices = None
        elif use_explicit_splits:
            r_size = retain_counts[client_idx]
            r_start = retain_offsets[client_idx]
            client_retain_indices = retain_indices[r_start : r_start + r_size]
            f_size = forget_counts[client_idx]
            f_start = forget_offsets[client_idx]
            client_forget_indices = forget_indices[f_start : f_start + f_size]
            # Full indices are only for bookkeeping; actual subsets use respective datasets
            client_indices = None
            retain_indices_local = (
                client_retain_indices.tolist()
                if hasattr(client_retain_indices, "tolist")
                else list(client_retain_indices)
            )
            if hasattr(client_forget_indices, "tolist"):
                forget_indices_local = client_forget_indices.tolist()
            else:
                forget_indices_local = list(client_forget_indices)
        elif global_retain_indices is not None and global_forget_indices is not None:
            r_size = retain_counts[client_idx]
            r_start = retain_offsets[client_idx]
            client_retain_indices = retain_indices[r_start : r_start + r_size]
            f_size = forget_counts[client_idx]
            f_start = forget_offsets[client_idx]
            client_forget_indices = forget_indices[f_start : f_start + f_size]
            if hasattr(client_retain_indices, "tolist"):
                client_retain_indices = client_retain_indices.tolist()
            if hasattr(client_forget_indices, "tolist"):
                client_forget_indices = client_forget_indices.tolist()
            client_indices = np.array(client_retain_indices + client_forget_indices)
            retain_indices_local = client_retain_indices
            forget_indices_local = client_forget_indices
        else:
            raise RuntimeError("Unreachable: retain/forget splits not provided.")

        if use_explicit_splits:
            retain_subset = Subset(retain_dataset, retain_indices_local)
            forget_subset = Subset(forget_dataset, forget_indices_local)
            full_subset = ConcatDataset([retain_subset, forget_subset])
        else:
            full_subset = Subset(full_dataset, client_indices)
            if global_retain_indices is not None and global_forget_indices is not None:
                retain_subset = Subset(full_dataset, client_retain_indices)
                forget_subset = Subset(full_dataset, client_forget_indices)
            else:
                raise RuntimeError("Unreachable: retain/forget splits not provided.")

        client_datasets[client_idx] = ClientFullRetainForget(
            full=full_subset,
            retain=retain_subset,
            forget=forget_subset,
        )

        if split_save_dir is not None:
            os.makedirs(split_save_dir, exist_ok=True)
            payload = {
                "full_indices": client_indices.tolist() if client_indices is not None else [],
                "retain_indices": list(retain_indices_local),
                "forget_indices": list(forget_indices_local),
            }
            with open(
                os.path.join(split_save_dir, f"client{client_idx}_indices.json"), "w"
            ) as f:
                json.dump(payload, f)
            with open(
                os.path.join(split_save_dir, f"client{client_idx}_forget_indices.json"),
                "w",
            ) as f:
                json.dump(payload["forget_indices"], f)
            with open(
                os.path.join(split_save_dir, f"client{client_idx}_retain_indices.json"),
                "w",
            ) as f:
                json.dump(payload["retain_indices"], f)

        # Collect for global indices (for eval)
        retain_all_for_global.extend(retain_indices_local)
        forget_all_for_global.extend(forget_indices_local)

        print(
            f"[FedPerm split] client {client_idx}: full={len(full_subset)}, "
            f"retain={len(retain_subset)}, forget={len(forget_subset)}"
        )

    if split_save_dir is not None:
        os.makedirs(split_save_dir, exist_ok=True)
        retain_all_for_global = sorted(set(retain_all_for_global))
        forget_all_for_global = sorted(set(forget_all_for_global))
        with open(os.path.join(split_save_dir, "global_retain_indices.json"), "w") as f:
            json.dump(retain_all_for_global, f)
        with open(os.path.join(split_save_dir, "global_forget_indices.json"), "w") as f:
            json.dump(forget_all_for_global, f)
        metadata = {
            "num_clients": int(num_clients),
            "source_client_indices": list(source_client_indices),
            "source_to_targets": {
                str(int(src_idx)): [int(tgt_idx) for tgt_idx in target_indices]
                for src_idx, target_indices in (source_to_targets or {}).items()
            },
            "retain_client_weights": [float(v) for v in retain_weights],
            "forget_source_weights": [float(v) for v in forget_weights],
        }
        with open(os.path.join(split_save_dir, "fedperm_topology.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    return client_datasets


def get_federated_data(
    data,
    num_clients=3,
    target_client_idx=0,
    source_client_indices=None,
    anchor="retain",
    seed: int = 0,
):
    """
    Distribute data for federated learning scenarios
    Ensure that all clients' retain and forget are ForgetRetainDataset types
    """
    # Extract original datasets (must be ForgetRetainDataset types)
    original_retain: ForgetRetainDataset = data["retain"]
    original_forget: ForgetRetainDataset = data["forget"]

    # Parameter validation
    if target_client_idx >= num_clients:
        raise ValueError(
            f"Target client index {target_client_idx} >= num_clients {num_clients}"
        )
    if not isinstance(original_retain, ForgetRetainDataset) or not isinstance(
        original_forget, ForgetRetainDataset
    ):
        raise TypeError("Input datasets must be ForgetRetainDataset types")

    source_client_indices = _normalize_client_index_list(
        source_client_indices, num_clients
    )
    if not source_client_indices:
        source_client_indices = [int(target_client_idx)]

    # Randomly split retain dataset across clients.
    retain_len = len(original_retain.retain)
    chunk_size = retain_len // num_clients
    rng = np.random.RandomState(seed)
    shuffled_indices = rng.permutation(retain_len).tolist()
    forget_base = getattr(original_forget, "forget", original_forget)

    client_datasets = {}
    for client_idx in range(num_clients):
        start = client_idx * chunk_size
        end = start + chunk_size if client_idx < num_clients - 1 else retain_len
        retain_indices = shuffled_indices[start:end]

        # Create client's retain dataset (keep ForgetRetainDataset type)
        client_retain = ForgetRetainDataset(
            forget=None,  # retain subset does not contain forget data
            retain=Subset(original_retain.retain, retain_indices),
            anchor="retain",  # Force set to retain anchor
        )

        # Assign forget data
        if client_idx in source_client_indices:
            # Source client gets the forget dataset.
            client_forget = ForgetRetainDataset(
                forget=forget_base, retain=None, anchor="forget"
            )
        else:
            # Non-source clients do not receive forget data.
            client_forget = ForgetRetainDataset(
                forget=None, retain=None, anchor="retain"
            )

        # Create client final dataset
        client_dataset = ForgetRetainDataset(
            forget=client_forget, retain=client_retain, anchor=anchor
        )

        client_datasets[client_idx] = client_dataset

        # Type validation
        print(f"Client {client_idx} Data Type Validation:")
        print(
            f"Forget Type: {type(client_dataset.forget)}"
        )  # Should output ForgetRetainDataset
        print(
            f"Retain Type: {type(client_dataset.retain)}"
        )  # Should output ForgetRetainDataset

    return client_datasets


# Register datasets
_register_data(QADataset)
_register_data(QAwithIdkDataset)
_register_data(PretrainingDataset)
_register_data(CompletionDataset)

# Register composite datasets used in unlearning
# groups: unlearn
_register_data(ForgetRetainDataset)

# Register collators
_register_collator(DataCollatorForSupervisedDataset)
