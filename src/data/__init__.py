# Update: Xinyu Yan, 2025-06-21
# TL;DR: This file is used to register the datasets and collators

from omegaconf import DictConfig
from typing import Dict, Any, Union
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
    - retain_dataset / forget_dataset：默认分别均分到客户端，full 为两者的 concat。
    - assign_forget_to_revoker_only=True 时，forget 全部分配给 revoker_client_idx（用于过滤器训练 / forget_utility 评测）。
    """
    import os
    import json

    dataset_size = len(full_dataset)
    base_size = dataset_size // num_clients  # 仅用于全局索引分配
    extras = dataset_size % num_clients

    rng = np.random.RandomState(seed)
    use_explicit_splits = retain_dataset is not None and forget_dataset is not None

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
        retain_base = len(retain_indices) // num_clients
        retain_extras = len(retain_indices) % num_clients
        forget_base = len(forget_indices) // num_clients
        forget_extras = len(forget_indices) % num_clients
        indices = None if use_explicit_splits else np.arange(dataset_size)
    elif use_explicit_splits:
        retain_indices = np.arange(len(retain_dataset))
        forget_indices = np.arange(len(forget_dataset))
        rng.shuffle(retain_indices)
        rng.shuffle(forget_indices)
        retain_base = len(retain_indices) // num_clients
        retain_extras = len(retain_indices) % num_clients
        forget_base = len(forget_indices) // num_clients
        forget_extras = len(forget_indices) % num_clients
        indices = None
    elif global_retain_indices is not None and global_forget_indices is not None:
        retain_indices = list(global_retain_indices)
        forget_indices = list(global_forget_indices)
        rng.shuffle(retain_indices)
        rng.shuffle(forget_indices)
        retain_base = len(retain_indices) // num_clients
        retain_extras = len(retain_indices) % num_clients
        forget_base = len(forget_indices) // num_clients
        forget_extras = len(forget_indices) % num_clients
        indices = np.arange(dataset_size)
    else:
        raise ValueError("Must provide retain/forget splits or explicit indices; fallback by retain_ratio is disabled.")

    client_datasets = {}
    retain_all_for_global = []
    forget_all_for_global = []
    start_idx = 0
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
            r_size = retain_base + (1 if client_idx < retain_extras else 0)
            r_start = client_idx * retain_base + min(client_idx, retain_extras)
            client_retain_indices = retain_indices[r_start : r_start + r_size]
            if assign_forget_to_revoker_only:
                client_forget_indices = (
                    forget_indices.tolist()
                    if client_idx == revoker_client_idx
                    else []
                )
            else:
                f_size = forget_base + (1 if client_idx < forget_extras else 0)
                f_start = client_idx * forget_base + min(client_idx, forget_extras)
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
            r_size = retain_base + (1 if client_idx < retain_extras else 0)
            r_start = client_idx * retain_base + min(client_idx, retain_extras)
            client_retain_indices = retain_indices[r_start : r_start + r_size]
            if assign_forget_to_revoker_only:
                client_forget_indices = (
                    forget_indices.tolist()
                    if client_idx == revoker_client_idx
                    else []
                )
            else:
                f_size = forget_base + (1 if client_idx < forget_extras else 0)
                f_start = client_idx * forget_base + min(client_idx, forget_extras)
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

    return client_datasets


def get_federated_data(
    data,
    num_clients=3,
    target_client_idx=0,
    anchor="retain",
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

    # Split retain dataset
    retain_len = len(original_retain.retain)  # Actual length of original retain data
    chunk_size = retain_len // num_clients
    forget_base = getattr(original_forget, "forget", original_forget)

    client_datasets = {}
    for client_idx in range(num_clients):
        # Split retain indices (based on actual length of original retain data)
        start = client_idx * chunk_size
        end = start + chunk_size if client_idx < num_clients - 1 else retain_len
        retain_indices = list(range(start, end))

        # Create client's retain dataset (keep ForgetRetainDataset type)
        client_retain = ForgetRetainDataset(
            forget=None,  # retain subset does not contain forget data
            retain=Subset(original_retain.retain, retain_indices),
            anchor="retain",  # Force set to retain anchor
        )

        # Assign forget data
        if client_idx == target_client_idx:
            # Target client gets full forget dataset
            client_forget = ForgetRetainDataset(
                forget=forget_base, retain=None, anchor="forget"
            )
        else:
            # Non-target client's forget is set to None
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
