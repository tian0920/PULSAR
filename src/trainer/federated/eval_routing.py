import json
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple


CLIENT_STAGE_PATTERN = re.compile(r"client(\d+)_")


def parse_client_stage(stage: str) -> Optional[int]:
    match = CLIENT_STAGE_PATTERN.match(stage or "")
    if not match:
        return None
    return int(match.group(1))


def invert_source_to_targets(source_to_targets: Optional[Dict[int, List[int]]]) -> Dict[int, List[int]]:
    target_to_sources: Dict[int, List[int]] = {}
    if not source_to_targets:
        return target_to_sources
    for raw_src, raw_targets in source_to_targets.items():
        src_idx = int(raw_src)
        for raw_target in raw_targets or []:
            target_idx = int(raw_target)
            target_to_sources.setdefault(target_idx, []).append(src_idx)
    for target_idx in list(target_to_sources.keys()):
        target_to_sources[target_idx] = sorted(set(target_to_sources[target_idx]))
    return target_to_sources


def load_index_list(path: Optional[str]) -> List[int]:
    if not path or not os.path.isfile(path):
        return []
    with open(path, "r") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        payload = payload.get("forget_indices", [])
    if not isinstance(payload, list):
        return []
    return [int(idx) for idx in payload]


def materialize_index_cache(
    split_dir: str,
    cache_name: str,
    indices: Iterable[int],
) -> Optional[str]:
    merged = sorted({int(idx) for idx in indices})
    os.makedirs(split_dir, exist_ok=True)
    cache_path = os.path.join(split_dir, cache_name)
    with open(cache_path, "w") as f:
        json.dump(merged, f)
    return cache_path


def build_eval_variants(
    stage: str,
    forget_sources: List[int],
) -> List[Tuple[str, Optional[int]]]:
    return [(stage, None)]


def resolve_forget_indices_path(
    split_dir: str,
    stage: str,
    source_to_targets: Optional[Dict[int, List[int]]],
    source_client_indices: Optional[List[int]] = None,
    forget_source_idx: Optional[int] = None,
) -> Optional[str]:
    source_to_targets = {
        int(src_idx): [int(tgt_idx) for tgt_idx in target_indices]
        for src_idx, target_indices in (source_to_targets or {}).items()
    }
    source_clients = sorted(
        {int(idx) for idx in (source_client_indices or source_to_targets.keys())}
    )
    target_to_sources = invert_source_to_targets(source_to_targets)

    def _client_forget_path(client_idx: int) -> str:
        return os.path.join(split_dir, f"client{int(client_idx)}_forget_indices.json")

    def _global_forget_path() -> str:
        return os.path.join(split_dir, "global_forget_indices.json")

    def _merge_client_paths(tag: str, client_indices: Iterable[int]) -> Optional[str]:
        merged: List[int] = []
        for client_idx in client_indices:
            merged.extend(load_index_list(_client_forget_path(int(client_idx))))
        cache_name = f"{tag}.json"
        return materialize_index_cache(split_dir, cache_name, merged)

    client_idx = parse_client_stage(stage)
    if client_idx is None:
        global_path = _global_forget_path()
        return global_path if os.path.isfile(global_path) else None

    if client_idx in source_clients:
        source_path = _client_forget_path(client_idx)
        return source_path if os.path.isfile(source_path) else None

    mapped_sources = target_to_sources.get(client_idx, [])
    if mapped_sources:
        if len(mapped_sources) == 1:
            source_path = _client_forget_path(mapped_sources[0])
            return source_path if os.path.isfile(source_path) else None
        return _merge_client_paths(
            f"eval_forget_indices__client{client_idx}_target_from_sources_"
            + "_".join(str(idx) for idx in mapped_sources),
            mapped_sources,
        )

    global_path = _global_forget_path()
    if os.path.isfile(global_path):
        return global_path
    return None


def resolve_server_post_reuse_stage(
    target_stage: str,
    client_idx: int,
    source_to_targets: Optional[Dict[int, List[int]]],
    source_client_indices: Optional[List[int]] = None,
) -> Optional[str]:
    source_to_targets = {
        int(src_idx): [int(tgt_idx) for tgt_idx in target_indices]
        for src_idx, target_indices in (source_to_targets or {}).items()
    }
    source_clients = sorted(
        {int(idx) for idx in (source_client_indices or source_to_targets.keys())}
    )
    target_to_sources = invert_source_to_targets(source_to_targets)
    if client_idx in source_clients:
        return None
    if client_idx in target_to_sources:
        return None
    return "server_post_agg"


def resolve_server_pre_reuse_stage(
    target_stage: str,
    client_idx: int,
    source_to_targets: Optional[Dict[int, List[int]]],
    source_client_indices: Optional[List[int]] = None,
) -> Optional[str]:
    source_to_targets = {
        int(src_idx): [int(tgt_idx) for tgt_idx in target_indices]
        for src_idx, target_indices in (source_to_targets or {}).items()
    }
    source_clients = sorted(
        {int(idx) for idx in (source_client_indices or source_to_targets.keys())}
    )
    target_to_sources = invert_source_to_targets(source_to_targets)
    if client_idx in source_clients:
        return None
    if client_idx in target_to_sources:
        return None
    return "server_pre_agg"
