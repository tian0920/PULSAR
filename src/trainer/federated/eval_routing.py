import json
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple


CLIENT_STAGE_PATTERN = re.compile(r"client(\d+)_")
FORGET_SOURCE_STAGE_PATTERN = re.compile(r"^(.*)__forget_src_(\d+)$")
# Target 额外评测：与当轮 source/授权客户端相同的「本轮全部已采样 source 的 forget 合并」
ROUND_FORGET_MERGE_STAGE_SUFFIX = "__forget_merge_round_sources"


def parse_client_stage(stage: str) -> Optional[int]:
    match = CLIENT_STAGE_PATTERN.match(stage or "")
    if not match:
        return None
    return int(match.group(1))


def split_stage_for_forget_source(stage: str) -> Tuple[str, Optional[int]]:
    match = FORGET_SOURCE_STAGE_PATTERN.match(stage or "")
    if not match:
        return stage, None
    return match.group(1), int(match.group(2))


def format_stage_with_forget_source(stage: str, forget_source_idx: int) -> str:
    base_stage, existing_source = split_stage_for_forget_source(stage)
    if existing_source is not None:
        return stage
    return f"{base_stage}__forget_src_{int(forget_source_idx)}"


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


def is_client_target_in_topology(
    client_idx: int, source_to_targets: Optional[Dict[int, List[int]]]
) -> bool:
    if not source_to_targets:
        return False
    for tgts in source_to_targets.values():
        if client_idx in (tgts or []):
            return True
    return False


def should_run_target_round_forget_merge_eval(
    enabled: bool,
    client_idx: int,
    round_source_clients: List[int],
    source_to_targets: Optional[Dict[int, List[int]]],
) -> bool:
    if not enabled:
        return False
    if client_idx in round_source_clients:
        return False
    return is_client_target_in_topology(client_idx, source_to_targets)


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
    source_to_targets: Optional[Dict[int, List[int]]] = None,
    source_client_indices: Optional[List[int]] = None,
) -> List[Tuple[str, Optional[int]]]:
    base_stage, stage_forget_source = split_stage_for_forget_source(stage)
    if stage_forget_source is not None:
        return [(stage, stage_forget_source)]
    if base_stage.endswith(ROUND_FORGET_MERGE_STAGE_SUFFIX):
        return [(base_stage, None)]
    client_idx = parse_client_stage(base_stage)
    if client_idx is not None:
        return [(base_stage, None)]
    if base_stage.startswith("server"):
        return [(base_stage, None)]

    unique_sources = sorted({int(idx) for idx in (forget_sources or [])})
    if not unique_sources:
        return [(stage, None)]

    source_to_targets = {
        int(src_idx): [int(tgt_idx) for tgt_idx in target_indices]
        for src_idx, target_indices in (source_to_targets or {}).items()
    }
    source_clients = sorted(
        {int(idx) for idx in (source_client_indices or source_to_targets.keys())}
    )
    if source_clients:
        client_idx = parse_client_stage(base_stage)
        if client_idx is None:
            variant_sources = unique_sources
        elif client_idx in source_clients:
            variant_sources = [client_idx] if client_idx in unique_sources else []
        else:
            target_to_sources = invert_source_to_targets(source_to_targets)
            variant_sources = [
                src_idx
                for src_idx in target_to_sources.get(client_idx, [])
                if src_idx in unique_sources
            ]
        if not variant_sources:
            return []
        unique_sources = variant_sources

    return [
        (format_stage_with_forget_source(base_stage, forget_source_idx), forget_source_idx)
        for forget_source_idx in unique_sources
    ]


def resolve_forget_indices_path(
    split_dir: str,
    stage: str,
    source_to_targets: Optional[Dict[int, List[int]]],
    source_client_indices: Optional[List[int]] = None,
    forget_source_idx: Optional[int] = None,
    active_source_client_indices: Optional[List[int]] = None,
) -> Optional[str]:
    global_path = os.path.join(split_dir, "global_forget_indices.json")
    return global_path if os.path.isfile(global_path) else None


def resolve_server_post_reuse_stage(
    target_stage: str,
    client_idx: int,
    source_to_targets: Optional[Dict[int, List[int]]],
    source_client_indices: Optional[List[int]] = None,
) -> Optional[str]:
    _, forget_source_idx = split_stage_for_forget_source(target_stage)
    source_to_targets = {
        int(src_idx): [int(tgt_idx) for tgt_idx in target_indices]
        for src_idx, target_indices in (source_to_targets or {}).items()
    }
    source_clients = sorted(
        {int(idx) for idx in (source_client_indices or source_to_targets.keys())}
    )
    target_to_sources = invert_source_to_targets(source_to_targets)
    if forget_source_idx is not None:
        if client_idx == forget_source_idx:
            return None
        if client_idx in source_to_targets.get(forget_source_idx, []):
            return None
    else:
        if client_idx in source_clients:
            return None
        if client_idx in target_to_sources:
            return None
    if forget_source_idx is not None:
        return format_stage_with_forget_source("server_post_agg", forget_source_idx)
    return "server_post_agg"


def resolve_server_pre_reuse_stage(
    target_stage: str,
    client_idx: int,
    source_to_targets: Optional[Dict[int, List[int]]],
    source_client_indices: Optional[List[int]] = None,
) -> Optional[str]:
    _, forget_source_idx = split_stage_for_forget_source(target_stage)
    source_to_targets = {
        int(src_idx): [int(tgt_idx) for tgt_idx in target_indices]
        for src_idx, target_indices in (source_to_targets or {}).items()
    }
    source_clients = sorted(
        {int(idx) for idx in (source_client_indices or source_to_targets.keys())}
    )
    target_to_sources = invert_source_to_targets(source_to_targets)
    if forget_source_idx is not None:
        if client_idx == forget_source_idx:
            return None
        if client_idx in source_to_targets.get(forget_source_idx, []):
            return None
    else:
        if client_idx in source_clients:
            return None
        if client_idx in target_to_sources:
            return None
    if forget_source_idx is not None:
        return format_stage_with_forget_source("server_pre_agg", forget_source_idx)
    return "server_pre_agg"
