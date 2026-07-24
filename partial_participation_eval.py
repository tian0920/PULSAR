import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

ROLES_ORDER = ("source", "target", "authorized")


def _round_dirs(run_dir: Path) -> List[Path]:
    eval_root = run_dir / "eval_rounds"
    if not eval_root.exists():
        return []
    return sorted(
        (p for p in eval_root.glob("round_*") if p.is_dir()),
        key=lambda p: int(p.name.split("_")[1]),
    )


def _normalize_client(client: Union[int, str]) -> Tuple[int, str]:
    if isinstance(client, str) and client.startswith("client"):
        client_idx = int(client[len("client"):])
    else:
        client_idx = int(client)
    return client_idx, f"client{client_idx}"


def _stage_name(stage: str, forget_source: Optional[int]) -> str:
    if forget_source is None or "__forget_src_" in stage:
        return stage
    return f"{stage}__forget_src_{int(forget_source)}"


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _participated_in_round(round_dir: Path, client_idx: int, stage: str, summary_file: str) -> bool:
    participation_meta = _load_json(round_dir / "participation.json")
    if participation_meta is not None:
        selected = participation_meta.get("selected_clients", [])
        return int(client_idx) in {int(idx) for idx in selected}

    _, client_name = _normalize_client(client_idx)
    summary_path = round_dir / f"{client_name}_{stage}" / summary_file
    return summary_path.exists()


def find_last_participation_round(
    run_dir: Union[str, Path],
    client: Union[int, str],
    max_round: Optional[int] = None,
    stage: str = "post_train",
    summary_file: str = "TOFU_SUMMARY.json",
    forget_source: Optional[int] = None,
) -> Optional[int]:
    run_dir = Path(run_dir)
    client_idx, _ = _normalize_client(client)
    stage = _stage_name(stage, forget_source)
    rounds = _round_dirs(run_dir)
    if max_round is not None:
        rounds = [r for r in rounds if int(r.name.split("_")[1]) <= int(max_round)]

    for round_dir in reversed(rounds):
        if _participated_in_round(round_dir, client_idx, stage=stage, summary_file=summary_file):
            return int(round_dir.name.split("_")[1])
    return None


def load_client_last_summary(
    run_dir: Union[str, Path],
    client: Union[int, str],
    max_round: Optional[int] = None,
    stage: str = "post_train",
    summary_file: str = "TOFU_SUMMARY.json",
    forget_source: Optional[int] = None,
) -> Tuple[Optional[int], Optional[Dict]]:
    run_dir = Path(run_dir)
    client_idx, client_name = _normalize_client(client)
    stage = _stage_name(stage, forget_source)
    round_id = find_last_participation_round(
        run_dir=run_dir,
        client=client_idx,
        max_round=max_round,
        stage=stage,
        summary_file=summary_file,
        forget_source=forget_source,
    )
    if round_id is None:
        return None, None

    summary_path = (
        run_dir
        / "eval_rounds"
        / f"round_{round_id}"
        / f"{client_name}_{stage}"
        / summary_file
    )
    payload = _load_json(summary_path)
    if payload is not None:
        return round_id, payload

    rounds = _round_dirs(run_dir)
    rounds = [r for r in rounds if int(r.name.split("_")[1]) <= round_id]
    for round_dir in reversed(rounds):
        fallback_path = round_dir / f"{client_name}_{stage}" / summary_file
        payload = _load_json(fallback_path)
        if payload is not None:
            return int(round_dir.name.split("_")[1]), payload
    return None, None


def _topology_from_fedperm_cache(
    run_dir: Path, round_id: int
) -> Tuple[Optional[List[int]], Dict[int, List[int]], List[int]]:
    cache_path = run_dir / "client_splits" / "fedperm_round_topology.json"
    payload = _load_json(cache_path)
    if not payload:
        return None, {}, []
    entry = (payload.get("rounds") or {}).get(str(int(round_id)))
    if not entry:
        return None, {}, []
    src_list = [int(x) for x in entry.get("source_clients", [])]
    st_map: Dict[int, List[int]] = {
        int(src): [int(t) for t in (tgts or [])]
        for src, tgts in (entry.get("source_to_targets", {}) or {}).items()
    }
    target_union = {t for tgts in st_map.values() for t in tgts}
    selected = sorted(set(src_list) | target_union)
    return selected, st_map, src_list


def _topology_for_round(
    run_dir: Path, round_id: int
) -> Tuple[Optional[List[int]], Dict[int, List[int]], List[int]]:
    """Return (selected_clients, source_to_targets, source_clients)."""
    round_dir = run_dir / "eval_rounds" / f"round_{int(round_id)}"
    meta = _load_json(round_dir / "participation.json")
    if meta:
        selected = [int(x) for x in meta.get("selected_clients", [])]
        src_list = [int(x) for x in meta.get("source_clients", [])]
        raw_map = meta.get("source_to_targets", {}) or {}
        st_map: Dict[int, List[int]] = {}
        for raw_src, tgts in raw_map.items():
            st_map[int(raw_src)] = [int(t) for t in (tgts or [])]
        if selected and (src_list or st_map):
            return selected, st_map, src_list
        if selected:
            fb_sel, fb_map, fb_src = _topology_from_fedperm_cache(run_dir, round_id)
            if fb_sel is not None and (fb_src or fb_map):
                return fb_sel, fb_map, fb_src
            return selected, st_map, src_list

    return _topology_from_fedperm_cache(run_dir, round_id)


def _role_for_client(
    client_idx: int,
    selected: List[int],
    source_clients: List[int],
    source_to_targets: Dict[int, List[int]],
) -> Optional[str]:
    if int(client_idx) not in {int(x) for x in selected}:
        return None
    src_set = {int(x) for x in source_clients}
    target_set = {t for tgts in source_to_targets.values() for t in tgts}
    c = int(client_idx)
    if c in src_set:
        return "source"
    if c in target_set:
        return "target"
    return "authorized"


def _eval_subdir_name(
    client_idx: int, stage: str, client_eval_prefix: str = "client"
) -> str:
    return f"{client_eval_prefix}{int(client_idx)}_{stage}"


def aggregate_role_metrics_by_round(
    run_dir: Union[str, Path],
    metrics: Iterable[str],
    max_round: Optional[int] = None,
    stage: str = "post_train",
    summary_file: str = "TOFU_SUMMARY.json",
    forget_source: Optional[int] = None,
    client_eval_prefix: str = "client",
) -> Dict[str, Any]:
    """
    Per round: classify selected clients into source / target / authorized from
    ``participation.json`` (or ``client_splits/fedperm_round_topology.json``), then
    average each metric within each role. Macro-average those per-round role means
    over rounds (only rounds where that role has at least one numeric value).

    ``client_eval_prefix`` defaults to ``"client"`` so summaries live under
    ``client{n}_{stage}`` (e.g. ``client0_post_train``). Must match on-disk eval layout.
    """
    run_dir = Path(run_dir)
    metrics = list(metrics)
    st = _stage_name(stage, forget_source)
    rounds = _round_dirs(run_dir)
    if max_round is not None:
        rounds = [r for r in rounds if int(r.name.split("_")[1]) <= int(max_round)]

    per_round: List[Dict[str, Dict[str, Optional[float]]]] = []

    for round_dir in rounds:
        round_id = int(round_dir.name.split("_")[1])
        selected, st_map, src_list = _topology_for_round(run_dir, round_id)
        if not selected:
            per_round.append({role: {m: None for m in metrics} for role in ROLES_ORDER})
            continue

        role_to_values: Dict[str, Dict[str, List[float]]] = {
            role: {m: [] for m in metrics} for role in ROLES_ORDER
        }

        for client_idx in selected:
            role = _role_for_client(client_idx, selected, src_list, st_map)
            if role is None:
                continue
            subdir = _eval_subdir_name(client_idx, st, client_eval_prefix)
            summary_path = round_dir / subdir / summary_file
            summary = _load_json(summary_path)
            if not summary:
                continue
            for m in metrics:
                v = summary.get(m)
                if isinstance(v, (int, float)):
                    role_to_values[role][m].append(float(v))

        round_means: Dict[str, Dict[str, Optional[float]]] = {}
        for role in ROLES_ORDER:
            round_means[role] = {}
            for m in metrics:
                vals = role_to_values[role][m]
                round_means[role][m] = sum(vals) / len(vals) if vals else None
        per_round.append(round_means)

    macro_mean_over_rounds: Dict[str, Dict[str, Optional[float]]] = {}
    for role in ROLES_ORDER:
        macro_mean_over_rounds[role] = {}
        for m in metrics:
            vals = [pr[role][m] for pr in per_round if pr[role][m] is not None]
            macro_mean_over_rounds[role][m] = (
                sum(vals) / len(vals) if vals else None
            )

    return {
        "run_dir": str(run_dir),
        "stage": st,
        "max_round": max_round,
        "forget_source": forget_source,
        "client_eval_prefix": client_eval_prefix,
        "per_round_role_means": per_round,
        "macro_mean_over_rounds": macro_mean_over_rounds,
    }

def aggregate_client_last_summaries(
    run_dir: Union[str, Path],
    clients: Iterable[Union[int, str]],
    metrics: Iterable[str],
    max_round: Optional[int] = None,
    stage: str = "post_train",
    summary_file: str = "TOFU_SUMMARY.json",
    forget_source: Optional[int] = None,
) -> Dict:
    per_client = {}
    metric_values = {metric: [] for metric in metrics}

    for client in clients:
        client_idx, client_name = _normalize_client(client)
        round_id, summary = load_client_last_summary(
            run_dir=run_dir,
            client=client_idx,
            max_round=max_round,
            stage=stage,
            summary_file=summary_file,
            forget_source=forget_source,
        )
        if summary is None:
            per_client[client_name] = {"round": None, "summary": None}
            continue

        selected_metrics = {
            metric: summary.get(metric) for metric in metrics if metric in summary
        }
        for metric, value in selected_metrics.items():
            if isinstance(value, (int, float)):
                metric_values[metric].append(float(value))

        per_client[client_name] = {
            "round": round_id,
            "summary": selected_metrics,
        }

    averages = {}
    for metric, values in metric_values.items():
        averages[metric] = sum(values) / len(values) if values else None

    return {
        "run_dir": str(Path(run_dir)),
        "stage": stage,
        "max_round": max_round,
        "forget_source": forget_source,
        "per_client": per_client,
        "average": averages,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate client metrics using each client's last participated round."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--clients", required=True, help="Comma-separated client ids, e.g. 1,6 or client1,client6")
    parser.add_argument("--metrics", required=True, help="Comma-separated metric names")
    parser.add_argument("--max-round", type=int, default=None)
    parser.add_argument("--stage", default="post_train")
    parser.add_argument("--summary-file", default="TOFU_SUMMARY.json")
    parser.add_argument("--forget-source", type=int, default=None)
    args = parser.parse_args()

    clients = [item.strip() for item in args.clients.split(",") if item.strip()]
    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    payload = aggregate_client_last_summaries(
        run_dir=args.run_dir,
        clients=clients,
        metrics=metrics,
        max_round=args.max_round,
        stage=args.stage,
        summary_file=args.summary_file,
        forget_source=args.forget_source,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
