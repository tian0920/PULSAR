#!/usr/bin/env python3
"""
离线重跑：仅对「本轮非 source、但在拓扑中作为 target」的客户端执行
client{n}_post_train__forget_merge_round_sources 的 TOFU 评测（与训练时 trainer 逻辑一致）。

不重新训练；需要磁盘上仍有该轮可用的客户端权重目录。优先使用：
  {run_dir}/round_{R}_target_client_{n}
若不存在且允许回退，可尝试 {run_dir}/round_{R}（聚合后模型，语义可能不等同于该客户端局部模型）。

用法示例（在仓库根目录、已激活项目环境）：
  python scripts/rerun_target_merge_eval.py --run-dir /path/to/output \\
    --forget-split forget10 --holdout-split holdout10

  # 只跑第 2、3 轮，且强制覆盖已有 TOFU_SUMMARY.json
  python scripts/rerun_target_merge_eval.py --run-dir ... --rounds 2,3 --force
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _parse_round_dirs(eval_rounds: Path) -> List[Tuple[int, Path]]:
    out: List[Tuple[int, Path]] = []
    if not eval_rounds.is_dir():
        return out
    for p in sorted(eval_rounds.iterdir()):
        if not p.is_dir() or not p.name.startswith("round_"):
            continue
        try:
            rid = int(p.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        out.append((rid, p))
    return out


def _load_participation(round_dir: Path) -> Optional[Dict[str, Any]]:
    path = round_dir / "participation.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _source_to_targets(raw: Any) -> Dict[int, List[int]]:
    if not raw or not isinstance(raw, dict):
        return {}
    out: Dict[int, List[int]] = {}
    for k, v in raw.items():
        try:
            sk = int(k)
        except (TypeError, ValueError):
            continue
        if not isinstance(v, list):
            continue
        out[sk] = [int(x) for x in v]
    return out


def _forget_eval_sources(participation: Dict[str, Any]) -> List[int]:
    """与 FederatedFinetuneTrainer._forget_eval_sources 一致。"""
    selected: Set[int] = {int(x) for x in participation.get("selected_clients", [])}
    raw_sources = participation.get("source_clients")
    if raw_sources:
        return [int(s) for s in raw_sources]
    st = _source_to_targets(participation.get("source_to_targets"))
    source_keys = sorted(st.keys())
    return sorted(s for s in source_keys if s in selected)


def _round_source_clients(participation: Dict[str, Any]) -> List[int]:
    raw = participation.get("source_clients")
    if raw:
        return [int(x) for x in raw]
    return sorted(_source_to_targets(participation.get("source_to_targets")).keys())


def _looks_like_checkpoint(d: Path) -> bool:
    if not d.is_dir():
        return False
    markers = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "pytorch_model.bin",
        "model.safetensors",
    )
    return any((d / m).exists() for m in markers)


def _resolve_checkpoint(
    run_dir: Path,
    round_id: int,
    client_idx: int,
    allow_server_fallback: bool,
) -> Tuple[Optional[Path], str]:
    target_dir = run_dir / f"round_{round_id}_target_client_{client_idx}"
    if _looks_like_checkpoint(target_dir):
        return target_dir, "target_client"
    if allow_server_fallback:
        server_dir = run_dir / f"round_{round_id}"
        if _looks_like_checkpoint(server_dir):
            return server_dir, "round_aggregate_fallback"
    return None, "missing"


def _run_eval_subprocess(
    repo_root: Path,
    python_exe: str,
    experiment: str,
    forget_split: str,
    holdout_split: str,
    task_name: str,
    ckpt_dir: Path,
    out_dir: Path,
    forget_indices_path: str,
    retain_indices_path: str,
    force_overwrite: bool,
    dry_run: bool,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "TOFU_SUMMARY.json"
    if summary.is_file() and summary.stat().st_size > 0 and not force_overwrite:
        print(f"[skip] existing summary: {summary}")
        return 0

    cmd: List[str] = [
        python_exe,
        str(repo_root / "src" / "eval.py"),
        f"experiment={experiment}",
        f"forget_split={forget_split}",
        f"holdout_split={holdout_split}",
        f"task_name={task_name}",
        f"model.model_args.pretrained_model_name_or_path={ckpt_dir}",
        f"paths.output_dir={out_dir}",
        f"hydra.run.dir={out_dir}",
        f"++eval.tofu.forget_indices_path={forget_indices_path}",
        f"++eval.tofu.forget_dataset_name={forget_split}",
        f"++eval.tofu.retain_indices_path={retain_indices_path}",
    ]
    if force_overwrite:
        cmd.append("++eval.tofu.overwrite=true")

    env = os.environ.copy()
    pp = str(repo_root / "src")
    env["PYTHONPATH"] = pp if not env.get("PYTHONPATH") else pp + os.pathsep + env["PYTHONPATH"]

    print("[run]", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=str(repo_root), env=env)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True, help="联邦实验 output_dir（含 eval_rounds、client_splits）")
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=None,
        help="client_splits 目录，默认 {run_dir}/client_splits",
    )
    parser.add_argument("--experiment", type=str, default="eval/tofu/default.yaml")
    parser.add_argument("--forget-split", type=str, default="forget10")
    parser.add_argument("--holdout-split", type=str, default="holdout10")
    parser.add_argument(
        "--client-stage-prefix",
        type=str,
        default="post_train",
        help="与训练时一致，一般为 post_train",
    )
    parser.add_argument(
        "--rounds",
        type=str,
        default="",
        help="逗号分隔轮次（与 eval_rounds/round_R 一致）；空表示全部",
    )
    parser.add_argument(
        "--clients",
        type=str,
        default="",
        help="仅这些 client 索引（逗号分隔）；空表示按拓扑自动选 target",
    )
    parser.add_argument(
        "--allow-server-fallback",
        action="store_true",
        help="无 target 专用 checkpoint 时尝试使用 round_R 聚合目录（可能不等价）",
    )
    parser.add_argument("--force", action="store_true", help="覆盖已有 TOFU_SUMMARY.json")
    parser.add_argument("--dry-run", action="store_true", help="只打印将执行的命令")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python 解释器")
    return parser


def run_with_args(args: argparse.Namespace) -> int:
    from trainer.federated.eval_routing import (
        ROUND_FORGET_MERGE_STAGE_SUFFIX,
        resolve_forget_indices_path,
        should_run_target_round_forget_merge_eval,
    )

    def build_merge_stage(client_idx: int, client_stage_prefix: str) -> str:
        return f"client{int(client_idx)}_{client_stage_prefix}{ROUND_FORGET_MERGE_STAGE_SUFFIX}"

    run_dir = args.run_dir.resolve()
    split_dir = (args.split_dir or (run_dir / "client_splits")).resolve()
    eval_rounds = run_dir / "eval_rounds"

    if not split_dir.is_dir():
        print(f"错误: split_dir 不存在: {split_dir}", file=sys.stderr)
        return 1

    round_filter: Optional[Set[int]] = None
    if args.rounds.strip():
        round_filter = {int(x.strip()) for x in args.rounds.split(",") if x.strip()}

    client_filter: Optional[Set[int]] = None
    if args.clients.strip():
        client_filter = {int(x.strip()) for x in args.clients.split(",") if x.strip()}

    exit_code = 0
    for round_id, round_path in _parse_round_dirs(eval_rounds):
        if round_filter is not None and round_id not in round_filter:
            continue
        participation = _load_participation(round_path)
        if not participation:
            print(f"[warn] skip round_{round_id}: no participation.json")
            continue

        source_to_targets = _source_to_targets(participation.get("source_to_targets"))
        round_sources = _round_source_clients(participation)
        active_sources = _forget_eval_sources(participation)
        selected = [int(x) for x in participation.get("selected_clients", [])]

        for client_idx in selected:
            if client_filter is not None and client_idx not in client_filter:
                continue
            if not should_run_target_round_forget_merge_eval(
                True,
                client_idx,
                round_source_clients=round_sources,
                source_to_targets=source_to_targets,
            ):
                continue

            stage = build_merge_stage(client_idx, args.client_stage_prefix)
            out_dir = round_path / stage

            forget_path = resolve_forget_indices_path(
                split_dir=str(split_dir),
                stage=stage,
                source_to_targets=source_to_targets,
                source_client_indices=sorted(source_to_targets.keys()),
                forget_source_idx=None,
                active_source_client_indices=active_sources,
            )
            if not forget_path or not os.path.isfile(forget_path):
                print(
                    f"[warn] round_{round_id} client{client_idx}: cannot resolve forget indices for merge eval"
                )
                exit_code = 1
                continue

            retain_path = os.path.join(split_dir, f"client{client_idx}_retain_indices.json")
            if not os.path.isfile(retain_path):
                print(f"[warn] round_{round_id} client{client_idx}: missing retain file {retain_path}")
                exit_code = 1
                continue

            ckpt, ckpt_kind = _resolve_checkpoint(
                run_dir, round_id, client_idx, allow_server_fallback=args.allow_server_fallback
            )
            if ckpt is None:
                print(
                    f"[warn] round_{round_id} client{client_idx}: no checkpoint "
                    f"(expected {run_dir}/round_{round_id}_target_client_{client_idx})"
                )
                exit_code = 1
                continue
            if ckpt_kind == "round_aggregate_fallback":
                print(
                    f"[warn] round_{round_id} client{client_idx}: using aggregate round_{round_id} "
                    "checkpoint — metrics may not match original client model"
                )

            rc = _run_eval_subprocess(
                repo_root=REPO_ROOT,
                python_exe=args.python,
                experiment=args.experiment,
                forget_split=args.forget_split,
                holdout_split=args.holdout_split,
                task_name=stage,
                ckpt_dir=ckpt,
                out_dir=out_dir,
                forget_indices_path=forget_path,
                retain_indices_path=retain_path,
                force_overwrite=args.force,
                dry_run=args.dry_run,
            )
            if rc != 0:
                exit_code = rc

    return exit_code


def main() -> int:
    args = build_arg_parser().parse_args()
    return run_with_args(args)


if __name__ == "__main__":
    sys.exit(main())
