import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_curve(run_dir: Path, client: str, eval_name: str):
    """Load pre/post metrics for one run directory."""
    xs = []
    ys = []
    rounds = sorted(
        (run_dir / "eval_rounds").glob("round_*"),
        key=lambda p: int(p.name.split("_")[1]),
    )
    for rpath in rounds:
        rid = int(rpath.name.split("_")[1])
        fpath = rpath / f"{client}_{"post_train"}" / "TOFU_EVAL.json"
        if not fpath.exists():
            print(fpath)
            continue
        try:
            payload = json.loads(fpath.read_text())
        except json.JSONDecodeError:
            continue
        if eval_name not in payload:
            continue
        xs.append(rid)
        ys.append(payload[eval_name]["agg_value"])
    return xs, ys


def plot_runs(method_root: Path, fl_method: str, ul_method: str, retain_split: str, epoch: int, ce0: int, alpha: int, source_client: str, target_client: str, eval_name: str):
    runs = {
        "full": method_root / f"full_{fl_method}_Llama-3.2-1B-Instruct-lora_{retain_split}",
        "retain": method_root / f"retain_{fl_method}_Llama-3.2-1B-Instruct-lora_{retain_split}",
        "fedperm": method_root / f"unlearn_{fl_method}_Llama-3.2-1B-Instruct-lora_{retain_split}_{ul_method}_filter{epoch}_CE{ce0}_Alpha{alpha}",
    }

    colors = {
        "full": "#1f77b4",
        "retain": "#2ca02c",
        "fedperm": "#d62728"
    }
    linestyles = {"source": "--", "target": "-"}

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, path in runs.items():
        if not path.exists():
            print(path)
            continue
        xs, ys = load_curve(path, client=source_client, eval_name=eval_name)
        if name == "fedperm":
            xs = [i+9 for i in xs]
        ax.plot(
            xs,
            ys,
            label=f"{name} {"post"} {source_client}",
            color=colors.get(name),
            linestyle=linestyles["source"],
            marker="o",
        )
        xs, ys = load_curve(path, client=target_client, eval_name=eval_name)
        if name == "fedperm":
            xs = [i+9 for i in xs]
        ax.plot(
            xs,
            ys,
            label=f"{name} {"post"} {target_client}",
            color=colors.get(name),
            linestyle=linestyles["target"],
            marker="^",
        )

    ax.set_xlabel("round")
    ax.set_ylabel(eval_name)
    ax.set_title(f"{method_root.name}_{retain_split}_{source_client}_{target_client}_{epoch}_{ce0}_{alpha}")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f"{method_root.name}_{eval_name}_ULM_{ul_method}.png"
    plt.savefig(out, dpi=300)
    plt.show()
    print(f"Saved plot to {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot FedPerm eval curves.")
    parser.add_argument("--method", default="fedperm_only", help="Root under saves/")
    parser.add_argument("--retain_split", default="retain90")
    parser.add_argument("--fl_method", default="FedAvg")
    parser.add_argument("--ul_method", default="SimNPO")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--ce0", type=int, default=1)
    parser.add_argument("--alpha", type=int, default=0.5)
    parser.add_argument("--client0", default="client0")
    parser.add_argument("--client1", default="client1")
    # parser.add_argument("--eval_name", default="forget_Q_A_ROUGE", help="e.g., model_utility or forget_utility")
    parser.add_argument("--eval_name", default="model_utility", help="e.g., model_utility or forget_utility")
    args = parser.parse_args()

    # method_root = Path("saves") / args.method
    method_root = Path("saves/TOFU") / args.method

    plot_runs(
        method_root=method_root,
        fl_method=args.fl_method,
        ul_method=args.ul_method,
        retain_split=args.retain_split,
        epoch=args.epoch,
        source_client=args.client0,
        target_client=args.client1,
        ce0=args.ce0,
        alpha=args.alpha,
        eval_name=args.eval_name,
    )


if __name__ == "__main__":
    main()
