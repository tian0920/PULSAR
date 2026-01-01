import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_curve(run_dir: Path, client: str, eval_name: str):
    """Load pre/post metrics for one run directory."""
    series = {"pre": [], "post": []}
    rounds = sorted(
        (run_dir / "eval_rounds").glob("round_*"),
        key=lambda p: int(p.name.split("_")[1]),
    )
    for rpath in rounds:
        rid = int(rpath.name.split("_")[1])
        for stage_key, stage_dir in (("pre", "pre_train"), ("post", "post_train")):
            fpath = rpath / f"{client}_{stage_dir}" / "TOFU_SUMMARY.json"
            if not fpath.exists():
                continue
            try:
                payload = json.loads(fpath.read_text())
            except json.JSONDecodeError:
                continue
            if eval_name not in payload:
                continue
            series[stage_key].append((rid, payload[eval_name]))
    return series


def plot_runs(method_root: Path, retain_split: str, epoch: int, client: str, eval_name: str):
    runs = {
        "full": method_root / f"full_Llama-3.2-1B-Instruct-lora_{retain_split}_{epoch}",
        "retain": method_root / f"retain_Llama-3.2-1B-Instruct-lora_{retain_split}_{epoch}",
        "fedperm": method_root / f"fedperm_Llama-3.2-1B-Instruct-lora_{retain_split}_{epoch}",
        "fedperm_L0.01": method_root / f"fedperm_L0.01_Llama-3.2-1B-Instruct-lora_{retain_split}_{epoch}",
        "fedperm_L10": method_root / f"fedperm_L10_Llama-3.2-1B-Instruct-lora_{retain_split}_{epoch}",

    }

    colors = {
        "full": "#1f77b4",
        "retain": "#2ca02c",
        "fedperm": "#d62728",
        "fedperm_L0.01": "#9467bd",  # distinct purple, valid hex
        "fedperm_L10": "#000000",  # distinct purple, valid hex
    }
    linestyles = {"pre": "--", "post": "-"}

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, path in runs.items():
        if not path.exists():
            print(path)
            continue
        series = load_curve(path, client=client, eval_name=eval_name)
        for stage in ("pre", "post"):
            if not series[stage]:
                continue
            xs, ys = zip(*series[stage])
            ax.plot(
                xs,
                ys,
                label=f"{name} {stage}",
                color=colors.get(name),
                linestyle=linestyles[stage],
                marker="o",
            )

    ax.set_xlabel("round")
    ax.set_ylabel(eval_name)
    ax.set_title(f"{method_root.name}_{retain_split}_{client}_{epoch}")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = f"{method_root.name}_{retain_split}_{eval_name}_{client}_{epoch}.png"
    plt.savefig(out, dpi=300)
    plt.show()
    print(f"Saved plot to {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot FedPerm eval curves.")
    parser.add_argument("--method", default="fedperm", help="Root under saves/")
    parser.add_argument("--retain_split", default="retain90")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--client", default="client1")
    parser.add_argument("--eval_name", default="model_utility", help="e.g., model_utility or forget_utility")
    args = parser.parse_args()

    method_root = Path("saves") / args.method
    # method_root = Path("results/alpha=0.7") / args.method

    plot_runs(
        method_root=method_root,
        retain_split=args.retain_split,
        epoch=args.epoch,
        client=args.client,
        eval_name=args.eval_name,
    )


if __name__ == "__main__":
    main()
