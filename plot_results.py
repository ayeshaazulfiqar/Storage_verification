"""Rebuild Figure 1 from results/full_results.csv. No API calls needed.

Usage:
    python plot_results.py
    python plot_results.py path/to/full_results.csv path/to/output.png
"""
import ast
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to file without opening a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DEFAULT_IN = Path("results/full_results.csv")
DEFAULT_OUT = Path("results/silent_wrong_vs_fault.png")
STYLES = {  # condition: (marker, line style, colour)
    "trust": ("o", "-", "C0"),
    "per_step": ("s", "-", "C1"),
    "end_only": ("^", "--", "C2"),
}


def make_figure(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for col, difficulty in enumerate(["easy", "hard"]):
        sub = df[df.difficulty == difficulty]

        # Top row: silently wrong orders, with 95% bootstrap intervals
        ax = axes[0, col]
        for condition, (marker, style, colour) in STYLES.items():
            c = sub[sub.condition == condition].sort_values("fault")
            y = c.silent_wrong_per_order.to_numpy()
            bounds = [ast.literal_eval(s) for s in c.silent_wrong_ci]
            lo = np.array([a for a, _ in bounds])
            hi = np.array([b for _, b in bounds])
            ax.errorbar(c.fault, y, yerr=np.vstack([y - lo, hi - y]), marker=marker,
                        linestyle=style, color=colour, capsize=3, label=condition)
        ax.set_title(f"{difficulty.capitalize()} orders: silently wrong")
        ax.set_xlabel("write-drop probability")
        ax.set_ylabel("silently wrong per order")

        # Bottom row: orders flagged by each policy
        ax = axes[1, col]
        per_step = sub[sub.condition == "per_step"].sort_values("fault")
        end_only = sub[sub.condition == "end_only"].sort_values("fault")
        ax.plot(per_step.fault, per_step.escalated_per_order, "s-", color="C1",
                label="per_step: escalated during pipeline")
        ax.plot(end_only.fault, end_only.escalated_per_order, "^--", color="C2",
                label="end_only: flagged after pipeline")
        ax.set_title(f"{difficulty.capitalize()} orders: flagged")
        ax.set_xlabel("write-drop probability")
        ax.set_ylabel("flagged per order")

        for row in range(2):
            axes[row, col].grid(alpha=0.3)
            axes[row, col].legend(fontsize=8)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IN
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    if not src.exists():
        sys.exit(f"missing {src}: run storage_verification.py first, or check the path")
    make_figure(pd.read_csv(src), dst)


if __name__ == "__main__":
    main()
