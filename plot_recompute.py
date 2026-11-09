"""
Re-check figure: silently wrong orders against lie rate, by policy and lie type.

Usage (from the repository root):
    python plot_recompute.py
    python plot_recompute.py results/offline_fresh_results.csv results/figure_recompute.png
"""
import ast
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write to file without opening a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DEFAULT_IN = Path("results/offline_fresh_results.csv")
DEFAULT_OUT = Path("results/figure_recompute.png")
POLICIES = {  # policy: (marker, colour, line style)
    "trust": ("o", "C0", "-"),
    "per_step": ("s", "C1", "--"),
    "per_step+recompute": ("D", "C2", "-"),
}
MODES = {"false_completion": "False completion", "consistent": "Consistent lie"}


def make_figure(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)

    for row, fault in enumerate([0.0, 0.25]):
        for col, mode in enumerate(MODES):
            ax = axes[row, col]
            for policy, (marker, colour, style) in POLICIES.items():
                base = df[(df.policy == policy) & (df.lie_mode == "none") & (df.fault == fault)]
                sub = df[(df.policy == policy) & (df.lie_mode == mode) & (df.fault == fault)]
                sub = sub.sort_values("lie_rate")

                # Lie rate 0 is the honest baseline, shared by both lie types
                xs = [0.0] + list(sub.lie_rate)
                ys = [float(base.silent_wrong_per_order.iloc[0])] + list(sub.silent_wrong_per_order)
                bounds = [ast.literal_eval(s) for s in base.silent_wrong_ci] + \
                         [ast.literal_eval(s) for s in sub.silent_wrong_ci]
                lo = np.array([a for a, _ in bounds])
                hi = np.array([b for _, b in bounds])
                y = np.array(ys)
                ax.errorbar(xs, ys, yerr=np.vstack([y - lo, hi - y]), marker=marker,
                            color=colour, linestyle=style, capsize=3, label=policy)

            ax.set_title(f"{MODES[mode]}, write-drop {fault:g}")
            ax.set_xlabel("lie rate per decision")
            if col == 0:
                ax.set_ylabel("silently wrong per order")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IN
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    if not src.exists():
        sys.exit(f"missing {src}: check the path, or copy offline_fresh_results.csv into results/")
    make_figure(pd.read_csv(src), dst)


if __name__ == "__main__":
    main()