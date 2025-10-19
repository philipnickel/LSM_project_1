from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .utils import PLOTS_V2_DIR, ensure_output_dir, ensure_style


def main() -> None:
    ensure_style()

    csv_path = Path("numerical_optimization") / "bench_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run numerical_optimization/optimization.py first."
        )

    df = pd.read_csv(csv_path)
    df["Pixels"] = df["Image Size"].str.lower().str.split("x").apply(
        lambda parts: int(parts[0]) * int(parts[1])
    )
    df = df.sort_values("Pixels")

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(
        data=df,
        x="Pixels",
        y="Time (s)",
        hue="Implementation",
        marker="o",
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Pixels per image")
    ax.set_ylabel("Wall time [s]")
    ax.set_title("Baseline vs Numba Implementations")
    ax.legend(title="Implementation")
    fig.tight_layout()

    out_dir = ensure_output_dir(PLOTS_V2_DIR / "1_sanity")
    fig.savefig(out_dir / "1.1_baseline_vs_numba.pdf", bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
