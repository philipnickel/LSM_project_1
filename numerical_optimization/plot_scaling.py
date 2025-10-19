from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from postprocessing.utils import PLOTS_DIR, ensure_output_dir, ensure_style  # noqa: E402


OUTPUT_SUBDIR = PLOTS_DIR / "0_numba_vs_baseline"
CSV_PATH = Path("numerical_optimization/bench_results.csv")


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Benchmark results not found at {csv_path}")

    df = pd.read_csv(csv_path)
    if "Image Size" not in df.columns:
        raise KeyError("Expected 'Image Size' column in benchmark results.")
    if "Time (s)" not in df.columns or "Implementation" not in df.columns:
        raise KeyError("Benchmark file must include 'Time (s)' and 'Implementation' columns.")

    df = df.copy()
    df["Problem Size (Pixels)"] = (
        df["Image Size"]
        .str.split("x")
        .apply(lambda dims: int(dims[0]) * int(dims[1]) if isinstance(dims, list) and len(dims) == 2 else np.nan)
    )
    df = df.dropna(subset=["Problem Size (Pixels)"])
    df = df.sort_values("Problem Size (Pixels)")
    return df


def plot_scaling(df: pd.DataFrame, out_dir: Path) -> None:
    ensure_style()
    ensure_output_dir(out_dir)

    fig, ax = plt.subplots(figsize=(10, 6))
    implementations = df["Implementation"].unique()
    palette = sns.color_palette("tab10", n_colors=len(implementations))
    sns.lineplot(
        data=df,
        x="Problem Size (Pixels)",
        y="Time (s)",
        hue="Implementation",
        style="Implementation",
        markers=True,
        dashes=False,
        hue_order=implementations,
        palette=palette,
        linewidth=2.0,
        ax=ax,
    )

    ax.set(xscale="log", yscale="log")
    ax.set_xlabel("Problem size (pixels)")
    ax.set_ylabel("Wall time [s]")
    ax.set_title("Scaling of Mandelbrot implementations")

    if df["Problem Size (Pixels)"].nunique() > 1:
        x_ref = np.array(
            [
                df["Problem Size (Pixels)"].min(),
                df["Problem Size (Pixels)"].max(),
            ]
        )
        ref_start = df["Time (s)"].iloc[0]
        y_ref = ref_start * (x_ref / x_ref[0])
        ax.plot(
            x_ref,
            y_ref,
            "--",
            color="gray",
            linewidth=1.5,
            label="O(N) reference",
        )

    ax.legend(title="Implementation", frameon=False, loc="best")
    fig.tight_layout()
    output_path = out_dir / "numba_vs_baseline.pdf"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_data(CSV_PATH)
    plot_scaling(df, OUTPUT_SUBDIR)


if __name__ == "__main__":
    main()
