from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .utils import PLOTS_V2_DIR, ensure_output_dir, ensure_style

CACHE_DIR = Path("_exp_mlcache")
RUNS_IDX = CACHE_DIR / "runs_indexed.parquet"
SUITE = "scaling_im"


def load_runs() -> pd.DataFrame:
    if not RUNS_IDX.exists():
        raise FileNotFoundError(
            f"{RUNS_IDX} not found. Run postprocessing/data_loading.py first."
        )
    runs_idx = pd.read_parquet(RUNS_IDX)
    if not isinstance(runs_idx.index, pd.MultiIndex):
        raise ValueError("runs_indexed.parquet must be a MultiIndex table")
    if runs_idx.index.names[-1] != "Suite":
        runs_idx.index = runs_idx.index.set_names(
            [
                "Schedule",
                "Communication",
                "N Ranks",
                "Chunk Size",
                "Domain",
                "Image Size",
                "Run Id",
                "Suite",
            ]
        )
    try:
        subset = runs_idx.xs(SUITE, level="Suite").reset_index()
    except KeyError as exc:
        raise ValueError(f"Suite '{SUITE}' not present in indexed data") from exc

    subset["Pixels"] = subset["Image Size"].str.lower().str.split("x").apply(
        lambda parts: int(parts[0]) * int(parts[1])
    )
    subset = subset.sort_values("Pixels")
    subset["Config"] = (
        subset["Schedule"].astype(str) + " / " + subset["Communication"].astype(str)
    )
    return subset


def plot_wall_time(subset: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(
        data=subset,
        x="Pixels",
        y="Wall Time(s)",
        hue="Config",
        marker="o",
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Pixels per image")
    ax.set_ylabel("Wall time [s]")
    ax.set_title("Wall Time vs Problem Size")
    fig.tight_layout()
    fig.savefig(out_dir / "4.1_wall_time_vs_pixels.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_time_per_pixel(subset: pd.DataFrame, out_dir: Path) -> None:
    subset = subset.copy()
    subset["Time per Pixel"] = subset["Wall Time(s)"] / subset["Pixels"]
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(
        data=subset,
        x="Pixels",
        y="Time per Pixel",
        hue="Config",
        marker="o",
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Pixels per image")
    ax.set_ylabel("Time per pixel [s]")
    ax.set_title("Time per Pixel vs Problem Size")
    fig.tight_layout()
    fig.savefig(out_dir / "4.2_time_per_pixel.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_style()
    subset = load_runs()
    out_dir = ensure_output_dir(PLOTS_V2_DIR / "4_scaling_problem")
    plot_wall_time(subset, out_dir)
    plot_time_per_pixel(subset, out_dir)


if __name__ == "__main__":
    main()
