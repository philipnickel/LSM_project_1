from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from postprocessing.utils import (
    CONFIG_ORDER,
    PLOTS_DIR,
    ensure_output_dir,
    ensure_style,
    ensure_config_level,
    load_suite_chunks,
    load_suite_ranks,
    config_label,
    config_palette,
)

SUITE = "chunks"

def plot_heatmaps(chunk_times: pd.Series, out_dir: Path) -> None:
    if chunk_times.empty:
        return

    means = (
        chunk_times.groupby(
            level=["Chunk Size", "chunk_id"],
            observed=False,
        )
        .mean()
        .unstack("chunk_id")
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        means,
        cmap="mako",
        ax=ax,
        cbar_kws={"label": "Mean chunk compute time [s]"},
    )
    ax.set_xlabel("Chunk ID")
    ax.set_ylabel("Chunk Size")
    ax.set_title("Chunk Compute Time vs Chunk Size")
    fig.tight_layout()
    fig.savefig(out_dir / "2.4_chunk_size_vs_chunk_id_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

def plot_bars(ranks: pd.Series, out_dir: Path) -> None:

    fig, ax = plt.subplots(figsize=(20, 10))
    sns.barplot(
        ranks, 
        x="rank",
        y="comp_time",
        hue="Config",
        errorbar=None,

    )
    ax.set_xlabel("Rank")
    ax.set_ylabel("Time")
    ax.set_title("Chunk Compute Time vs Chunk Size")
    fig.tight_layout()
    fig.savefig(out_dir / "2.0Ranks.pdf", bbox_inches="tight")
    plt.close(fig)




def main() -> None:
    ensure_style()
    out_dir = ensure_output_dir(PLOTS_DIR / "2_load_balancing")
    ranks_idx = ensure_config_level(load_suite_ranks(SUITE, as_index=True))
    chunk_counts = ranks_idx.groupby(level=["Domain", "Chunk Size", "rank"], observed=False)[
        "chunks"
    ].sum()

    chunks_idx = ensure_config_level(load_suite_chunks(SUITE, as_index=True))

    plot_heatmaps(chunks_idx["comp_time"], out_dir)
    plot_bars(ranks_idx, out_dir)


if __name__ == "__main__":
    main()
