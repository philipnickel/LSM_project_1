from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import f
import pandas as pd
from pandas.core.dtypes.dtypes import pa
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

def plot_bars(ranks: pd.DataFrame, out_dir: Path) -> None:
    if ranks.empty:
        return

    df = ranks.reset_index().rename(columns={"rank": "Rank"})

    component_map = {"comp_time": "Compute", "comm_time": "Communication"}
    missing_components = [key for key in component_map if key not in df.columns]
    if missing_components:
        raise KeyError(f"Missing expected time columns: {', '.join(missing_components)}")

    long_df = df.melt(
        id_vars=["Config", "Rank", "Chunk Size"],
        value_vars=list(component_map.keys()),
        var_name="Component",
        value_name="Time",
    )
    long_df["Component"] = long_df["Component"].map(component_map)
    long_df = long_df.dropna(subset=["Time"])

    summary = (
        long_df.groupby(["Config", "Rank", "Chunk Size", "Component"], observed=False)["Time"]
        .mean()
        .reset_index()
    )

    if summary.empty:
        return

    pivot = summary.pivot_table(
        index=["Config", "Rank", "Chunk Size"],
        columns="Component",
        values="Time",
        observed=False,
    ).fillna(0.0)

    # Ensure both components are present even if missing in the data slice
    for label in component_map.values():
        if label not in pivot.columns:
            pivot[label] = 0.0
    pivot = pivot[list(component_map.values())]

    available_configs = pivot.index.get_level_values("Config").unique().tolist()
    config_order_labels = [config_label(s, c) for s, c in CONFIG_ORDER]
    configs = [cfg for cfg in config_order_labels if cfg in available_configs]
    if not configs:
        configs = available_configs

    ncols = 2 if len(configs) > 1 else 1
    nrows = int(np.ceil(len(configs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows), sharey=True)
    axes = np.atleast_1d(axes).flatten()

    component_palette = {
        "Compute": sns.color_palette("tab10")[0],
        "Communication": sns.color_palette("tab10")[1],
    }

    legend_handles: list[Patch] = []

    for ax, config in zip(axes, configs):
        config_data = pivot.xs(config, level="Config")
        ranks = sorted(config_data.index.get_level_values("Rank").unique().tolist())
        chunk_sizes = sorted(config_data.index.get_level_values("Chunk Size").unique().tolist())
        if not ranks or not chunk_sizes:
            ax.set_visible(False)
            continue

        x = np.arange(len(ranks))
        total_width = 0.8
        bar_width = total_width / max(len(chunk_sizes), 1)
        ymax = 0.0

        for idx, chunk_size in enumerate(chunk_sizes):
            chunk_slice = config_data.xs(chunk_size, level="Chunk Size").reindex(ranks, fill_value=0.0)
            compute_vals = chunk_slice.get("Compute", pd.Series(0.0, index=ranks)).to_numpy()
            comm_vals = chunk_slice.get("Communication", pd.Series(0.0, index=ranks)).to_numpy()

            positions = x + (idx - (len(chunk_sizes) - 1) / 2) * bar_width

            compute_bar = ax.bar(
                positions,
                compute_vals,
                width=bar_width,
                color=component_palette["Compute"],
                label="Compute" if (idx == 0 and not legend_handles) else "_nolegend_",
            )
            comm_bar = ax.bar(
                positions,
                comm_vals,
                width=bar_width,
                bottom=compute_vals,
                color=component_palette["Communication"],
                label="Communication" if (idx == 0 and not legend_handles) else "_nolegend_",
            )

            totals = compute_vals + comm_vals
            ymax = max(ymax, float(np.nanmax(totals)))
            for xpos, total in zip(positions, totals, strict=False):
                if np.isnan(total):
                    continue
                ax.text(
                    xpos,
                    total + 0.01 * max(ymax, 1.0),
                    f"{int(chunk_size)}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            if not legend_handles:
                legend_handles = [compute_bar, comm_bar]

        ax.set_title(config)
        ax.set_xticks(x)
        ax.set_xticklabels([str(rank) for rank in ranks])
        ax.set_xlabel("Rank")
        ax.set_ylim(top=ymax * 1.15 if ymax else 1)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    for extra_ax in axes[len(configs) :]:
        extra_ax.remove()

    axes[0].set_ylabel("Mean time [s]")

    if legend_handles:
        labels = ["Compute", "Communication"]
        fig.legend(
            [h[0] for h in legend_handles],
            labels,
            loc="upper center",
            ncol=len(labels),
            frameon=False,
        )

    fig.suptitle("Rank mean compute vs communication time", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
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
