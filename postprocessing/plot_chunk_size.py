from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from postprocessing.utils import (
    PLOTS_DIR,
    config_palette,
    ensure_output_dir,
    ensure_style,
    ensure_config_level,
    load_suite_chunks,
    load_suite_ranks,
    load_suite_runs,
)

SUITE = "chunks"


def select_representative_runs(runs_idx: pd.DataFrame) -> set[str]:
    series = (
        runs_idx["Wall Time(s)"]
        .groupby(level=["Chunk Size", "Config", "Run Id"], observed=False)
        .mean()
    )

    def _pick_median(group: pd.Series) -> str | None:
        if group.empty:
            return None
        ordered = group.sort_values()
        return ordered.index[len(ordered) // 2][-1]

    reps = series.groupby(level=["Chunk Size", "Config"], observed=False).apply(_pick_median)
    return {rep for rep in reps.dropna().astype(str)}


def plot_runtime_vs_chunk_size(runs: pd.DataFrame, out_dir: Path) -> None:
    palette = config_palette(runs["Config"].unique())
    g = sns.relplot(
        data=runs,
        x="Chunk Size",
        y="Wall Time(s)",
        hue="Config",
        style="Domain",
        col="Chunk Size",
        kind="line",
        estimator=None,
        markers=True,
        palette=palette,
        height=4,
        aspect=1.2,
        col_wrap=3,
    )
    g.set_axis_labels("Chunk size", "Wall time [s]")
    for ax in g.axes.flatten():
        ax.set_xscale("log")
        ax.set_yscale("log")
    g.fig.suptitle("Runtime vs Chunk Size", y=1.02)
    g.fig.savefig(out_dir / "2.1_runtime_vs_chunk_size.pdf", bbox_inches="tight")
    plt.close(g.fig)


def prep_rank_totals(ranks: pd.DataFrame) -> pd.DataFrame:
    ranks = ranks.copy()
    ranks["Metric"] = "Comp Total"
    comp = ranks[["Chunk Size", "Config", "Run Id", "rank", "Metric", "comp_time"]]
    comp = comp.rename(columns={"comp_time": "Time"})

    comm_total = ranks.get("Comm Total")
    if comm_total is None:
        comm_total = (
            ranks.get("comm_send_time", 0.0).fillna(0.0)
            + ranks.get("comm_recv_time", 0.0).fillna(0.0)
        )
    comm_df = ranks[["Chunk Size", "Config", "Run Id", "rank"]].copy()
    comm_df["Metric"] = "Comm Total"
    comm_df["Time"] = comm_total

    return pd.concat([comp, comm_df], ignore_index=True)


def plot_rank_bars(
    ranks: pd.DataFrame,
    chunk_counts: pd.DataFrame,
    out_dir: Path,
) -> None:
    melted = prep_rank_totals(ranks)
    merged = melted.merge(
        chunk_counts,
        on=["Chunk Size", "Config", "Run Id", "rank"],
        how="left",
    )

    g = sns.catplot(
        data=merged,
        kind="bar",
        x="rank",
        y="Time",
        hue="Metric",
        col="Chunk Size",
        row="Config",
        estimator=sum,
        palette="pastel",
        sharey=False,
        height=4,
        aspect=1.2,
    )
    g.set_axis_labels("Rank", "Time [s]")
    g.fig.suptitle("Computation vs Communication per Rank", y=1.02)

    grouped = list(merged.groupby(["Chunk Size", "Config"]))
    for ax, ((chunk_size, config), data) in zip(g.axes.flatten(), grouped):
        counts = (
            data.drop_duplicates(subset=["rank", "Run Id"])
            .groupby("rank", observed=False)["Chunks Processed"]
            .sum()
        )
        for patch in ax.patches:
            rank_val = int(patch.get_x() + patch.get_width() / 2)
            count = counts.get(rank_val)
            if pd.notna(count):
                ax.text(
                    patch.get_x() + patch.get_width() / 2,
                    patch.get_height(),
                    f"{int(count)}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    g.fig.subplots_adjust(top=0.88)
    g.fig.savefig(out_dir / "2.2_rank_time_breakdown.pdf", bbox_inches="tight")
    plt.close(g.fig)


def plot_heatmaps(chunks: pd.DataFrame, out_dir: Path) -> None:
    aggregated = (
        chunks.groupby(
            ["Chunk Size", "chunk_id"],
            observed=False,
        )["comp_time"]
        .mean()
        .reset_index()
    )

    pivot = aggregated.pivot_table(
        index="Chunk Size",
        columns="chunk_id",
        values="comp_time",
        aggfunc="mean",
    ).sort_index()

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        pivot,
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


def _sanitize(text: str) -> str:
    safe = text
    for ch in " [](),/":
        safe = safe.replace(ch, "_")
    return safe.strip("_")


def main() -> None:
    ensure_style()
    out_dir = ensure_output_dir(PLOTS_DIR / "2_load_balancing")
    runs_idx = ensure_config_level(load_suite_runs(SUITE, as_index=True))
    reps = select_representative_runs(runs_idx)
    run_mask = runs_idx.index.get_level_values("Run Id").isin(reps)
    runs = runs_idx[run_mask].reset_index()

    ranks_idx = ensure_config_level(load_suite_ranks(SUITE, as_index=True))
    rank_mask = ranks_idx.index.get_level_values("Run Id").isin(reps)
    ranks = ranks_idx[rank_mask].reset_index()

    chunks_idx = ensure_config_level(load_suite_chunks(SUITE, as_index=True))
    chunk_mask = chunks_idx.index.get_level_values("Run Id").isin(reps)
    filtered_chunks_idx = chunks_idx[chunk_mask]
    chunk_counts = (
        filtered_chunks_idx.groupby(
            level=["Chunk Size", "Config", "Run Id", "rank"],
            observed=False,
        )
        .size()
        .rename("Chunks Processed")
        .reset_index()
    )
    chunks = filtered_chunks_idx.reset_index()

    plot_runtime_vs_chunk_size(runs, out_dir)
    plot_rank_bars(ranks, chunk_counts, out_dir)
    plot_heatmaps(chunks, out_dir)


if __name__ == "__main__":
    main()
