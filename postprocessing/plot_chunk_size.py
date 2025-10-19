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
    ensure_output_dir,
    ensure_style,
    ensure_config_level,
    load_suite_chunks,
    load_suite_ranks,
    load_suite_runs,
)

SUITE = "chunks"


def plot_rank_bars(
    ranks_idx: pd.DataFrame,
    chunk_counts: pd.Series,
    out_dir: Path,
) -> None:
    sum_cols = [
        col
        for col in ("comp_time", "comm_time", "comm_send_time", "comm_recv_time")
        if col in ranks_idx.columns
    ]
    if "comp_time" not in sum_cols:
        raise ValueError("Rank data must include 'comp_time' column")

    aggregated = ranks_idx[sum_cols].groupby(level=["Domain", "Chunk Size", "rank"]).sum()
    if aggregated.empty:
        return

    comp_total = aggregated["comp_time"].fillna(0.0)
    if "comm_time" in aggregated.columns:
        comm_total = aggregated["comm_time"].fillna(0.0)
    else:
        comm_total = (
            aggregated.get("comm_send_time", 0.0).fillna(0.0)
            + aggregated.get("comm_recv_time", 0.0).fillna(0.0)
        )

    domains = aggregated.index.get_level_values("Domain").unique().tolist()
    chunk_sizes = sorted(aggregated.index.get_level_values("Chunk Size").unique())
    palette = sns.color_palette("viridis", len(chunk_sizes)) if chunk_sizes else sns.color_palette("viridis", 1)

    for domain in domains:
        comp_matrix = (
            comp_total.xs(domain, level="Domain")
            .unstack("Chunk Size")
            .reindex(columns=chunk_sizes)
            .fillna(0.0)
        )
        comm_matrix = (
            comm_total.xs(domain, level="Domain")
            .unstack("Chunk Size")
            .reindex(columns=chunk_sizes)
            .fillna(0.0)
        )
        ranks = comp_matrix.index.tolist()
        if not ranks:
            continue

        counts_matrix = (
            chunk_counts.xs(domain, level="Domain")
            .unstack("Chunk Size")
            .reindex(index=ranks, columns=chunk_sizes)
            .fillna(0)
        )

        width = 0.8 / max(len(chunk_sizes), 1)
        fig, ax = plt.subplots(figsize=(12, 6))

        for idx, chunk_size in enumerate(chunk_sizes):
            offsets = (idx - (len(chunk_sizes) - 1) / 2) * width
            positions = [r + offsets for r in ranks]
            comp_vals = comp_matrix[chunk_size].reindex(ranks).values
            comm_vals = comm_matrix[chunk_size].reindex(ranks).values
            totals = comp_vals + comm_vals
            counts = counts_matrix[chunk_size].reindex(ranks).values

            ax.bar(
                positions,
                comp_vals,
                width=width,
                color=palette[idx],
                alpha=0.85,
            )
            ax.bar(
                positions,
                comm_vals,
                width=width,
                bottom=comp_vals,
                color=palette[idx],
                alpha=0.35,
            )

            for x, total, count in zip(positions, totals, counts):
                if total <= 0:
                    continue
                ax.text(
                    x,
                    total,
                    f"{total:.2f}\n({int(count)} chunks)",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        ax.set_xticks(ranks)
        ax.set_xlabel("Rank")
        ax.set_ylabel("Time [s]")
        ax.set_title(f"Domain {domain}")
        ax.grid(axis="y", alpha=0.3)

        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, color=palette[idx], label=f"Chunk {int(cs)}")
            for idx, cs in enumerate(chunk_sizes)
        ]
        if legend_handles:
            ax.legend(handles=legend_handles, title="Chunk size")

        fig.tight_layout()
        fname = f"2.2_rank_time_breakdown_domain_{_sanitize(str(domain))}.pdf"
        fig.savefig(out_dir / fname, bbox_inches="tight")
        plt.close(fig)


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


def _sanitize(text: str) -> str:
    safe = text
    for ch in " [](),/":
        safe = safe.replace(ch, "_")
    return safe.strip("_")


def main() -> None:
    ensure_style()
    out_dir = ensure_output_dir(PLOTS_DIR / "2_load_balancing")
    ranks_idx = ensure_config_level(load_suite_ranks(SUITE, as_index=True))
    chunk_counts = ranks_idx.groupby(level=["Domain", "Chunk Size", "rank"], observed=False)[
        "chunks"
    ].sum()

    chunks_idx = ensure_config_level(load_suite_chunks(SUITE, as_index=True))

    plot_rank_bars(ranks_idx, chunk_counts, out_dir)
    plot_heatmaps(chunks_idx["comp_time"], out_dir)


if __name__ == "__main__":
    main()
