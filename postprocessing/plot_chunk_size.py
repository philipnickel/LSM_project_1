from __future__ import annotations

import sys
from pathlib import Path
from numbers import Integral

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

    aggregated = (
        ranks_idx[sum_cols]
        .groupby(level=["Domain", "Chunk Size", "rank"], observed=False)
        .sum()
        .reset_index()
    )
    if aggregated.empty:
        return

    comp_series = aggregated["comp_time"].fillna(0.0)
    if "comm_time" in aggregated.columns:
        comm_series = aggregated["comm_time"].fillna(0.0)
    else:
        send_series = (
            aggregated["comm_send_time"].fillna(0.0)
            if "comm_send_time" in aggregated.columns
            else pd.Series(0.0, index=aggregated.index)
        )
        recv_series = (
            aggregated["comm_recv_time"].fillna(0.0)
            if "comm_recv_time" in aggregated.columns
            else pd.Series(0.0, index=aggregated.index)
        )
        comm_series = send_series + recv_series

    plot_data = aggregated[["Domain", "Chunk Size", "rank"]].copy()
    plot_data["comp_time"] = comp_series
    plot_data["comm_time"] = comm_series

    if isinstance(chunk_counts, pd.DataFrame):
        if chunk_counts.shape[1] != 1:
            raise ValueError("chunk_counts must have exactly one value column")
        chunk_counts_series = chunk_counts.iloc[:, 0]
    else:
        chunk_counts_series = chunk_counts

    counts_df = chunk_counts_series.rename("chunks").reset_index()
    plot_data = plot_data.merge(counts_df, on=["Domain", "Chunk Size", "rank"], how="left")
    plot_data["chunks"] = plot_data["chunks"].fillna(0)

    unique_chunk_sizes = sorted(plot_data["Chunk Size"].unique())
    chunk_palette = sns.color_palette("viridis", max(len(unique_chunk_sizes), 1))
    color_map = dict(zip(unique_chunk_sizes, chunk_palette))

    def _chunk_label(value: object) -> str:
        if isinstance(value, Integral):
            return f"Chunk {int(value)}"
        if isinstance(value, float):
            return f"Chunk {int(value)}" if value.is_integer() else f"Chunk {value:g}"
        return f"Chunk {value}"

    for domain, domain_df in plot_data.groupby("Domain", sort=False):
        chunk_sizes = sorted(domain_df["Chunk Size"].unique())
        ranks = sorted(domain_df["rank"].unique())
        if not ranks:
            continue

        width = 0.8 / max(len(chunk_sizes), 1)
        fig, ax = plt.subplots(figsize=(12, 6))

        for idx, chunk_size in enumerate(chunk_sizes):
            offset = (idx - (len(chunk_sizes) - 1) / 2) * width
            positions = [rank + offset for rank in ranks]
            chunk_df = (
                domain_df[domain_df["Chunk Size"] == chunk_size]
                .set_index("rank")
                .reindex(ranks)
            )
            comp_vals = chunk_df["comp_time"].fillna(0.0).to_numpy()
            comm_vals = chunk_df["comm_time"].fillna(0.0).to_numpy()
            counts = chunk_df["chunks"].fillna(0).to_numpy()
            totals = comp_vals + comm_vals

            ax.bar(
                positions,
                comp_vals,
                width=width,
                color=color_map.get(chunk_size),
                alpha=0.85,
            )
            ax.bar(
                positions,
                comm_vals,
                width=width,
                bottom=comp_vals,
                color=color_map.get(chunk_size),
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
            plt.Rectangle((0, 0), 1, 1, color=color_map.get(cs), label=_chunk_label(cs))
            for cs in chunk_sizes
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
