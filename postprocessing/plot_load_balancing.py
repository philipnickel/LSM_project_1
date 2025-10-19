# %% load_balancing suite: per-domain breakdowns -----------------------------
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
import seaborn as sns

plt.style.use("science")
sns.set_style("whitegrid")
sns.set_palette("colorblind")
sns.set_context("talk")

CACHE_DIR = Path("_exp_mlcache")
RUNS_PATH = CACHE_DIR / "runs_df.parquet"
RANKS_PATH = CACHE_DIR / "ranks_df.parquet"
CHUNKS_PATH = CACHE_DIR / "chunks_df.parquet"
SUITE = "load_balancing"

for path in (RUNS_PATH, RANKS_PATH, CHUNKS_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run postprocessing/data_loading.py first."
        )

runs = pd.read_parquet(RUNS_PATH)
ranks = pd.read_parquet(RANKS_PATH)
chunks = pd.read_parquet(CHUNKS_PATH)

for df in (runs, ranks, chunks):
    if "suite" not in df.columns:
        raise KeyError("Column 'suite' missing. Re-run data_loading.py.")

runs = runs[runs["suite"] == SUITE].copy()
ranks = ranks[ranks["suite"] == SUITE].copy()
chunks = chunks[chunks["suite"] == SUITE].copy()

if runs.empty or ranks.empty or chunks.empty:
    raise ValueError(f"Missing data for suite '{SUITE}'.")

for df in (ranks, chunks):
    df["config"] = df["schedule"].astype(str) + " / " + df["communication"].astype(str)

plots_dir = Path("Plots") / SUITE
plots_dir.mkdir(parents=True, exist_ok=True)


def sanitize(value: object) -> str:
    text = str(value)
    for ch in " [](),":
        text = text.replace(ch, "_")
    return text.strip("_") or "domain"


domains = ranks["domain"].dropna().unique()

for domain in domains:
    domain_key = sanitize(domain)

    rank_subset = ranks[ranks["domain"] == domain].copy()
    if rank_subset.empty:
        continue

    if "comm_send_time" in rank_subset.columns and "comm_recv_time" in rank_subset.columns:
        rank_subset["comm_total"] = (
            rank_subset["comm_send_time"].fillna(0.0) + rank_subset["comm_recv_time"].fillna(0.0)
        )
    elif "comm_time" in rank_subset.columns:
        rank_subset["comm_total"] = rank_subset["comm_time"].fillna(0.0)
    else:
        rank_subset["comm_total"] = 0.0
    rank_subset = rank_subset.sort_values(["config", "run_id", "rank"])
    unique_series = rank_subset[["run_id", "config"]].drop_duplicates()
    num_series = len(unique_series)
    ranks_order = sorted(rank_subset["rank"].unique())
    x_positions = np.arange(len(ranks_order), dtype=float)
    width = 0.8 / max(num_series, 1)
    colors = sns.color_palette("colorblind", len(rank_subset["config"].unique()))
    config_colors = dict(zip(sorted(rank_subset["config"].unique()), colors))

    fig, ax = plt.subplots(figsize=(12, 8))
    for idx, (run_id, config) in enumerate(unique_series.itertuples(index=False, name=None)):
        data = rank_subset[(rank_subset["run_id"] == run_id) & (rank_subset["config"] == config)]
        comp = data.set_index("rank")["comp_time"].reindex(ranks_order, fill_value=0.0)
        comm = data.set_index("rank")["comm_total"].reindex(ranks_order, fill_value=0.0)
        offset = x_positions + (idx - (num_series - 1) / 2) * width
        label_base = f"{config} (run {str(run_id)[-4:]})"
        ax.bar(
            offset,
            comp.values,
            width=width,
            label=f"{label_base} compute",
            color=config_colors[config],
        )
        ax.bar(
            offset,
            comm.values,
            width=width,
            bottom=comp.values,
            label=f"{label_base} comm",
            color=config_colors[config],
            alpha=0.4,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(r) for r in ranks_order])
    ax.set_xlabel("Rank")
    ax.set_ylabel("Time [s]")
    ax.set_title(f"Per-Rank Time Breakdown\nDomain {domain}")
    ax.legend(ncol=2, fontsize="small", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.subplots_adjust(right=0.75)
    bar_path = plots_dir / f"{domain_key}_rank_time_breakdown.pdf"
    fig.savefig(bar_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] saved {bar_path}")

    chunk_subset = chunks[chunks["domain"] == domain].copy()
    if chunk_subset.empty:
        continue

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = sns.scatterplot(
        data=chunk_subset,
        x="chunk_id",
        y="comp_time",
        hue="rank",
        style="config",
        palette="tab10",
        alpha=0.7,
        ax=ax,
    )
    ax.set_title(f"Chunk Hotspots by Rank\nDomain {domain}")
    ax.set_xlabel("Chunk ID")
    ax.set_ylabel("Compute time [s]")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
    fig.tight_layout()
    scatter_path = plots_dir / f"{domain_key}_chunk_hotspots.pdf"
    fig.savefig(scatter_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] saved {scatter_path}")
