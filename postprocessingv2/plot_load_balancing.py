from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .utils import PLOTS_V2_DIR, ensure_output_dir, ensure_style

CACHE_DIR = Path("_exp_mlcache")
CHUNKS_IDX = CACHE_DIR / "chunks_indexed.parquet"
RANKS_IDX = CACHE_DIR / "ranks_indexed.parquet"
SUITE = "load_balancing"


def _load_indexed(path: Path, expected_levels: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run postprocessing/data_loading.py first."
        )
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError(f"{path.name} must contain a MultiIndex table")
    if df.index.names != expected_levels:
        df.index = df.index.set_names(expected_levels)
    return df


def _sanitize(text: str) -> str:
    safe = text
    for ch in " [](),/":
        safe = safe.replace(ch, "_")
    return safe.strip("_")


def prepare_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    chunks_idx = _load_indexed(
        CHUNKS_IDX,
        [
            "Schedule",
            "Communication",
            "N Ranks",
            "Chunk Size",
            "Domain",
            "Image Size",
            "Run Id",
            "Suite",
            "rank",
            "chunk_id",
        ],
    )
    ranks_idx = _load_indexed(
        RANKS_IDX,
        [
            "Schedule",
            "Communication",
            "N Ranks",
            "Chunk Size",
            "Domain",
            "Image Size",
            "Run Id",
            "Suite",
            "rank",
        ],
    )

    try:
        chunks = chunks_idx.xs(SUITE, level="Suite").reset_index()
        ranks = ranks_idx.xs(SUITE, level="Suite").reset_index()
    except KeyError as exc:
        raise ValueError(f"Suite '{SUITE}' not present in indexed data") from exc

    chunks["Config"] = (
        chunks["Schedule"].astype(str) + " / " + chunks["Communication"].astype(str)
    )
    ranks["Config"] = (
        ranks["Schedule"].astype(str) + " / " + ranks["Communication"].astype(str)
    )

    if {"comm_send_time", "comm_recv_time"}.issubset(ranks.columns):
        ranks["Comm Total"] = (
            ranks["comm_send_time"].fillna(0.0) + ranks["comm_recv_time"].fillna(0.0)
        )
    elif "comm_time" in ranks.columns:
        ranks["Comm Total"] = ranks["comm_time"].fillna(0.0)
    else:
        ranks["Comm Total"] = 0.0

    return chunks, ranks


def plot_rank_chunk_counts(chunks: pd.DataFrame, out_dir: Path) -> None:
    counts = (
        chunks.groupby(["Domain", "Communication", "Schedule", "rank"], observed=False)
        .size()
        .reset_index(name="Chunks Processed")
    )
    for (domain, comm), subset in counts.groupby(["Domain", "Communication"]):
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(
            data=subset,
            x="rank",
            y="Chunks Processed",
            hue="Schedule",
            ax=ax,
        )
        ax.set_title(f"Per-Rank Chunk Counts\nDomain {domain}, Comm={comm}")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Chunks processed")
        fig.tight_layout()
        domain_key = _sanitize(str(domain))
        fig.savefig(
            out_dir / f"2.1_chunk_counts_{domain_key}_{comm}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_rank_time_breakdown(ranks: pd.DataFrame, out_dir: Path) -> None:
    summary = ranks.copy()
    summary = summary.groupby(
        ["Domain", "Communication", "Config", "rank"], observed=False
    )[["comp_time", "Comm Total"]].mean().reset_index()

    for (domain, comm, config), subset in summary.groupby(["Domain", "Communication", "Config"]):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(subset["rank"], subset["comp_time"], label="Compute")
        ax.bar(
            subset["rank"],
            subset["Comm Total"],
            bottom=subset["comp_time"],
            label="Communication",
            alpha=0.6,
        )
        ax.set_title(f"Per-Rank Time Breakdown\n{config}, Domain {domain}, Comm={comm}")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Time [s]")
        ax.legend()
        fig.tight_layout()
        domain_key = _sanitize(str(domain))
        config_key = _sanitize(config)
        fig.savefig(
            out_dir / f"2.2_rank_time_{domain_key}_{comm}_{config_key}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_chunk_time_distribution(chunks: pd.DataFrame, out_dir: Path) -> None:
    for domain, domain_df in chunks.groupby("Domain"):
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.ecdfplot(
            data=domain_df,
            x="comp_time",
            hue="Config",
            ax=ax,
        )
        ax.set_title(f"Chunk Compute Time ECDF\nDomain {domain}")
        ax.set_xlabel("Chunk compute time [s]")
        ax.set_ylabel("ECDF")
        fig.tight_layout()
        fig.savefig(
            out_dir / f"2.3_chunk_ecdf_{_sanitize(str(domain))}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.histplot(
            data=domain_df,
            x="comp_time",
            hue="Config",
            element="step",
            stat="density",
            common_norm=False,
            ax=ax,
        )
        ax.set_title(f"Chunk Compute Time Distribution\nDomain {domain}")
        ax.set_xlabel("Chunk compute time [s]")
        fig.tight_layout()
        fig.savefig(
            out_dir / f"2.3_chunk_hist_{_sanitize(str(domain))}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_chunk_heatmaps(chunks: pd.DataFrame, out_dir: Path) -> None:
    for (domain, config), subset in chunks.groupby(["Domain", "Config"]):
        pivot = subset.pivot_table(
            index="rank",
            columns="chunk_id",
            values="comp_time",
            aggfunc="mean",
        ).sort_index()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot, cmap="mako", ax=ax)
        ax.set_title(f"Chunk Compute Times Heatmap\n{config}, Domain {domain}")
        ax.set_xlabel("Chunk ID")
        ax.set_ylabel("Rank")
        fig.tight_layout()
        fig.savefig(
            out_dir / f"2.4_heatmap_{_sanitize(str(domain))}_{_sanitize(config)}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


def main() -> None:
    ensure_style()
    chunks, ranks = prepare_data()
    out_dir = ensure_output_dir(PLOTS_V2_DIR / "2_load_balancing")
    plot_rank_chunk_counts(chunks, out_dir)
    plot_rank_time_breakdown(ranks, out_dir)
    plot_chunk_time_distribution(chunks, out_dir)
    plot_chunk_heatmaps(chunks, out_dir)


if __name__ == "__main__":
    main()
