from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .utils import PLOTS_V2_DIR, ensure_output_dir, ensure_style

CACHE_DIR = Path("_exp_mlcache")
RUNS_IDX = CACHE_DIR / "runs_indexed.parquet"
RANKS_IDX = CACHE_DIR / "ranks_indexed.parquet"
CHUNKS_IDX = CACHE_DIR / "chunks_indexed.parquet"
SUITE = "chunks"


def _load_index(path: Path, levels: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run postprocessing/data_loading.py first."
        )
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError(f"{path.name} must be a MultiIndex table")
    if df.index.names != levels:
        df.index = df.index.set_names(levels)
    return df


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs_idx = _load_index(
        RUNS_IDX,
        [
            "Schedule",
            "Communication",
            "N Ranks",
            "Chunk Size",
            "Domain",
            "Image Size",
            "Run Id",
            "Suite",
        ],
    )
    ranks_idx = _load_index(
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
    chunks_idx = _load_index(
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

    try:
        runs = runs_idx.xs(SUITE, level="Suite").reset_index()
        ranks = ranks_idx.xs(SUITE, level="Suite").reset_index()
        chunks = chunks_idx.xs(SUITE, level="Suite").reset_index()
    except KeyError as exc:
        raise ValueError(f"Suite '{SUITE}' not present in indexed data") from exc

    runs["Chunk Size"] = pd.to_numeric(runs["Chunk Size"], errors="coerce")
    runs = runs.dropna(subset=["Chunk Size"])
    runs = runs.sort_values("Chunk Size")
    runs["Config"] = runs["Schedule"].astype(str) + " / " + runs["Communication"].astype(str)

    ranks["Chunk Size"] = pd.to_numeric(ranks["Chunk Size"], errors="coerce")
    ranks = ranks.dropna(subset=["Chunk Size"])
    ranks["Config"] = ranks["Schedule"].astype(str) + " / " + ranks["Communication"].astype(str)

    chunks["Chunk Size"] = pd.to_numeric(chunks["Chunk Size"], errors="coerce")
    chunks = chunks.dropna(subset=["Chunk Size"])
    chunks["Config"] = chunks["Schedule"].astype(str) + " / " + chunks["Communication"].astype(str)

    return runs, ranks, chunks


def plot_runtime_vs_chunk_size(runs: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(
        data=runs,
        x="Chunk Size",
        y="Wall Time(s)",
        hue="Config",
        style="Domain",
        estimator=None,
        markers=True,
        dashes=False,
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Chunk size")
    ax.set_ylabel("Wall time [s]")
    ax.set_title("Runtime vs Chunk Size")
    fig.tight_layout()
    fig.savefig(out_dir / "3.1_runtime_vs_chunk_size.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_rel_comp_time(ranks: pd.DataFrame, out_dir: Path) -> None:
    comp_totals = (
        ranks.groupby(["Chunk Size", "Config", "Run Id", "rank"], observed=False)["comp_time"]
        .sum()
        .reset_index()
    )
    g = sns.relplot(
        data=comp_totals,
        kind="line",
        x="rank",
        y="comp_time",
        hue="Run Id",
        col="Chunk Size",
        row="Config",
        marker="o",
        height=4,
        aspect=1.1,
        facet_kws={"sharey": False},
    )
    g.set_axis_labels("Rank", "Compute time [s]")
    g.fig.suptitle("Per-Run Rank Compute Time", y=1.02)
    out_path = out_dir / "3.2_rank_compute_relplot.pdf"
    g.fig.savefig(out_path, bbox_inches="tight")
    plt.close(g.fig)


def plot_rank_bars_with_chunks(ranks: pd.DataFrame, chunks: pd.DataFrame, out_dir: Path) -> None:
    comp_totals = (
        ranks.groupby(["Chunk Size", "Config", "Run Id", "rank"], observed=False)["comp_time"]
        .sum()
        .reset_index()
    )
    chunk_counts = (
        chunks.groupby(["Chunk Size", "Config", "Run Id", "rank"], observed=False)
        .size()
        .reset_index(name="chunks_processed")
    )
    merged = comp_totals.merge(
        chunk_counts,
        on=["Chunk Size", "Config", "Run Id", "rank"],
        how="left",
    )

    g = sns.catplot(
        data=merged,
        kind="bar",
        x="rank",
        y="comp_time",
        hue="Run Id",
        col="Chunk Size",
        row="Config",
        height=4,
        aspect=1.1,
        sharey=False,
    )
    grouped = merged.groupby(["Chunk Size", "Config"])
    for ax, ((chunk_size, config), data) in zip(g.axes.flatten(), grouped):
        for container in ax.containers:
            for bar, (_, value) in zip(container, data.iterrows()):
                count = value.get("chunks_processed")
                if pd.notna(count):
                    text = f"{int(count)}"
                else:
                    text = ""
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    text,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    g.set_axis_labels("Rank", "Compute time [s]")
    g.fig.suptitle("Rank Compute Time with Chunk Counts", y=1.02)
    g.fig.subplots_adjust(top=0.88)
    out_path = out_dir / "3.3_rank_time_with_chunks.pdf"
    g.fig.savefig(out_path, bbox_inches="tight")
    plt.close(g.fig)


def main() -> None:
    ensure_style()
    runs, ranks, chunks = load_data()
    out_dir = ensure_output_dir(PLOTS_V2_DIR / "3_chunk_size")
    plot_runtime_vs_chunk_size(runs, out_dir)
    plot_rel_comp_time(ranks, out_dir)
    plot_rank_bars_with_chunks(ranks, chunks, out_dir)


if __name__ == "__main__":
    main()
