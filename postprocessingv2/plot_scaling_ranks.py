from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .utils import PLOTS_V2_DIR, ensure_output_dir, ensure_style

CACHE_DIR = Path("_exp_mlcache")
RUNS_IDX = CACHE_DIR / "runs_indexed.parquet"
RANKS_IDX = CACHE_DIR / "ranks_indexed.parquet"
SUITE = "scaling_proc"


def load_runs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not RUNS_IDX.exists() or not RANKS_IDX.exists():
        raise FileNotFoundError(
            "Indexed parquet files missing. Run postprocessing/data_loading.py first."
        )

    runs_idx = pd.read_parquet(RUNS_IDX)
    ranks_idx = pd.read_parquet(RANKS_IDX)
    if not isinstance(runs_idx.index, pd.MultiIndex) or not isinstance(
        ranks_idx.index, pd.MultiIndex
    ):
        raise ValueError("Indexed parquet files must contain MultiIndex tables")

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
    if ranks_idx.index.names[-1] != "Suite":
        ranks_idx.index = ranks_idx.index.set_names(
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
            ]
        )

    try:
        runs = runs_idx.xs(SUITE, level="Suite").reset_index()
        ranks = ranks_idx.xs(SUITE, level="Suite").reset_index()
    except KeyError as exc:
        raise ValueError(f"Suite '{SUITE}' not present in indexed data") from exc

    runs = runs.sort_values(["Image Size", "N Ranks"])
    runs["Config"] = (
        runs["Schedule"].astype(str) + " / " + runs["Communication"].astype(str)
    )

    if {"comp_total", "comm_total"}.issubset(runs.columns):
        totals = runs.copy()
    else:
        ranks["comm_total"] = (
            ranks.get("comm_send_time", 0.0).fillna(0.0)
            + ranks.get("comm_recv_time", 0.0).fillna(0.0)
        )
        totals = runs.merge(
            ranks.groupby("Run Id", observed=False)[["comp_time", "comm_total"]]
            .sum()
            .reset_index()
            .rename(columns={"comp_time": "comp_total"}),
            on="Run Id",
            how="left",
        )
    totals = totals.sort_values(["Image Size", "N Ranks"])
    totals["Comm Ratio"] = totals["comm_total"] / totals["comp_total"].clip(lower=1e-12)
    return runs, totals


def plot_wall_time(runs: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(
        data=runs,
        x="N Ranks",
        y="Wall Time(s)",
        hue="Config",
        style="Image Size",
        markers=True,
        estimator=None,
        ax=ax,
    )
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Wall time [s]")
    ax.set_title("Wall Time vs Rank Count")
    fig.tight_layout()
    fig.savefig(out_dir / "5.1_wall_time_vs_ranks.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_comm_ratio(totals: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(
        data=totals,
        x="N Ranks",
        y="Comm Ratio",
        hue="Config",
        style="Image Size",
        markers=True,
        estimator=None,
        ax=ax,
    )
    ax.set_xlabel("MPI ranks")
    ax.set_ylabel("Communication / compute ratio")
    ax.set_title("Communication Share vs Rank Count")
    fig.tight_layout()
    fig.savefig(out_dir / "5.2_comm_ratio_vs_ranks.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_style()
    runs, totals = load_runs()
    out_dir = ensure_output_dir(PLOTS_V2_DIR / "5_scaling_ranks")
    plot_wall_time(runs, out_dir)
    plot_comm_ratio(totals, out_dir)


if __name__ == "__main__":
    main()
