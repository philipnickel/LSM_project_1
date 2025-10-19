from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from postprocessing.utils import (
    PLOTS_DIR,
    config_palette,
    ensure_config_level,
    ensure_output_dir,
    ensure_style,
    load_suite_ranks,
    load_suite_runs,
)

SUITE = "scaling_proc"


def prepare_data() -> pd.DataFrame:
    runs_idx = ensure_config_level(load_suite_runs(SUITE, as_index=True))
    runs = runs_idx.reset_index()
    runs["N Ranks"] = pd.to_numeric(runs["N Ranks"], errors="coerce")
    runs = runs.dropna(subset=["N Ranks"])
    runs["N Ranks"] = runs["N Ranks"].astype(int)
    runs = runs.sort_values(["Image Size", "N Ranks"])

    if {"Comp Total", "Comm Total"}.issubset(runs.columns):
        totals = runs.rename(columns={"Comp Total": "comp_total", "Comm Total": "comm_total"})
    else:
        ranks_idx = ensure_config_level(load_suite_ranks(SUITE, as_index=True))
        agg = (
            ranks_idx.groupby(level="Run Id")[["comp_time", "comm_send_time", "comm_recv_time"]]
            .sum()
            .rename(columns={"comp_time": "comp_total"})
            .reset_index()
        )
        agg["comm_total"] = agg["comm_send_time"].fillna(0.0) + agg["comm_recv_time"].fillna(0.0)
        totals = runs.merge(agg[["Run Id", "comp_total", "comm_total"]], on="Run Id", how="left")
    totals["Comm Fraction"] = totals["comm_total"] / (totals["comm_total"] + totals["comp_total"])
    totals["N Ranks"] = pd.to_numeric(totals["N Ranks"], errors="coerce")
    totals = totals.dropna(subset=["N Ranks"])
    totals["N Ranks"] = totals["N Ranks"].astype(int)
    totals = totals.sort_values(["Image Size", "N Ranks"])
    return totals


def plot_wall_time(totals: pd.DataFrame, out_dir: Path) -> None:
    palette = config_palette(totals["Config"].unique())
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(
        data=totals,
        x="N Ranks",
        y="Wall Time(s)",
        hue="Config",
        style="Config",
        markers=True,
        dashes=False,
        estimator=None,
        sort=False,
        ax=ax,
        palette=palette,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of MPI ranks")
    ax.set_ylabel("Wall time [s]")
    ax.set_title("Wall Time vs Rank Count")
    fig.tight_layout()
    fig.savefig(out_dir / "5.1_wall_time_vs_ranks.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_style()
    totals = prepare_data()
    out_dir = ensure_output_dir(PLOTS_DIR / "5_scaling_ranks")
    plot_wall_time(totals, out_dir)


if __name__ == "__main__":
    main()
