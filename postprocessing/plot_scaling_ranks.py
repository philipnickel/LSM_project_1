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
    config_label,
    config_palette,
    ensure_output_dir,
    ensure_style,
    load_suite_ranks,
    load_suite_runs,
)

SUITE = "scaling_proc"


def prepare_data() -> pd.DataFrame:
    runs = load_suite_runs(SUITE)
    runs["Config"] = [
        config_label(s, c) for s, c in zip(runs["Schedule"], runs["Communication"])
    ]
    runs = runs.sort_values(["Image Size", "N Ranks"])

    if {"Comp Total", "Comm Total"}.issubset(runs.columns):
        totals = runs.rename(columns={"Comp Total": "comp_total", "Comm Total": "comm_total"})
    else:
        ranks = load_suite_ranks(SUITE)
        ranks["config"] = [
            config_label(s, c) for s, c in zip(ranks["Schedule"], ranks["Communication"])
        ]
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
    totals["Comm Fraction"] = totals["comm_total"] / (totals["comm_total"] + totals["comp_total"])
    totals["Config"] = [
        config_label(s, c) for s, c in zip(totals["Schedule"], totals["Communication"])
    ]
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
        style="Image Size",
        markers=True,
        estimator=None,
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


def plot_comm_breakdown(totals: pd.DataFrame, out_dir: Path) -> None:
    melted = totals.melt(
        id_vars=["N Ranks", "Config", "Image Size"],
        value_vars=["comp_total", "comm_total"],
        var_name="Metric",
        value_name="Time",
    )
    g = sns.catplot(
        data=melted,
        kind="bar",
        x="N Ranks",
        y="Time",
        hue="Metric",
        col="Config",
        row="Image Size",
        sharey=False,
        height=4,
        aspect=1.2,
    )
    g.set_axis_labels("Number of MPI ranks", "Time [s]")
    g.fig.suptitle("Computation vs Communication by Rank Count", y=1.02)
    g.fig.savefig(out_dir / "5.2_comm_breakdown_vs_ranks.pdf", bbox_inches="tight")
    plt.close(g.fig)


def main() -> None:
    ensure_style()
    totals = prepare_data()
    out_dir = ensure_output_dir(PLOTS_DIR / "5_scaling_ranks")
    plot_wall_time(totals, out_dir)
    plot_comm_breakdown(totals, out_dir)


if __name__ == "__main__":
    main()
