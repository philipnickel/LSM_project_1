# %% Imports -----------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401  (registers "science" styles)
import seaborn as sns
from IPython.display import display

# Global styling consistent with report figures
plt.style.use(["science", "grid"])
sns.set_palette("colorblind")
sns.set_context("talk")

# %% Load cached parquet tables ----------------------------------------------
cache_dir = Path("_exp_mlcache")
required_files = {
    "runs": cache_dir / "runs_df.parquet",
    "ranks": cache_dir / "ranks_df.parquet",
    "chunks": cache_dir / "chunks_df.parquet",
}
missing = [name for name, path in required_files.items() if not path.exists()]
if missing:
    raise FileNotFoundError(
        f"Missing cached parquet files for: {', '.join(missing)}. "
        "Run postprocessing/data_loading.py first."
    )

runs_df = pd.read_parquet(required_files["runs"])
ranks_df = pd.read_parquet(required_files["ranks"])
chunks_df = pd.read_parquet(required_files["chunks"])

# Ensure categorical ordering for nicer plotting
for df in (runs_df, ranks_df, chunks_df):
    if "schedule" in df.columns:
        df["schedule"] = pd.Categorical(df["schedule"], ["static", "dynamic"])
    if "communication" in df.columns:
        df["communication"] = pd.Categorical(df["communication"], ["blocking", "nonblocking"])
    if "suite" in df.columns:
        df["suite"] = df["suite"].fillna("default")


def _size_to_pixels(series: pd.Series) -> pd.Series:
    def _convert(val: object) -> float:
        if isinstance(val, str) and "x" in val.lower():
            try:
                w, h = val.lower().split("x")
                return int(w.strip()) * int(h.strip())
            except ValueError:
                return float("nan")
        return float("nan")

    return series.map(_convert)


if "image_size" in runs_df.columns and "image_pixels" not in runs_df.columns:
    runs_df["image_pixels"] = _size_to_pixels(runs_df["image_size"])

# Root directory for suite-specific plots
plots_root = Path("Plots")
plots_root.mkdir(exist_ok=True)


def _pick_facet_column(df: pd.DataFrame, candidates: list[str], max_levels: int = 6) -> str | None:
    for col in candidates:
        if col in df.columns:
            values = df[col].dropna().unique()
            if 1 < len(values) <= max_levels:
                return col
    return None


def analyse_suite(
    label: str,
    runs: pd.DataFrame,
    ranks: pd.DataFrame,
    chunks: pd.DataFrame,
) -> None:
    if runs.empty:
        print(f"[info] skipping suite '{label}' (no runs)")
        return

    suite_dir = plots_root / label
    suite_dir.mkdir(exist_ok=True)

    def save_fig(fig: plt.Figure, name: str) -> None:
        path = suite_dir / name
        fig.savefig(path, bbox_inches="tight")
        print(f"[plots] saved {path}")
        plt.close(fig)

    wall_order = ["static", "dynamic"] if "schedule" in runs.columns else None

    # Wall time summary ------------------------------------------------------
    wall_group_cols = ["schedule", "communication"]
    wall_facet = _pick_facet_column(runs, ["image_size", "n_ranks", "chunk_size", "domain"])
    if wall_facet:
        wall_group_cols.append(wall_facet)

    wall_summary = (
        runs.groupby(wall_group_cols, observed=False)["wall_time"]
        .median()
        .reset_index()
        .sort_values(wall_group_cols)
    )

    cat_kwargs: dict[str, object] = {
        "kind": "bar",
        "x": "schedule",
        "y": "wall_time",
        "hue": "communication",
        "order": wall_order,
        "height": 6,
        "aspect": 1.3,
    }
    if wall_facet:
        cat_kwargs["col"] = wall_facet
        cat_kwargs["col_wrap"] = min(len(wall_summary[wall_facet].unique()), 3)

    g = sns.catplot(data=wall_summary, **cat_kwargs)
    g.set_axis_labels("Scheduler", "Median wall time [s]")
    if g._legend:
        g._legend.set_title("Communication")
    g.fig.suptitle(f"Median Wall Time ({label})", y=1.02)
    save_fig(g.fig, "mpi_wall_time.pdf")

    # Runtime breakdown ------------------------------------------------------
    if {"comp_total", "comm_total"}.issubset(runs.columns):
        phase_group_cols = ["schedule", "communication"]
        if wall_facet and wall_facet in runs.columns:
            phase_group_cols.append(wall_facet)
        phase_summary = (
            runs.groupby(phase_group_cols, observed=False)[["comp_total", "comm_total"]]
            .median()
            .reset_index()
        )
        breakdown = phase_summary.melt(
            id_vars=["schedule", "communication"],
            value_vars=["comp_total", "comm_total"],
            var_name="phase",
            value_name="duration",
        )
        breakdown["phase"] = breakdown["phase"].map(
            {"comp_total": "Compute", "comm_total": "Communication"}
        )
        cat_kwargs = {
            "kind": "bar",
            "x": "schedule",
            "y": "duration",
            "hue": "phase",
            "order": wall_order,
            "height": 6,
            "aspect": 1.3,
        }
        if wall_facet and wall_facet in breakdown.columns:
            cat_kwargs["col"] = wall_facet
            cat_kwargs["col_wrap"] = min(len(breakdown[wall_facet].unique()), 3)

        g = sns.catplot(data=breakdown, **cat_kwargs)
        if g._legend:
            g._legend.set_title("")
        g.set_axis_labels("Scheduler", "Median time [s]")
        g.fig.suptitle(f"Runtime Breakdown ({label})", y=1.02)
        save_fig(g.fig, "mpi_runtime_breakdown.pdf")

    # Rank-level compute/comm -------------------------------------------------
    if not ranks.empty:
        ranks_sorted = ranks.sort_values("rank") if "rank" in ranks.columns else ranks
        g = sns.catplot(
            data=ranks_sorted,
            kind="box",
            x="rank",
            y="comp_time",
            hue="schedule",
            height=6,
            aspect=1.6,
        )
        g.set_axis_labels("Rank", "Compute time [s]")
        g.fig.suptitle(f"Per-Rank Compute Distribution ({label})", y=1.02)
        save_fig(g.fig, "mpi_rank_compute_balance.pdf")

        if {"comm_send_time", "comm_recv_time"}.issubset(ranks.columns):
            ranks_melt = ranks_sorted.melt(
                id_vars=["rank", "schedule", "communication"],
                value_vars=["comm_send_time", "comm_recv_time"],
                var_name="phase",
                value_name="duration",
            )
            ranks_melt["phase"] = ranks_melt["phase"].map(
                {"comm_send_time": "Send", "comm_recv_time": "Receive"}
            )
            g = sns.catplot(
                data=ranks_melt,
                kind="box",
                x="rank",
                y="duration",
                hue="phase",
                height=6,
                aspect=1.6,
            )
            g.set_axis_labels("Rank", "Communication time [s]")
            g.fig.suptitle(f"Per-Rank Communication ({label})", y=1.02)
            save_fig(g.fig, "mpi_rank_comm_balance.pdf")

    # Chunk hotspots ---------------------------------------------------------
    if not chunks.empty:
        hue = "schedule" if "schedule" in chunks.columns else None
        style = "communication" if "communication" in chunks.columns else None
        chunk_facet = _pick_facet_column(chunks, ["rank", "image_size", "n_ranks"], max_levels=4)
        rel_kwargs: dict[str, object] = {
            "kind": "scatter",
            "x": "chunk_id",
            "y": "comp_time",
            "hue": hue,
            "style": style,
            "height": 6,
            "aspect": 1.6,
            "alpha": 0.6,
        }
        if chunk_facet:
            rel_kwargs["col"] = chunk_facet
            rel_kwargs["col_wrap"] = min(len(chunks[chunk_facet].dropna().unique()), 4)

        g = sns.relplot(data=chunks, **rel_kwargs)
        g.set_axis_labels("Chunk ID", "Compute time [s]")
        g.fig.suptitle(f"Chunk Hotspots ({label})", y=1.02)
        save_fig(g.fig, "mpi_chunk_hotspots.pdf")

    # Chunk-size sensitivity --------------------------------------------------
    if "chunk_size" in runs.columns and runs["chunk_size"].notna().any():
        chunk_facet = _pick_facet_column(runs, ["image_size", "n_ranks"], max_levels=6)
        group_cols = [
            col
            for col in ["schedule", "communication", chunk_facet, "chunk_size"]
            if col
        ]
        chunk_summary = (
            runs.dropna(subset=["chunk_size"])
            .groupby(group_cols, observed=False)["wall_time"]
            .median()
            .reset_index()
            .sort_values("chunk_size")
        )
        rel_kwargs = {
            "kind": "line",
            "x": "chunk_size",
            "y": "wall_time",
            "hue": "schedule",
            "style": "communication",
            "markers": True,
            "height": 6,
            "aspect": 1.6,
        }
        if chunk_facet and chunk_facet in chunk_summary.columns:
            rel_kwargs["col"] = chunk_facet
            rel_kwargs["col_wrap"] = min(len(chunk_summary[chunk_facet].unique()), 4)

        g = sns.relplot(data=chunk_summary, **rel_kwargs)
        g.set_axis_labels("Chunk size", "Median wall time [s]")
        g.fig.suptitle(f"Chunk Size Sensitivity ({label})", y=1.02)
        save_fig(g.fig, "mpi_chunk_size_walltime.pdf")

    # Problem size scaling ----------------------------------------------------
    if "image_pixels" in runs.columns and runs["image_pixels"].notna().any():
        pixel_facet = _pick_facet_column(runs, ["n_ranks", "chunk_size"], max_levels=6)
        group_cols = [
            col
            for col in ["schedule", "communication", pixel_facet, "image_pixels"]
            if col
        ]
        pixels_summary = (
            runs.dropna(subset=["image_pixels"])
            .groupby(group_cols, observed=False)["wall_time"]
            .median()
            .reset_index()
            .sort_values("image_pixels")
        )
        rel_kwargs = {
            "kind": "line",
            "x": "image_pixels",
            "y": "wall_time",
            "hue": "schedule",
            "style": "communication",
            "markers": True,
            "height": 6,
            "aspect": 1.6,
        }
        if pixel_facet and pixel_facet in pixels_summary.columns:
            rel_kwargs["col"] = pixel_facet
            rel_kwargs["col_wrap"] = min(len(pixels_summary[pixel_facet].unique()), 4)

        g = sns.relplot(data=pixels_summary, **rel_kwargs)
        g.set_axis_labels("Pixels per image", "Median wall time [s]")
        for ax in g.axes.flat:
            ax.set_xscale("log")
        g.fig.suptitle(f"Problem Size Scaling ({label})", y=1.02)
        save_fig(g.fig, "mpi_problem_size_walltime.pdf")

    # Rank scaling ------------------------------------------------------------
    if "n_ranks" in runs.columns and runs["n_ranks"].notna().any():
        rank_facet = _pick_facet_column(runs, ["image_size", "chunk_size"], max_levels=6)
        group_cols = [
            col
            for col in ["schedule", "communication", rank_facet, "n_ranks"]
            if col
        ]
        ranks_summary = (
            runs.dropna(subset=["n_ranks"])
            .groupby(group_cols, observed=False)["wall_time"]
            .median()
            .reset_index()
            .sort_values("n_ranks")
        )
        rel_kwargs = {
            "kind": "line",
            "x": "n_ranks",
            "y": "wall_time",
            "hue": "schedule",
            "style": "communication",
            "markers": True,
            "height": 6,
            "aspect": 1.6,
        }
        if rank_facet and rank_facet in ranks_summary.columns:
            rel_kwargs["col"] = rank_facet
            rel_kwargs["col_wrap"] = min(len(ranks_summary[rank_facet].unique()), 4)

        g = sns.relplot(data=ranks_summary, **rel_kwargs)
        g.set_axis_labels("MPI ranks", "Median wall time [s]")
        g.fig.suptitle(f"Rank Scaling ({label})", y=1.02)
        save_fig(g.fig, "mpi_rank_scaling.pdf")

    # Summary table for quick inspection -------------------------------------
    summary_cols = [
        col
        for col in ["wall_time", "comp_total", "comm_total", "comm_send_total", "comm_recv_total"]
        if col in runs.columns
    ]
    print(f"Runs loaded ({label}): {len(runs)}")
    if summary_cols:
        summary = (
            runs.groupby(["schedule", "communication"], observed=False)[summary_cols]
            .median()
            .sort_index()
        )
        print("\nMedian timing summary by configuration:")
        display(summary)


# Determine which suites to analyse
if "suite" in runs_df.columns:
    suite_labels = sorted(runs_df["suite"].dropna().unique())
else:
    suite_labels = []

if not suite_labels:
    suite_labels = ["default"]

for suite in suite_labels:
    runs_subset = runs_df[runs_df.get("suite", "default") == suite].copy()
    run_ids = runs_subset.get("run_id", pd.Series(dtype=str))
    ranks_subset = ranks_df[ranks_df.get("run_id", "").isin(run_ids)].copy()
    chunks_subset = chunks_df[chunks_df.get("run_id", "").isin(run_ids)].copy()
    analyse_suite(str(suite), runs_subset, ranks_subset, chunks_subset)
