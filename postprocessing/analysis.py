# %% Imports -----------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401 - registers the "science" styles
import seaborn as sns
from IPython.display import display

# Configure styling to match the rest of the project
sns.set_style()
plt.style.use(["science"])

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

# %%

# Ensure categorical ordering for nicer plotting
for df in (runs_df, ranks_df, chunks_df):
    if "schedule" in df.columns:
        df["schedule"] = pd.Categorical(df["schedule"], ["static", "dynamic"])
    if "communication" in df.columns:
        df["communication"] = pd.Categorical(df["communication"], ["blocking", "nonblocking"])

# Helper to extract numeric information from image_size field
if "image_size" in runs_df.columns and "image_pixels" not in runs_df.columns:
    def _size_to_pixels(val: object) -> float:
        if isinstance(val, str) and "x" in val.lower():
            try:
                w, h = val.lower().split("x")
                return int(w.strip()) * int(h.strip())
            except ValueError:
                return float("nan")
        return float("nan")

    runs_df["image_pixels"] = runs_df["image_size"].map(_size_to_pixels)

# Directory to persist plots
plots_dir = Path("Plots")
plots_dir.mkdir(exist_ok=True)

# %%
def save_fig(fig: plt.Figure, filename: str) -> None:
    out_path = plots_dir / filename
    fig.savefig(out_path, bbox_inches="tight")
    print(f"[plots] saved {out_path}")
    plt.close(fig)

# %% Wall-time comparison -----------------------------------------------------
# Provide stable ordering where available
wall_order = ["static", "dynamic"] if "schedule" in runs_df.columns else None
wall_summary = (
    runs_df.groupby(["schedule", "communication"], observed=False)["wall_time"]
    .median()
    .reset_index()
)
wall_summary = wall_summary.sort_values(["schedule", "communication"])
g = sns.catplot(
    data=wall_summary,
    kind="bar",
    x="schedule",
    y="wall_time",
    hue="communication",
    order=wall_order,
    height=6,
    aspect=1.3,
)
g.set_axis_labels("Scheduler", "Median wall time [s]")
g._legend.set_title("Communication")
g.fig.suptitle("Median Wall Time by Scheduler/Communication", y=1.02)
save_fig(g.fig, "mpi_wall_time.pdf")

# %% Compute vs communication breakdown ---------------------------------------
if {"comp_total", "comm_total"}.issubset(runs_df.columns):
    phase_summary = (
        runs_df.groupby(["schedule", "communication"], observed=False)[
            ["comp_total", "comm_total"]
        ]
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
    g = sns.catplot(
        data=breakdown,
        kind="bar",
        x="schedule",
        y="duration",
        hue="phase",
        order=wall_order,
        height=6,
        aspect=1.3,
    )
    g.set_axis_labels("Scheduler", "Median time [s]")
    g._legend.set_title("")
    g.fig.suptitle("Runtime Breakdown (Median per Phase)", y=1.02)
    save_fig(g.fig, "mpi_runtime_breakdown.pdf")

# %% Load balancing across ranks ---------------------------------------------
if not ranks_df.empty:
    # Sort rank categories numerically for nicer plotting
    if "rank" in ranks_df.columns and pd.api.types.is_numeric_dtype(ranks_df["rank"]):
        ranks_df = ranks_df.sort_values("rank")

    g = sns.catplot(
        data=ranks_df,
        kind="box",
        x="rank",
        y="comp_time",
        hue="schedule",
        height=6,
        aspect=1.6,
    )
    g.set_axis_labels("Rank", "Compute time [s]")
    g.fig.suptitle("Per-Rank Compute Time Distribution", y=1.02)
    save_fig(g.fig, "mpi_rank_compute_balance.pdf")

    if {"comm_send_time", "comm_recv_time"}.issubset(ranks_df.columns):
        ranks_melt = ranks_df.melt(
            id_vars=["rank", "schedule", "communication"],
            value_vars=["comm_send_time", "comm_recv_time"],
            var_name="phase",
            value_name="duration",
        )
        phase_map = {
            "comm_send_time": "Send",
            "comm_recv_time": "Receive",
        }
        ranks_melt["phase"] = ranks_melt["phase"].map(phase_map)

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
        g.fig.suptitle("Per-Rank Communication Time", y=1.02)
        save_fig(g.fig, "mpi_rank_comm_balance.pdf")

# %% Chunk hotspots -----------------------------------------------------------
if not chunks_df.empty:
    hue = "schedule" if "schedule" in chunks_df.columns else None
    style = "communication" if "communication" in chunks_df.columns else None
    g = sns.relplot(
        data=chunks_df,
        kind="scatter",
        x="chunk_id",
        y="comp_time",
        hue=hue,
        style=style,
        height=6,
        aspect=1.6,
        alpha=0.6,
    )
    g.set_axis_labels("Chunk ID", "Compute time [s]")
    g.fig.suptitle("Chunk Compute Time Hotspots", y=1.02)
    save_fig(g.fig, "mpi_chunk_hotspots.pdf")

# %% Chunk-size sensitivity ---------------------------------------------------
if "chunk_size" in runs_df.columns:
    chunk_summary = (
        runs_df.dropna(subset=["chunk_size"])
        .groupby(["schedule", "communication", "chunk_size"], observed=False)["wall_time"]
        .median()
        .reset_index()
        .sort_values("chunk_size")
    )
    g = sns.relplot(
        data=chunk_summary,
        kind="line",
        x="chunk_size",
        y="wall_time",
        hue="schedule",
        style="communication",
        markers=True,
        height=6,
        aspect=1.6,
    )
    g.set_axis_labels("Chunk size", "Median wall time [s]")
    g.fig.suptitle("Wall Time vs. Chunk Size", y=1.02)
    save_fig(g.fig, "mpi_chunk_size_walltime.pdf")

# %% Problem size scaling -----------------------------------------------------
if "image_pixels" in runs_df.columns and runs_df["image_pixels"].notna().any():
    pixels_summary = (
        runs_df.dropna(subset=["image_pixels"])
        .groupby(["schedule", "communication", "image_pixels"], observed=False)["wall_time"]
        .median()
        .reset_index()
        .sort_values("image_pixels")
    )
    g = sns.relplot(
        data=pixels_summary,
        kind="line",
        x="image_pixels",
        y="wall_time",
        hue="schedule",
        style="communication",
        markers=True,
        height=6,
        aspect=1.6,
    )
    g.set_axis_labels("Pixels per image", "Median wall time [s]")
    g.ax.set_xscale("log")
    g.fig.suptitle("Wall Time vs. Problem Size", y=1.02)
    save_fig(g.fig, "mpi_problem_size_walltime.pdf")

# %% Scaling vs number of ranks ----------------------------------------------
if "n_ranks" in runs_df.columns:
    ranks_summary = (
        runs_df.dropna(subset=["n_ranks"])
        .groupby(["schedule", "communication", "n_ranks"], observed=False)["wall_time"]
        .median()
        .reset_index()
        .sort_values("n_ranks")
    )
    g = sns.relplot(
        data=ranks_summary,
        kind="line",
        x="n_ranks",
        y="wall_time",
        hue="schedule",
        style="communication",
        markers=True,
        height=6,
        aspect=1.6,
    )
    g.set_axis_labels("Number of MPI ranks", "Median wall time [s]")
    g.fig.suptitle("Wall Time vs. Rank Count", y=1.02)
    save_fig(g.fig, "mpi_rank_scaling.pdf")

# %% Optional: display summaries ----------------------------------------------
print(f"Runs loaded: {len(runs_df)}")
summary_cols = [
    c
    for c in ["wall_time", "comp_total", "comm_total", "comm_send_total", "comm_recv_total"]
    if c in runs_df.columns
]
if summary_cols:
    summary = (
        runs_df.groupby(["schedule", "communication"], observed=False)[summary_cols]
        .median()
        .sort_index()
    )
    print("\nMedian timing summary by configuration:")
    display(summary)
