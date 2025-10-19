# %% scaling_im suite: problem-size scaling ----------------------------------
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
RUNS_IDX_PATH = CACHE_DIR / "runs_indexed.parquet"
RANKS_IDX_PATH = CACHE_DIR / "ranks_indexed.parquet"
SUITE = "scaling_im"

for path in (RUNS_IDX_PATH, RANKS_IDX_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run postprocessing/data_loading.py first."
        )

runs_idx = pd.read_parquet(RUNS_IDX_PATH)
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

try:
    subset = runs_idx.xs(SUITE, level="Suite").reset_index()
except KeyError as exc:
    raise ValueError(f"No runs found for suite '{SUITE}'") from exc

subset["config"] = subset["Schedule"].astype(str) + " / " + subset["Communication"].astype(str)

if "image_pixels" not in subset.columns:
    def _size_to_pixels(value: object) -> float:
        if isinstance(value, str) and "x" in value.lower():
            try:
                w, h = value.lower().split("x")
                return int(w.strip()) * int(h.strip())
            except ValueError:
                return float("nan")
        return float("nan")

    subset["image_pixels"] = subset["Image Size"].map(_size_to_pixels)

ranks_idx = pd.read_parquet(RANKS_IDX_PATH)
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

if {"comp_total", "comm_total"}.issubset(subset.columns):
    totals = subset.copy()
else:
    ranks_subset = ranks_idx.xs(SUITE, level="Suite").reset_index()
    if ranks_subset.empty:
        raise ValueError("Rank-level data required to derive totals.")
    ranks_subset["comm_total"] = (
        ranks_subset.get("comm_send_time", 0.0).fillna(0.0)
        + ranks_subset.get("comm_recv_time", 0.0).fillna(0.0)
    )
    totals = subset.merge(
        ranks_subset.groupby("Run Id", observed=False)[["comp_time", "comm_total"]]
        .sum()
        .reset_index()
        .rename(columns={"comp_time": "comp_total"}),
        on="Run Id",
        how="left",
    )

# Sort for readability
totals = totals.sort_values("image_pixels")

plots_dir = Path("Plots") / SUITE
plots_dir.mkdir(parents=True, exist_ok=True)

# Log-log wall-time scaling --------------------------------------------------
wall_df = totals.dropna(subset=["image_pixels"]).sort_values("image_pixels")

fig, ax = plt.subplots(figsize=(12, 8))
for config, config_df in wall_df.groupby("config"):
    ax.plot(
        config_df["image_pixels"],
        config_df["Wall Time(s)"] if "Wall Time(s)" in config_df else config_df["wall_time"],
        marker="o",
        label=config,
    )
    for _, row in config_df.iterrows():
        ax.annotate(
            row.get("Image Size", row.get("image_size")),
            (
                row["image_pixels"],
                row.get("Wall Time(s)", row.get("wall_time")),
            ),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
        )
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Pixels per image")
ax.set_ylabel("Wall time [s]")
ax.set_title("Wall Time vs Problem Size")
ax.legend(title="Scheduler / Communication")
fig.tight_layout()
out_wall = plots_dir / "problem_size_walltime.pdf"
fig.savefig(out_wall, bbox_inches="tight")
plt.close(fig)
print(f"[plots] saved {out_wall}")

# Compute vs communication totals -------------------------------------------
bar_df = totals[["config", "Image Size", "Run Id", "comp_total", "comm_total"]].copy()
bar_df["label"] = bar_df.apply(lambda r: f"{r['Image Size']}\n{r['config']}", axis=1)

fig, ax = plt.subplots(figsize=(12, 8))
positions = np.arange(len(bar_df))
width = 0.6
ax.bar(positions, bar_df["comp_total"], width=width, label="Compute")
ax.bar(
    positions,
    bar_df["comm_total"],
    width=width,
    bottom=bar_df["comp_total"],
    label="Communication",
    alpha=0.5,
)
ax.set_xticks(positions)
ax.set_xticklabels(bar_df["label"], rotation=45, ha="right")
ax.set_xlabel("Image size / configuration")
ax.set_ylabel("Total time [s]")
ax.set_title("Compute vs Communication Totals")
ax.legend()
fig.tight_layout()
out_phase = plots_dir / "problem_size_phase_breakdown.pdf"
fig.savefig(out_phase, bbox_inches="tight")
plt.close(fig)
print(f"[plots] saved {out_phase}")
