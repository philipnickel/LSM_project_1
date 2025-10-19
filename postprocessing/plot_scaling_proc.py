# %% scaling_proc suite: rank scaling ----------------------------------------
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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
SUITE = "scaling_proc"

for path in (RUNS_PATH, RANKS_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run postprocessing/data_loading.py first."
        )

runs = pd.read_parquet(RUNS_PATH)
if "suite" not in runs.columns:
    raise KeyError("Column 'suite' missing. Re-run data_loading.py.")

subset = runs[runs["suite"] == SUITE].copy()
if subset.empty:
    raise ValueError(f"No runs found for suite '{SUITE}'.")

subset["schedule"] = pd.Categorical(subset["schedule"], ["static", "dynamic"])
subset["communication"] = pd.Categorical(
    subset["communication"], ["blocking", "nonblocking"]
)
subset["config"] = subset["schedule"].astype(str) + " / " + subset["communication"].astype(str)

if {"comp_total", "comm_total"}.issubset(subset.columns):
    totals = subset.copy()
else:
    ranks = pd.read_parquet(RANKS_PATH)
    ranks_subset = ranks[ranks.get("suite", "") == SUITE].copy()
    if ranks_subset.empty:
        raise ValueError("Rank-level data required to derive totals.")
    ranks_subset["comm_total"] = (
        ranks_subset.get("comm_send_time", 0.0).fillna(0.0)
        + ranks_subset.get("comm_recv_time", 0.0).fillna(0.0)
    )
    totals = subset.merge(
        ranks_subset.groupby("run_id", observed=False)[["comp_time", "comm_total"]]
        .sum()
        .reset_index()
        .rename(columns={"comp_time": "comp_total"}),
        on="run_id",
        how="left",
    )

plots_dir = Path("Plots") / SUITE
plots_dir.mkdir(parents=True, exist_ok=True)

# Wall time vs ranks ---------------------------------------------------------
wall_df = totals.dropna(subset=["n_ranks"]).sort_values(["image_size", "n_ranks"])

fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(
    data=wall_df,
    x="n_ranks",
    y="wall_time",
    hue="config",
    style="image_size",
    markers=True,
    estimator=None,
    ax=ax,
)
ax.set_xlabel("MPI ranks")
ax.set_ylabel("Median wall time [s]")
ax.set_title("Wall Time vs Rank Count")
ax.legend(title="Config / Image size", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
out_wall = plots_dir / "rank_scaling_walltime.pdf"
fig.savefig(out_wall, bbox_inches="tight")
plt.close(fig)
print(f"[plots] saved {out_wall}")

# Communication ratio vs ranks ----------------------------------------------
ratio_df = totals.copy()
ratio_df["comm_ratio"] = ratio_df["comm_total"] / ratio_df["comp_total"].clip(lower=1e-12)
ratio_df = ratio_df.dropna(subset=["n_ranks"]).sort_values(["image_size", "n_ranks"])

fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(
    data=ratio_df,
    x="n_ranks",
    y="comm_ratio",
    hue="config",
    style="image_size",
    markers=True,
    estimator=None,
    ax=ax,
)
ax.set_xlabel("MPI ranks")
ax.set_ylabel("Communication / compute ratio")
ax.set_title("Communication Share vs Rank Count")
ax.legend(title="Config / Image size", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
out_ratio = plots_dir / "rank_scaling_comm_ratio.pdf"
fig.savefig(out_ratio, bbox_inches="tight")
plt.close(fig)
print(f"[plots] saved {out_ratio}")
