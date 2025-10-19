# %% chunks suite: wall time vs chunk size -----------------------------------
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401
import seaborn as sns

plt.style.use(["science", "grid"])
sns.set_palette("colorblind")
sns.set_context("talk")

CACHE_DIR = Path("_exp_mlcache")
RUNS_PATH = CACHE_DIR / "runs_df.parquet"
SUITE = "chunks"

if not RUNS_PATH.exists():
    raise FileNotFoundError(
        f"{RUNS_PATH} not found. Run postprocessing/data_loading.py first."
    )

runs = pd.read_parquet(RUNS_PATH)
if "suite" not in runs.columns:
    raise KeyError("Column 'suite' missing. Re-run data_loading.py.")

subset = runs[runs["suite"] == SUITE].copy()
if subset.empty:
    raise ValueError(f"No runs available for suite '{SUITE}'.")

subset["schedule"] = pd.Categorical(subset["schedule"], ["static", "dynamic"])
subset["communication"] = pd.Categorical(
    subset["communication"], ["blocking", "nonblocking"]
)
subset = subset.sort_values("chunk_size")
subset["config"] = (
    subset["schedule"].astype(str) + " / " + subset["communication"].astype(str)
)

plots_dir = Path("Plots") / SUITE
plots_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(
    data=subset,
    x="chunk_size",
    y="wall_time",
    hue="config",
    marker="o",
    estimator=None,
    ax=ax,
)
ax.set_title("Wall Time vs Chunk Size")
ax.set_xlabel("Chunk size")
ax.set_ylabel("Wall time [s]")
ax.legend(title="Scheduler / Communication")
fig.tight_layout()
out_path = plots_dir / "chunk_size_walltime.pdf"
fig.savefig(out_path, bbox_inches="tight")
plt.close(fig)
print(f"[plots] saved {out_path}")
