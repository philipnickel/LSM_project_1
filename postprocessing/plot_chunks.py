# %% chunks suite: wall time vs chunk size -----------------------------------
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401
import seaborn as sns

plt.style.use("science")
sns.set_style("whitegrid")

CACHE_DIR = Path("_exp_mlcache")
RUNS_IDX_PATH = CACHE_DIR / "runs_indexed.parquet"
SUITE = "chunks"

if not RUNS_IDX_PATH.exists():
    raise FileNotFoundError(
        f"{RUNS_IDX_PATH} not found. Run postprocessing/data_loading.py first."
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
    raise ValueError(f"No runs available for suite '{SUITE}'") from exc


# ensure numeric ordering
subset["Chunk Size"] = pd.to_numeric(subset["Chunk Size"], errors="coerce")
subset = subset.sort_values("Chunk Size")
subset["Config"] = (
    subset["Schedule"].astype(str) + " / " + subset["Communication"].astype(str)
)

plots_dir = Path("Plots") / SUITE
plots_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(
    data=subset,
    x="Chunk Size",
    y="Wall Time(s)",
    hue="Config",
    style="Communication",
    markers=True,
    dashes=False,
    ax=ax,
    legend="full"
)
ax.set_title("Wall Time vs Chunk Size")
ax.set_xlabel("Chunk size")
ax.set_ylabel("Wall time [s]")
ax.set_xscale("log")
ax.set_yscale("log")
#ax.legend(title="Scheduler / Communication")
#fig.tight_layout()
out_path = plots_dir / "chunk_size_walltime.pdf"
fig.savefig(out_path, bbox_inches="tight")
plt.close(fig)
print(f"[plots] saved {out_path}")
