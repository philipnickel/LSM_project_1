# %% Imports
from __future__ import annotations
from pathlib import Path
import pandas as pd
from IPython.display import display, HTML


# %% Data loading
runs_df = pd.read_parquet("_exp_mlcache/runs_df.parquet")
runs_idx = pd.read_parquet("_exp_mlcache/runs_indexed.parquet")
chunks_df = pd.read_parquet("_exp_mlcache/chunks_df.parquet")
chunks_idx = pd.read_parquet("_exp_mlcache/chunks_indexed.parquet")


# %% Display tables
print(f"\n=== runs_df ({len(runs_df)} rows) ===")
runs_df.head()


# %% Display tables
print(f"\n=== runs_indexed ({len(runs_idx)} rows) ===")
runs_idx.head()

# %% Display tables
print(f"\n=== chunks_df ({len(chunks_df)} rows) ===")
chunks_df.head()

# %% Display tables
print(f"\n=== chunks_indexed ({len(chunks_idx)} entries) ===")
chunks_idx.head()
