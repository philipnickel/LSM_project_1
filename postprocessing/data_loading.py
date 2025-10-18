# %% ── Imports & display ──────────────────────────────────────────────────────
from __future__ import annotations

import ast
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import mlflow
import pandas as pd

pd.set_option("display.float_format", lambda x: f"{x:.2e}")

# %% ── Config ────────────────────────────────────────────────────────────────
ARTIFACT_NAME = "chunks.json"
RANKS_ARTIFACT_NAME = "ranks.json"
OUT_DIR = "_exp_mlcache"
os.makedirs(OUT_DIR, exist_ok=True)

# %% ── Login & load runs ─────────────────────────────────────────────────────
mlflow.login(backend="databricks")

runs_df = mlflow.search_runs(search_all_experiments=True).copy()

# Drop tags.*
runs_df = runs_df.drop(runs_df.filter(regex=r"^tags\.").columns, axis=1)

# Strip params./metrics.
for p in ("params.", "metrics."):
    runs_df.columns = runs_df.columns.str.replace(f"^{p}", "", regex=True)

print(f"[info] Loaded {len(runs_df)} runs with {len(runs_df.columns)} columns")

# Prefer new wall_time metric; fall back to older total_time if needed
if "wall_time" not in runs_df.columns and "total_time" in runs_df.columns:
    runs_df["wall_time"] = runs_df["total_time"]

for legacy_col in ["total_time", "total_comp_time", "total_comm_time", "comp_std", "comm_std"]:
    if legacy_col in runs_df.columns:
        runs_df.drop(columns=[legacy_col], inplace=True)

# %% ── Domain & image size cleanup ───────────────────────────────────────────
for col in ["xlim", "ylim"]:
    if col in runs_df.columns:
        runs_df[col] = runs_df[col].map(
            lambda v: tuple(ast.literal_eval(v)) if isinstance(v, str) else v
        )


def fmt_domain(row):
    x, y = row.get("xlim"), row.get("ylim")
    if isinstance(x, (tuple, list)) and isinstance(y, (tuple, list)):
        return f"[{x[0]:.2f},{x[1]:.2f}]×[{y[0]:.2f},{y[1]:.2f}]"
    return None


if {"xlim", "ylim"}.issubset(runs_df.columns):
    runs_df["domain"] = runs_df.apply(fmt_domain, axis=1)
    runs_df.drop(columns=["xlim", "ylim"], inplace=True, errors="ignore")

# Create image size column: "widthxheight"
if {"width", "height"}.issubset(runs_df.columns):
    runs_df["image_size"] = (
        runs_df["width"].fillna(0).astype(int).astype(str)
        + "x"
        + runs_df["height"].fillna(0).astype(int).astype(str)
    )
    runs_df.drop(columns=["width", "height"], inplace=True, errors="ignore")

# %% ── Filter columns & MultiIndex ───────────────────────────────────────────
keep = [
    "image_size",
    "n_ranks",
    "communication",
    "schedule",
    "chunk_size",
    "wall_time",
    "domain",
    "run_id",
]
runs_df = runs_df[[c for c in keep if c in runs_df.columns]].copy()
runs_df.dropna(subset=["schedule"], inplace=True)

levels = [
    c
    for c in [
        "schedule",
        "communication",
        "n_ranks",
        "chunk_size",
        "domain",
        "image_size",
        "run_id",
    ]
    if c in runs_df.columns
]
runs_idx = runs_df.set_index(levels).sort_index()
print("[info] MultiIndex levels:", runs_idx.index.names)

# %% ── Load chunk artifacts (fast via load_dict + threads) ───────────────────

print("[info] Loading chunk artifacts...")
meta_cols = [
    c
    for c in [
        "schedule",
        "communication",
        "chunk_size",
        "n_ranks",
        "domain",
        "image_size",
        "run_id",
    ]
    if c in runs_df.columns
]


def _load_chunk_one(row_tuple) -> pd.DataFrame | tuple[str, str]:
    run_id = row_tuple.run_id
    uri = f"runs:/{run_id}/{ARTIFACT_NAME}"
    try:
        data = mlflow.artifacts.load_dict(uri)  # fastest & cleanest for JSON
        # Accept {"columns","data"} or list-of-dicts
        if isinstance(data, dict) and "columns" in data and "data" in data:
            df = pd.DataFrame(data["data"], columns=data["columns"])
        else:
            df = pd.DataFrame(data)
        # attach metadata
        for c in meta_cols:
            if hasattr(row_tuple, c):
                df[c] = getattr(row_tuple, c)
        return df
    except Exception as e:
        return (run_id, str(e))


dfs, missing = [], []
max_workers = 8
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    futs = {ex.submit(_load_chunk_one, row): row.run_id for row in runs_df.itertuples(index=False)}
    for fut in as_completed(futs):
        res = fut.result()
        if isinstance(res, pd.DataFrame):
            dfs.append(res)
        else:
            missing.append(res)

chunks_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
print(f"[info] chunks: {len(chunks_df)} rows from {len(dfs)} runs | missing: {len(missing)}")
if missing[:5]:
    print("[warn] missing examples:", missing[:5])

if not chunks_df.empty:
    expected = ["rank", "chunk_id", "comp_time"]
    missing_cols = [c for c in expected if c not in chunks_df.columns]
    if missing_cols:
        print("[warn] missing columns in chunks_df:", missing_cols)

    for col in ["n_ranks", "chunk_size", "rank", "chunk_id"]:
        if col in chunks_df.columns:
            chunks_df[col] = pd.to_numeric(chunks_df[col], errors="coerce").astype("Int64")

    numeric_cols = [
        col
        for col in ["comp_time", "comm_time", "start_time", "end_time"]
        if col in chunks_df.columns
    ]
    for col in numeric_cols:
        if col in chunks_df.columns:
            chunks_df[col] = pd.to_numeric(chunks_df[col], errors="coerce")

    for col, categories in {
        "schedule": ["static", "dynamic"],
        "communication": ["blocking", "nonblocking"],
    }.items():
        if col in chunks_df.columns:
            chunks_df[col] = pd.Categorical(chunks_df[col], categories=categories, ordered=True)

    chunk_levels = [
        c
        for c in [
            "schedule",
            "communication",
            "n_ranks",
            "chunk_size",
            "domain",
            "image_size",
            "run_id",
            "rank",
            "chunk_id",
        ]
        if c in chunks_df.columns
    ]

    chunks_idx = chunks_df.set_index(chunk_levels).sort_index()
    print("[info] chunks index levels:", chunks_idx.index.names)
else:
    chunks_idx = chunks_df

# %% ── Load rank artifacts ───────────────────────────────────────────────────

print("[info] Loading rank artifacts...")


def _load_rank_one(row_tuple) -> pd.DataFrame | tuple[str, str]:
    run_id = row_tuple.run_id
    uri = f"runs:/{run_id}/{RANKS_ARTIFACT_NAME}"
    try:
        data = mlflow.artifacts.load_dict(uri)
        if isinstance(data, dict) and "columns" in data and "data" in data:
            df = pd.DataFrame(data["data"], columns=data["columns"])
        else:
            df = pd.DataFrame(data)
        for c in meta_cols:
            if hasattr(row_tuple, c):
                df[c] = getattr(row_tuple, c)
        return df
    except Exception as e:
        return (run_id, str(e))


rank_dfs: list[pd.DataFrame] = []
rank_missing: list[tuple[str, str]] = []
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    futures = {
        ex.submit(_load_rank_one, row): row.run_id for row in runs_df.itertuples(index=False)
    }
    for fut in as_completed(futures):
        result = fut.result()
        if isinstance(result, pd.DataFrame):
            rank_dfs.append(result)
        else:
            rank_missing.append(result)

ranks_df = pd.concat(rank_dfs, ignore_index=True) if rank_dfs else pd.DataFrame()
print(
    f"[info] ranks: {len(ranks_df)} rows from {len(rank_dfs)} runs | missing: {len(rank_missing)}"
)
if rank_missing[:5]:
    print("[warn] missing rank artifacts examples:", rank_missing[:5])

if not ranks_df.empty:
    expected = ["rank", "comp_time", "comm_time", "chunks"]
    missing_cols = [c for c in expected if c not in ranks_df.columns]
    if missing_cols:
        print("[warn] missing columns in ranks_df:", missing_cols)

    for col in ["n_ranks", "chunk_size", "rank", "chunks"]:
        if col in ranks_df.columns:
            ranks_df[col] = pd.to_numeric(ranks_df[col], errors="coerce").astype("Int64")

    for col in ["comp_time", "comm_time", "comm_send_time", "comm_recv_time"]:
        if col in ranks_df.columns:
            ranks_df[col] = pd.to_numeric(ranks_df[col], errors="coerce")

    for col, categories in {
        "schedule": ["static", "dynamic"],
        "communication": ["blocking", "nonblocking"],
    }.items():
        if col in ranks_df.columns:
            ranks_df[col] = pd.Categorical(ranks_df[col], categories=categories, ordered=True)

    rank_levels = [
        c
        for c in [
            "schedule",
            "communication",
            "n_ranks",
            "chunk_size",
            "domain",
            "image_size",
            "run_id",
            "rank",
        ]
        if c in ranks_df.columns
    ]

    ranks_idx = ranks_df.set_index(rank_levels).sort_index()
    print("[info] ranks index levels:", ranks_idx.index.names)
else:
    ranks_idx = ranks_df

# %% ── Save both tables ──────────────────────────────────────────────────────
runs_df.to_parquet(os.path.join(OUT_DIR, "runs_df.parquet"), index=False)
runs_idx.to_parquet(os.path.join(OUT_DIR, "runs_indexed.parquet"))
chunks_df.to_parquet(os.path.join(OUT_DIR, "chunks_df.parquet"), index=False)
chunks_idx.to_parquet(os.path.join(OUT_DIR, "chunks_indexed.parquet"))
ranks_df.to_parquet(os.path.join(OUT_DIR, "ranks_df.parquet"), index=False)
ranks_idx.to_parquet(os.path.join(OUT_DIR, "ranks_indexed.parquet"))
print("[info] saved parquet files in", OUT_DIR)
