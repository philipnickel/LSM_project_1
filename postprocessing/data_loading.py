from __future__ import annotations

import ast
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import mlflow
from mlflow.exceptions import MlflowException
import pandas as pd

pd.set_option("display.float_format", lambda x: f"{x:.2e}")

OUT_DIR = "_exp_mlcache"
ARTIFACTS = {"chunks": "chunks.json", "ranks": "ranks.json"}
MAX_WORKERS = 8
os.makedirs(OUT_DIR, exist_ok=True)

REQ = [
    "run_id",
    "metrics.wall_time",
    "metrics.comp_total",
    "metrics.comm_total",
    "metrics.comm_send_total",
    "metrics.comm_recv_total",
    "params.schedule",
    "params.communication",
    "params.n_ranks",
    "params.chunk_size",
    "params.width",
    "params.height",
    "params.xlim",
    "params.ylim",
    "tags.suite",
]


def strip(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns
    cols = cols.str.replace(r"^params\.", "", regex=True)
    cols = cols.str.replace(r"^metrics\.", "", regex=True)
    return df.set_axis(cols, axis=1)


def _parse_bounds(value: object, fallback: tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return fallback
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return fallback
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return fallback
    return fallback


def mk_domain(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["xlim"] = df["xlim"].map(lambda v: _parse_bounds(v, (-2.2, 0.75)))
    df["ylim"] = df["ylim"].map(lambda v: _parse_bounds(v, (-1.3, 1.3)))
    df["Domain"] = df.apply(
        lambda r: (
            f"[{r['xlim'][0]:.2f},{r['xlim'][1]:.2f}]×[{r['ylim'][0]:.2f},{r['ylim'][1]:.2f}]"
        ),
        axis=1,
    )
    return df.drop(columns=["xlim", "ylim"])


def mk_image_size(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    width = df["width"].fillna(0).astype(int).astype(str)
    height = df["height"].fillna(0).astype(int).astype(str)
    df["Image Size"] = width + "x" + height
    return df.drop(columns=["width", "height"])


def build_index(df: pd.DataFrame, levels: list[str]) -> pd.DataFrame:
    return df.set_index(levels).sort_index()


def load_table_for_all_runs(
    runs_df: pd.DataFrame,
    artifact_name: str,
    meta_cols: list[str],
) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []

    def _one(row: pd.Series) -> pd.DataFrame:
        run_id = row.get("Run Id") or row.get("run_id")
        if run_id is None:
            raise KeyError("Run Id column missing from runs table")
        uri = f"runs:/{run_id}/{artifact_name}"
        try:
            data = mlflow.artifacts.load_dict(uri)
        except (FileNotFoundError, MlflowException):
            print(f"[warn] Missing artifact '{artifact_name}' for run {run_id}, skipping.")
            return pd.DataFrame()
        if isinstance(data, dict) and "columns" in data and "data" in data:
            tbl = pd.DataFrame(data["data"], columns=data["columns"])
        else:
            tbl = pd.DataFrame(data)
        for col in meta_cols:
            tbl[col] = row.get(col)
        return tbl

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(_one, r) for _, r in runs_df.iterrows()]
        for fut in as_completed(futures):
            result = fut.result()
            if not result.empty:
                dfs.append(result)
    if not dfs:
        raise RuntimeError(f"No data retrieved for artifact '{artifact_name}'")
    return pd.concat(dfs, ignore_index=True)


mlflow.login(backend="databricks")
raw = mlflow.search_runs(search_all_experiments=True).copy()

missing = [c for c in REQ if c not in raw.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

raw = raw.assign(Suite=raw["tags.suite"]).drop(columns=raw.filter(regex=r"^tags\\.").columns)
raw = strip(raw)
raw = mk_domain(raw)
raw = mk_image_size(raw)

raw = raw.rename(
    columns={
        "wall_time": "Wall Time(s)",
        "comp_total": "Comp Total",
        "comm_total": "Comm Total",
        "comm_send_total": "Comm Send Total",
        "comm_recv_total": "Comm Recv Total",
        "n_ranks": "N Ranks",
        "chunk_size": "Chunk Size",
        "schedule": "Schedule",
        "communication": "Communication",
        "run_id": "Run Id",
        "suite": "Suite",
    }
)

runs_df = raw[
    [
        "Image Size",
        "N Ranks",
        "Communication",
        "Schedule",
        "Chunk Size",
        "Domain",
        "Run Id",
        "Suite",
        "Wall Time(s)",
        "Comp Total",
        "Comm Total",
        "Comm Send Total",
        "Comm Recv Total",
    ]
].copy()

runs_idx = build_index(
    runs_df,
    levels=[
        "Schedule",
        "Communication",
        "N Ranks",
        "Chunk Size",
        "Domain",
        "Image Size",
        "Run Id",
        "Suite",
    ],
)

print(f"[info] Runs: {len(runs_df)} rows, cols={list(runs_df.columns)}")

meta_for_artifacts = [
    "Image Size",
    "N Ranks",
    "Communication",
    "Schedule",
    "Chunk Size",
    "Domain",
    "Run Id",
    "Suite",
]

chunks_df = load_table_for_all_runs(runs_df, ARTIFACTS["chunks"], meta_for_artifacts)
for col in ["comp_time", "comm_time", "start_time", "end_time"]:
    if col in chunks_df.columns:
        chunks_df[col] = pd.to_numeric(chunks_df[col], errors="coerce")
for col in ["N Ranks", "Chunk Size", "rank", "chunk_id"]:
    if col in chunks_df.columns:
        chunks_df[col] = pd.to_numeric(chunks_df[col], errors="coerce").astype("Int64")
chunks_idx = build_index(
    chunks_df,
    levels=[
        "Schedule",
        "Communication",
        "N Ranks",
        "Chunk Size",
        "Domain",
        "Image Size",
        "Run Id",
        "Suite",
        "rank",
        "chunk_id",
    ],
)

ranks_df = load_table_for_all_runs(runs_df, ARTIFACTS["ranks"], meta_for_artifacts)
for col in ["comp_time", "comm_time", "comm_send_time", "comm_recv_time"]:
    if col in ranks_df.columns:
        ranks_df[col] = pd.to_numeric(ranks_df[col], errors="coerce")
for col in ["N Ranks", "Chunk Size", "rank", "chunks"]:
    if col in ranks_df.columns:
        ranks_df[col] = pd.to_numeric(ranks_df[col], errors="coerce").astype("Int64")
ranks_idx = build_index(
    ranks_df,
    levels=[
        "Schedule",
        "Communication",
        "N Ranks",
        "Chunk Size",
        "Domain",
        "Image Size",
        "Run Id",
        "Suite",
        "rank",
    ],
)

# runs_df.to_parquet(os.path.join(OUT_DIR, "runs_df.parquet"), index=False)
runs_idx.to_parquet(os.path.join(OUT_DIR, "runs_indexed.parquet"))
# chunks_df.to_parquet(os.path.join(OUT_DIR, "chunks_df.parquet"), index=False)
chunks_idx.to_parquet(os.path.join(OUT_DIR, "chunks_indexed.parquet"))
# ranks_df.to_parquet(os.path.join(OUT_DIR, "ranks_df.parquet"), index=False)
ranks_idx.to_parquet(os.path.join(OUT_DIR, "ranks_indexed.parquet"))

print(f"[info] Saved all parquet files to {OUT_DIR}")
