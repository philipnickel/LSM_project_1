#!/usr/bin/env python3
"""Load MLflow runs and chunk artifacts into analysis-ready DataFrames."""

# %%
from __future__ import annotations

import os
from typing import Iterable

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException


# %%
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if not tracking_uri:
    raise RuntimeError("MLFLOW_TRACKING_URI is not set. Check your Databricks config.")

experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")
if not experiment_id:
    raise RuntimeError("MLFLOW_EXPERIMENT_ID is not set. Provide it via the environment.")

mlflow.set_tracking_uri(tracking_uri)
print(f"Connected to MLflow tracking URI: {tracking_uri}")
print(f"Using experiment: {experiment_id}")


# %%
runs_df = mlflow.search_runs(experiment_ids=[experiment_id]).sort_values("start_time", ascending=False)
print(f"Fetched {len(runs_df)} runs")
print(runs_df.head())


# %%
def load_chunk_tables(run_ids: Iterable[str], artifact_file: str = "chunks.json") -> pd.DataFrame:
    """Return concatenated chunk tables for the provided run IDs."""
    tables: list[pd.DataFrame] = []

    for run_id in run_ids:
        try:
            table = mlflow.load_table(artifact_file, run_ids=[run_id])
        except MlflowException as exc:
            print(f"[warning] Unable to load {artifact_file} for run {run_id}: {exc}")
            continue

        if table.empty:
            print(f"[warning] Empty chunk table for run {run_id}")
            continue

        if "mlflow.run_id" in table.columns:
            table = table.rename(columns={"mlflow.run_id": "run_id"})
        elif "run_id" not in table.columns:
            table["run_id"] = run_id

        tables.append(table)

    if not tables:
        return pd.DataFrame(columns=["run_id"])

    return pd.concat(tables, ignore_index=True)


chunks_df = load_chunk_tables(runs_df["run_id"])
print(f"Loaded {len(chunks_df)} chunk rows across {chunks_df['run_id'].nunique() if not chunks_df.empty else 0} runs")
if not chunks_df.empty:
    print(chunks_df.head())


# %%
if not chunks_df.empty:
    aggregations = {}
    if "comp_time" in chunks_df.columns:
        aggregations["chunk_comp_mean"] = ("comp_time", "mean")
        aggregations["chunk_comp_max"] = ("comp_time", "max")
    if "chunk_id" in chunks_df.columns:
        aggregations["chunk_count"] = ("chunk_id", "count")

    if aggregations:
        chunk_stats = chunks_df.groupby("run_id").agg(**aggregations).reset_index()
        runs_with_chunk_stats = runs_df.merge(chunk_stats, on="run_id", how="left")
        print("Preview of run-level metrics with chunk stats:")
        print(runs_with_chunk_stats.head())
    else:
        print("Chunk table loaded, but no recognized columns to aggregate.")
else:
    print("No chunk tables found to augment run metadata.")
