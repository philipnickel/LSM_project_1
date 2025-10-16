"""MLflow logging for Mandelbrot MPI experiments."""

from __future__ import annotations

import os
from typing import Dict, Optional

import mlflow
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from .config import RunConfig
from .report import ChunkReport

DEFAULT_TRACKING_URI = "databricks"


def log_to_mlflow(
    config: RunConfig,
    report: ChunkReport,
    suite_name: str = "default",
) -> None:
    """Log experiment to MLflow with rendered artifact and raw metrics.
    
    This function is called from within the MPI subprocess. If MLFLOW_RUN_ID
    is set in the environment, it continues an existing run (started by parent).
    Otherwise, it creates a new run.
    
    Args:
        config: Run configuration
        report: Combined outputs (image, timing stats, chunk table)
        suite_name: Name of the suite (TESTS, chunks, etc.) for tagging/filtering
    """
    # Skip logging in test mode
    if os.environ.get("SKIP_MLFLOW"):
        return
    
    tracking_uri = _resolve_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    _ensure_experiment()

    # Check if parent already started a run
    existing_run_id = os.environ.get("MLFLOW_RUN_ID")
    
    if existing_run_id:
        # Continue existing run (parent process started it)
        run_context = mlflow.start_run(run_id=existing_run_id)
    else:
        # Create new run (standalone execution)
        run_context = mlflow.start_run(run_name=config.run_name)
    
    with run_context as run:
        tags = {
            "schedule": config.schedule,
            "communication": config.communication,
            "n_ranks": config.n_ranks,
            "chunk_size": config.chunk_size,
            "image_width": config.width,
            "image_height": config.height,
            "node_name": os.uname().nodename,
            "suite": suite_name,  # Track which suite this run belongs to (TESTS, chunks, etc.)
        }

        job_id = (
            os.environ.get("LSB_JOBID")
            or os.environ.get("SLURM_JOB_ID")
            or os.environ.get("PBS_JOBID")
            or os.environ.get("JOB_ID")
        )
        if job_id:
            tags["job_id"] = job_id
            # Log LSF stdout/stderr if they exist
            _log_lsf_logs(job_id)

        mlflow.set_tags(tags)

        chunk_records = report.copy_chunks()
        chunk_df = None
        if chunk_records is not None:
            chunk_df = pd.DataFrame(chunk_records)
            mlflow.log_table(chunk_df, "chunks.json")
            mlflow.log_table(_summarize_chunks(chunk_df), "chunks_summary.json")

        _log_core_metrics(config, report.timing, chunk_df)
        if report.image is not None:
            _log_image_fig(report.image, "figures/mandelbrot.png")

        print(f"[MLflow] Logged run: {config.run_name} (suite: {suite_name})")
        print(f"[MLflow] Run ID: {run.info.run_id}")


def _log_core_metrics(
    config: RunConfig,
    timing_stats: dict,
    chunk_df: Optional[pd.DataFrame],
) -> None:
    mlflow.log_params(config.to_dict())

    metrics: Dict[str, float] = {
        "total_time": float(timing_stats.get("total_time", 0.0)),
        "total_comp_time": float(timing_stats.get("total_comp_time", 0.0)),
        "total_comm_time": float(timing_stats.get("total_comm_time", 0.0)),
        "comp_std": float(timing_stats.get("comp_std", 0.0)),
        "comm_std": float(timing_stats.get("comm_std", 0.0)),
        "total_chunks": float(timing_stats.get("total_chunks", 0)),
    }
    total_pixels = config.width * config.height
    metrics["total_pixels"] = total_pixels

    total_comp = metrics["total_comp_time"]
    total_comm = metrics["total_comm_time"]
    if total_comp is not None and total_comm is not None and total_pixels > 0:
        metrics["time_per_pixel"] = (total_comp + total_comm) / total_pixels
    elif metrics["total_time"] and total_pixels > 0:
        metrics["time_per_pixel"] = metrics["total_time"] / total_pixels

    metrics.update(_aggregate_rank_metrics(timing_stats))
    if chunk_df is not None and not chunk_df.empty:
        metrics.update(_aggregate_chunk_metrics(chunk_df))

    mlflow.log_metrics(metrics)


def _log_image_fig(image: np.ndarray, artifact_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image.T, cmap="viridis", origin="lower")
    ax.set_axis_off()
    mlflow.log_figure(fig, artifact_path)
    plt.close(fig)


def _summarize_chunks(chunk_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        chunk_df.groupby("rank", dropna=False)["comp_time"]
        .agg(mean="mean", std="std", min="min", max="max", count="count", sum="sum")
        .reset_index()
        .rename(columns={"sum": "total_comp_time"})
    )
    return summary


def _aggregate_rank_metrics(timing_stats: dict) -> Dict[str, float]:
    comp_times = [
        float(value)
        for key, value in timing_stats.items()
        if key.startswith("rank_") and key.endswith("_comp")
    ]
    comm_times = [
        float(value)
        for key, value in timing_stats.items()
        if key.startswith("rank_") and key.endswith("_comm")
    ]
    metrics: Dict[str, float] = {}
    if comp_times:
        metrics["avg_rank_comp_time"] = float(np.mean(comp_times))
        metrics["rank_comp_time_std"] = float(np.std(comp_times))
    if comm_times:
        metrics["avg_rank_comm_time"] = float(np.mean(comm_times))
        metrics["rank_comm_time_std"] = float(np.std(comm_times))
    return metrics


def _aggregate_chunk_metrics(chunk_df: pd.DataFrame) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    comp_series = chunk_df["comp_time"].dropna()
    if not comp_series.empty:
        metrics["avg_chunk_comp_time"] = float(comp_series.mean())
        metrics["chunk_comp_time_std"] = float(comp_series.std(ddof=0))
        metrics["max_chunk_comp_time"] = float(comp_series.max())
        metrics["min_chunk_comp_time"] = float(comp_series.min())
    return metrics


def _ensure_experiment() -> None:
    """Set active experiment using Databricks experiment ID or fallback to local."""
    experiment_id = os.environ.get("MLFLOW_EXPERIMENT_ID", "3399934008965459")
    # Always use MandelBrot_HPC experiment
    mlflow.set_experiment(experiment_id=experiment_id)


def _resolve_tracking_uri() -> str:
    """Resolve tracking URI, preferring Databricks when credentials are present."""
    uri = os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    return _normalize_tracking_uri(uri)


def _normalize_tracking_uri(uri: str) -> str:
    """Ensure bare paths are converted to valid file:// URIs."""
    if uri == "databricks" or uri.startswith("databricks:"):
        return uri
    if uri.startswith("file://") or "://" in uri:
        return uri
    return f"file://{uri}"


def _log_lsf_logs(job_id: str) -> None:
    """Log LSF stdout/stderr files as MLflow artifacts if they exist.
    
    This captures the full job output including module loads, uv sync, etc.
    """
    from pathlib import Path
    
    # Try environment variables first (set in job_template.sh)
    stdout_path = os.environ.get("MLFLOW_LSF_STDOUT")
    stderr_path = os.environ.get("MLFLOW_LSF_STDERR")
    
    for log_type, log_path in [("stdout", stdout_path), ("stderr", stderr_path)]:
        if log_path:
            log_file = Path(log_path)
            if log_file.exists():
                try:
                    mlflow.log_artifact(str(log_file), artifact_path="lsf_logs")
                    print(f"[MLflow] Logged {log_type}: {log_file.name}")
                except Exception as e:
                    print(f"[MLflow] Warning: Could not log {log_file}: {e}")
