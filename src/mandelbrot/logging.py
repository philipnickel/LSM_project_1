"""MLflow logging for Mandelbrot MPI experiments."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence

import mlflow
import pandas as pd
from matplotlib import pyplot as plt

from .config import RunConfig
from .report import ChunkReport

DEFAULT_TRACKING_URI = "databricks"
EXPERIMENT_ID = "3399934008965459"
EXPERIMENT_FALLBACK_NAME = "mandelbrot_local"


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
    mlflow.set_experiment(experiment_id=EXPERIMENT_ID)

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
            "node_name": os.uname().nodename,
            "suite": suite_name,  # Track which suite this run belongs to (TESTS, chunks, etc.)
        }

        job_id = os.environ.get("LSB_JOBID")
        if job_id:
            tags["job_id"] = job_id

        mlflow.set_tags(tags)

        chunk_records = report.copy_chunks()
        if chunk_records:
            mlflow.log_table(_records_to_table(chunk_records), "chunks.json")

        timing_stats = report.timing or {}

        rank_records = timing_stats.get("rank_stats")
        if isinstance(rank_records, list) and rank_records:
            mlflow.log_table(_records_to_table(rank_records), "ranks.json")

        mlflow.log_params(config.to_dict())

        wall_time = float(timing_stats.get("wall_time", 0.0))
        comp_total = float(timing_stats.get("comp_total", 0.0))
        comm_total = float(timing_stats.get("comm_total", 0.0))
        comm_send_total = float(timing_stats.get("comm_send_total", 0.0))
        comm_recv_total = float(timing_stats.get("comm_recv_total", 0.0))
        total_chunks = float(timing_stats.get("total_chunks", 0))

        metrics = {
            "wall_time": wall_time,
            "comp_total": comp_total,
            "comm_total": comm_total,
            "comm_send_total": comm_send_total,
            "comm_recv_total": comm_recv_total,
            "total_chunks": total_chunks,
        }
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        if report.image is not None:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(report.image.T)
            mlflow.log_figure(fig, "figures/mandelbrot.png")
            plt.close(fig)

        print(f"[MLflow] Logged run: {config.run_name} (suite: {suite_name})")
        print(f"[MLflow] Run ID: {run.info.run_id}")


def _records_to_table(chunk_records: Sequence[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Convert row-wise chunk records into MLflow table format."""

    frame = pd.DataFrame.from_records(chunk_records)
    return frame.to_dict(orient="list")


def _resolve_tracking_uri() -> str:
    """Resolve tracking URI."""
    return os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
