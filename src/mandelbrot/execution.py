"""Execution helpers for Mandelbrot CLI workflows."""

from __future__ import annotations

import os
import selectors
import subprocess
import sys
from pathlib import Path
from typing import Optional

from mpi4py import MPI

from .config import RunConfig, load_sweep_configs
from .logging import EXPERIMENT_ID, _resolve_tracking_uri, log_to_mlflow
from .mpi import run_mpi_computation


def run_single_experiment(
    config: RunConfig,
    suite_name: Optional[str],
) -> None:
    """Execute a single Mandelbrot computation within an MPI context."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(
            f"[Run] Starting computation '{config.run_name}' "
            f"(ranks={size}, schedule={config.schedule}, "
            f"communication={config.communication}, chunks={config.total_chunks})",
            flush=True,
        )

    report = run_mpi_computation(config)

    if rank != 0:
        return

    suite = suite_name or os.environ.get("MANDELBROT_SUITE") or "default"

    if os.environ.get("SKIP_MLFLOW"):
        print("[Run] SKIP_MLFLOW set - skipping MLflow logging.", flush=True)
    else:
        print("[Run] Computation finished, logging to MLflow...", flush=True)

    log_to_mlflow(config, report, suite)

    wall_time = report.timing.get("wall_time", 0.0)
    print(f"[Timing] Total: {wall_time:.4f}s")


def run_sweep(
    config_path: str | Path | None,
    task_id: Optional[int] = None,
    suite_name: Optional[str] = None,
    configs: Optional[list[RunConfig]] = None,
    descriptor: Optional[str] = None,
) -> int:
    """Run a sweep defined in a YAML configuration file or a pre-loaded list."""
    if configs is None:
        if config_path is None:
            raise ValueError("config_path must be provided when configs is None")
        configs = load_sweep_configs(config_path)
        descriptor = descriptor or str(config_path)
    else:
        descriptor = descriptor or (str(config_path) if config_path else "sweep")

    if not configs:
        print("ERROR: No configurations found in sweep", file=sys.stderr)
        return 1

    if task_id is not None:
        if task_id < 0 or task_id >= len(configs):
            print(f"ERROR: task-id {task_id} out of range [0, {len(configs) - 1}]", file=sys.stderr)
            return 1
        config = configs[task_id]
        print(f"[Task {task_id}] Running: {config.run_name}")
        run_single_config_subprocess(
            config,
            task_id,
            len(configs),
            show_progress=False,
            suite_name=suite_name,
        )
        return 0

    print("=" * 70)
    print(f"Running {len(configs)} configurations from {descriptor}")
    print("=" * 70)

    successes = 0
    failures: list[tuple[int, str]] = []

    for idx, cfg in enumerate(configs):
        success = run_single_config_subprocess(
            cfg,
            idx,
            len(configs),
            suite_name=suite_name,
        )
        if success:
            successes += 1
        else:
            failures.append((idx, cfg.run_name))

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total:      {len(configs)}")
    print(f"Successful: {successes}")
    print(f"Failed:     {len(failures)}")

    if failures:
        print("\nFailed configurations:")
        for idx, name in failures:
            print(f"  [{idx}] {name}")
        return 1

    return 0


def run_single_config_subprocess(
    config: RunConfig,
    config_idx: int,
    total_configs: int,
    *,
    show_progress: bool = True,
    suite_name: Optional[str] = None,
) -> bool:
    """Execute a single configuration as a subprocess via mpirun."""
    if show_progress:
        print(f"\n[{config_idx + 1}/{total_configs}] {config.run_name}")
        print(
            "    n_ranks=%s, chunk_size=%s, schedule=%s, communication=%s"
            % (config.n_ranks, config.chunk_size, config.schedule, config.communication)
        )

    # Skip MLflow logging in test mode
    skip_mlflow = os.environ.get("SKIP_MLFLOW")

    import mlflow

    run_context = None
    if not skip_mlflow:
        tracking_uri = _resolve_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_id=EXPERIMENT_ID)
        run_context = mlflow.start_run(run_name=config.run_name)
        run_context.__enter__()

    cmd, env = build_command(config)

    if suite_name:
        env["MANDELBROT_SUITE"] = suite_name

    # Pass the active run_id to subprocess
    if run_context:
        env["MLFLOW_RUN_ID"] = run_context.info.run_id

    # Run subprocess, and capture for MLflow
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    stdout_text = ""
    stderr_text = ""
    returncode: int | None = None

    proc = subprocess.Popen(
        cmd,
        text=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )

    selector = selectors.DefaultSelector()
    if proc.stdout is not None:
        selector.register(proc.stdout, selectors.EVENT_READ)
    if proc.stderr is not None:
        selector.register(proc.stderr, selectors.EVENT_READ)

    try:
        try:
            while selector.get_map():
                for key, _ in selector.select():
                    stream = key.fileobj
                    data = stream.readline()
                    if data == "":
                        selector.unregister(stream)
                        stream.close()
                        continue
                    if stream is proc.stdout:
                        print(data, end="")
                        stdout_chunks.append(data)
                        sys.stdout.flush()
                    else:
                        print(data, end="", file=sys.stderr)
                        stderr_chunks.append(data)
                        sys.stderr.flush()
        except KeyboardInterrupt:
            proc.kill()
            proc.wait()
            raise
        finally:
            selector.close()

        returncode = proc.wait()

        stdout_text = "".join(stdout_chunks)
        stderr_text = "".join(stderr_chunks)

        if run_context:
            if stdout_text:
                mlflow.log_text(stdout_text, "logs/stdout.txt")
            if stderr_text:
                mlflow.log_text(stderr_text, "logs/stderr.txt")
    finally:
        if run_context:
            exc_type, exc_value, exc_tb = sys.exc_info()
            run_context.__exit__(exc_type, exc_value, exc_tb)

    if returncode is None:
        return False

    if returncode != 0:
        print(f"    ✗ FAILED with exit code {returncode}", file=sys.stderr)
        if stderr_text:
            print(f"    Error: {stderr_text[:200]}...", file=sys.stderr)
        return False

    if show_progress:
        print("    ✓ Completed")
    return True


def build_command(config: RunConfig) -> tuple[list[str], dict[str, str]]:
    """Build the mpirun command and environment for a single configuration."""
    cmd = ["mpirun", "-n", str(config.n_ranks), sys.executable, sys.argv[0]]

    env = os.environ.copy()

    cmd.extend(config.to_cli_args())

    return cmd, env
