# %% Imports ---------------------------------------------------------------
from __future__ import annotations

import time
from pathlib import Path

import numba
import numpy as np
import pandas as pd

from mandelbrot.baseline import compute_mandelbrot as baseline_compute
from mandelbrot.computation import compute_chunk
from mandelbrot.config import RunConfig

# %% Benchmark configuration -----------------------------------------------
XLIM = (-2.2, 0.75)
YLIM = (-1.3, 1.3)
SIZES = ["125x125", "250x250", "500x500", "1000x1000", "2000x2000", "4000x4000"]
THREADS = [1, 2, 4, 8, numba.get_num_threads()]
MAX_ITER = 100


# %% Helpers ----------------------------------------------------------------


def _make_config(width: int, height: int) -> RunConfig:
    chunk_size = max(1, min(64, width))
    return RunConfig(
        n_ranks=1,
        chunk_size=chunk_size,
        schedule="static",
        communication="blocking",
        width=width,
        height=height,
        xlim=XLIM,
        ylim=YLIM,
    )


def _compute_with_numba(cfg: RunConfig, threads: int) -> None:
    image = np.empty((cfg.width, cfg.height), dtype=np.float64)
    previous = numba.get_num_threads()
    numba.set_num_threads(threads)
    try:
        for chunk_id in range(cfg.total_chunks):
            start, end, chunk = compute_chunk(cfg, chunk_id)
            image[start:end, :] = chunk
    finally:
        numba.set_num_threads(previous)


# Trigger JIT compilation once so timings below reflect steady-state behaviour.
_warmup_cfg = _make_config(64, 64)
compute_chunk(_warmup_cfg, 0)


# %% Benchmark loop ---------------------------------------------------------
rows: list[dict[str, object]] = []

for size_str in SIZES:
    width, height = map(int, size_str.lower().split("x"))
    size_tuple = (width, height)
    cfg = _make_config(width, height)

    print(f"Running baseline for size {size_str}...")
    start = time.perf_counter()
    baseline_compute(size_tuple, XLIM, YLIM)
    baseline_time = time.perf_counter() - start
    rows.append(
        {
            "Implementation": "Baseline",
            "threads": 0,
            "Image Size": size_str,
            "Time (s)": baseline_time,
            "numba_threads_actual": "",
        }
    )

    for threads in THREADS:
        if threads < 1:
            continue
        label = "Numba" if threads == 1 else f"Numba ({threads} threads)"
        print(f"Running {label.lower()} for size {size_str}...")
        start = time.perf_counter()
        _compute_with_numba(cfg, threads)
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "Implementation": label,
                "threads": threads,
                "Image Size": size_str,
                "Time (s)": elapsed,
                "numba_threads_actual": threads,
            }
        )


# %% Results ----------------------------------------------------------------
df = pd.DataFrame(rows)

try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    script_dir = Path.cwd()
out_path = script_dir / "bench_results.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved results to {out_path}")
