# %%
from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import numba
import numpy as np
import pandas as pd

from mandelbrot.baseline import compute_mandelbrot as baseline_compute

# %%

XLIM = (-2.2, 0.75)
YLIM = (-1.3, 1.3)
SIZES = ["125x125", "250x250", "500x500", "1000x1000", "2000x2000", "4000x4000"]
MAX_ITER = 100


# %%
@numba.njit(parallel=True)
def mandelbrot_numba_parallel(size: Tuple[int, int], xlim, ylim, max_iter: int = 100) -> np.ndarray:
    width, height = size
    image = np.zeros(size, dtype=np.float64)

    xconst = (xlim[1] - xlim[0]) / width
    yconst = (ylim[1] - ylim[0]) / height

    for x in numba.prange(width):
        cx = complex(xlim[0] + x * xconst, 0.0)
        for y in range(height):
            c = cx + complex(0.0, ylim[0] + y * yconst)
            z = 0.0 + 0.0j
            for i in range(max_iter):
                z = z * z + c
                if np.abs(z) > 2.0:
                    image[x, y] = i
                    break
    return image


@numba.njit
def mandelbrot_numba(size: Tuple[int, int], xlim, ylim, max_iter: int = 100) -> np.ndarray:
    width, height = size
    image = np.zeros(size, dtype=np.float64)

    xconst = (xlim[1] - xlim[0]) / width
    yconst = (ylim[1] - ylim[0]) / height

    for x in range(width):
        cx = complex(xlim[0] + x * xconst, 0.0)
        for y in range(height):
            c = cx + complex(0.0, ylim[0] + y * yconst)
            z = 0.0 + 0.0j
            for i in range(max_iter):
                z = z * z + c
                if np.abs(z) > 2.0:
                    image[x, y] = i
                    break
    return image


# %%
# Trigger JIT compilation once so timings below reflect steady-state performance.
mandelbrot_numba_parallel((64, 64), XLIM, YLIM, 20)

# %%

rows = []

for size_str in SIZES:
    width, height = map(int, size_str.lower().split("x"))
    size = (width, height)

    print(f"Running baseline for size {size_str}...")
    start = time.perf_counter()
    baseline_compute(size, XLIM, YLIM)
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

    print(f"Running numba (single-threaded) for size {size_str}...")
    start = time.perf_counter()
    mandelbrot_numba(size, XLIM, YLIM, MAX_ITER)
    numba_time = time.perf_counter() - start
    rows.append(
        {
            "Implementation": "Numba",
            "threads": numba.get_num_threads(),
            "Image Size": size_str,
            "Time (s)": numba_time,
            "numba_threads_actual": numba.get_num_threads(),
        }
    )

    print(f"Running numba (multi-threaded) for size {size_str}...")
    start = time.perf_counter()
    mandelbrot_numba_parallel(size, XLIM, YLIM, MAX_ITER)
    numba_time = time.perf_counter() - start
    rows.append(
        {
            "Implementation": "Numba Parallel",
            "threads": numba.get_num_threads(),
            "Image Size": size_str,
            "Time (s)": numba_time,
            "numba_threads_actual": numba.get_num_threads(),
        }
    )

# %%
df = pd.DataFrame(rows)

try:
    script_dir = Path(__file__).resolve().parent
except NameError:
    script_dir = Path.cwd()
out_path = script_dir / "bench_results.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved results to {out_path}")
