from __future__ import annotations

from typing import Tuple

import numpy as np
from numba import njit, prange

from .config import RunConfig

__all__ = ["allocate_image", "grid_constants", "compute_chunk"]


@njit
def _allocate_image(width: int, height: int) -> np.ndarray:
    return np.zeros((width, height))


def allocate_image(config: RunConfig) -> np.ndarray:
    return _allocate_image(config.width, config.height)


@njit
def _grid_constants(
    width: int,
    height: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> Tuple[float, float, float, float]:
    xconst = (x_max - x_min) / width
    yconst = (y_max - y_min) / height
    return x_min, xconst, y_min, yconst


def grid_constants(config: RunConfig) -> Tuple[float, float, float, float]:
    return _grid_constants(
        config.width,
        config.height,
        float(config.xlim[0]),
        float(config.xlim[1]),
        float(config.ylim[0]),
        float(config.ylim[1]),
    )


@njit(parallel=True)
def _compute_chunk(
    width: int,
    height: int,
    chunk_size: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    chunk_id: int,
) -> Tuple[int, int, np.ndarray]:
    start_row = chunk_id * chunk_size
    end_row = min(start_row + chunk_size, width)
    if start_row >= width:
        return start_row, start_row, np.zeros((0, height), dtype=np.float64)

    chunk_rows = end_row - start_row
    chunk = np.zeros((chunk_rows, height))

    xlim_0, xconst, ylim_0, yconst = _grid_constants(width, height, x_min, x_max, y_min, y_max)

    for local_x in prange(chunk_rows):
        global_x = start_row + local_x
        cx = complex(xlim_0 + global_x * xconst, 0.0)
        for y in range(height):
            c = cx + complex(0.0, ylim_0 + y * yconst)
            z = 0.0 + 0.0j
            for i in range(100):
                z = z * z + c
                if np.abs(z) > 2.0:
                    chunk[local_x, y] = i
                    break

    return start_row, end_row, chunk


def compute_chunk(config: RunConfig, chunk_id: int) -> Tuple[int, int, np.ndarray]:
    return _compute_chunk(
        config.width,
        config.height,
        config.chunk_size,
        float(config.xlim[0]),
        float(config.xlim[1]),
        float(config.ylim[0]),
        float(config.ylim[1]),
        chunk_id,
    )
