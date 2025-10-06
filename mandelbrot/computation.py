from __future__ import annotations

from typing import Tuple

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from main import RunConfig


def allocate_image(config: RunConfig) -> np.ndarray:
    return np.zeros(config.image_size)


def grid_constants(config: RunConfig) -> Tuple[float, float, float, float]:
    xconst = np.diff(config.xlim)[0] / config.width
    yconst = np.diff(config.ylim)[0] / config.height
    return config.xlim[0], xconst, config.ylim[0], yconst


def compute_chunk(config: RunConfig, chunk_id: int) -> Tuple[int, int, np.ndarray]:
    start_row = chunk_id * config.chunk_size
    end_row = min(start_row + config.chunk_size, config.width)
    if start_row >= config.width:
        return start_row, start_row, np.zeros((0, config.height), dtype=np.float64)

    chunk_rows = end_row - start_row
    chunk = np.zeros((chunk_rows, config.height))

    xlim_0, xconst, ylim_0, yconst = grid_constants(config)

    for local_x in range(chunk_rows):
        global_x = start_row + local_x
        cx = complex(xlim_0 + global_x * xconst, 0)
        for y in range(config.height):
            c = cx + complex(0, ylim_0 + y * yconst)
            z = 0
            for i in range(100):
                z = z * z + c
                if np.abs(z) > 2:
                    chunk[local_x, y] = i
                    break
    return start_row, end_row, chunk


def compute_full_image(config: RunConfig) -> np.ndarray:
    image = allocate_image(config)
    for chunk_id in range(config.total_chunks):
        start_row, end_row, chunk = compute_chunk(config, chunk_id)
        image[start_row:end_row, :] = chunk
    return image
