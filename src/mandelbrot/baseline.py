"""Baseline Mandelbrot implementation."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_mandelbrot(size: Tuple[int, int], xlim, ylim) -> np.ndarray:
    """Compute Mandelbrot set for provided bounds."""
    width, height = size
    image = np.zeros(size, dtype=float)

    xconst = (xlim[1] - xlim[0]) / width
    yconst = (ylim[1] - ylim[0]) / height

    for x in range(width):
        cx = complex(xlim[0] + x * xconst, 0)
        for y in range(height):
            c = cx + complex(0, ylim[0] + y * yconst)
            z = 0
            for i in range(100):
                z = z * z + c
                if np.abs(z) > 2:
                    image[x, y] = i
                    break

    return image
