"""Baseline (serial) Mandelbrot implementation."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Tuple

import numpy as np

try:  # pragma: no cover - keeps script usable via direct invocation
    from .config import default_run_config
except ImportError:  # pragma: no cover
    from config import default_run_config  # type: ignore

if TYPE_CHECKING:
    from .config import RunConfig

DEFAULT_CONFIG = default_run_config()


def compute_mandelbrot(size: Tuple[int, int], xlim, ylim, max_iter: int = 100) -> np.ndarray:
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
            for i in range(max_iter):
                z = z * z + c
                if np.abs(z) > 2:
                    image[x, y] = i
                    break

    return image


def compute_baseline(config: "RunConfig") -> np.ndarray:
    """Compute Mandelbrot set for a configuration (used by regression tests)."""
    size = (config.width, config.height)
    max_iter = getattr(config, "max_iter", DEFAULT_CONFIG.max_iter)
    return compute_mandelbrot(size, config.xlim, config.ylim, max_iter)


def _print_usage(script: str) -> None:
    print(
        f"""\
{script} [chunk-size] [size widthXheight] [limits xmin:xmax ymin:ymax]

Here are some examples:

Call it with a chunk-size of 10
$ {script} 10

Call it with a chunk-size of 10 and image size of 100 by 500 pixels
$ {script} 10 100x500

Call it with a chunk-size of 10 and image size of 100 by 500 pixels
spanning the coordinates x in 0.1-0.3 and y in 0.2-0.3
$ {script} 10 100x500 0.1:0.3 0.2:0.3
"""
    )


def _cli(argv: list[str]) -> int:
    chunk_size = DEFAULT_CONFIG.chunk_size
    size = (DEFAULT_CONFIG.width, DEFAULT_CONFIG.height)
    xlim = DEFAULT_CONFIG.xlim
    ylim = DEFAULT_CONFIG.ylim

    if argv:
        chunk_size = int(argv.pop(0))
    if argv:
        width, height = map(int, argv.pop(0).split("x"))
        size = (width, height)
    if argv:
        xmin, xmax = map(float, argv.pop(0).split(":"))
        xlim = (xmin, xmax)
    if argv:
        ymin, ymax = map(float, argv.pop(0).split(":"))
        ylim = (ymin, ymax)

    print(
        f"""\
Calculating the Mandelbrot set with these arguments:

chunk_size = {chunk_size}
size = {size}
xlim = {xlim}
ylim = {ylim}
"""
    )

    compute_mandelbrot(size, xlim, ylim, DEFAULT_CONFIG.max_iter)
    return 0


if __name__ == "__main__":
    if any(h in sys.argv for h in ("help", "-h", "-help", "--help")):
        _print_usage(sys.argv[0])
        sys.exit(0)

    sys.exit(_cli(sys.argv[1:]))
