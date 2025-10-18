"""Mandelbrot MPI - Simplified architecture with MLflow tracking."""

__version__ = "2.0.0"

# Core computation and config - lightweight, imported by MPI workers
from .computation import compute_chunk
from .config import RunConfig, default_run_config
from .report import ChunkReport


# Conditional imports - only loaded when needed
def __getattr__(name):
    """Lazy loading of heavy modules."""
    if name == "run_mpi_computation":
        from .mpi import run_mpi_computation

        return run_mpi_computation
    elif name == "load_sweep_configs":
        from .config import load_sweep_configs

        return load_sweep_configs
    elif name == "get_config_by_index":
        from .config import get_config_by_index

        return get_config_by_index
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RunConfig",
    "default_run_config",
    "compute_chunk",
    "run_mpi_computation",
    "ChunkReport",
    "load_sweep_configs",
    "get_config_by_index",
]
