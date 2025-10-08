#!/usr/bin/env python3
"""
Mandelbrot MPI Project - Main Entry Point

A modular implementation of the Mandelbrot set computation using MPI (Message Passing Interface)
for parallel processing. This project demonstrates different scheduling strategies (static/dynamic)
combined with communication patterns (blocking/non-blocking) for distributed computing.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict

import numpy as np
from mpi4py import MPI

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from mandelbrot import communication
from mandelbrot.scheduling import DynamicScheduler, StaticScheduler


@dataclass(frozen=True)
class RunConfig:
    chunk_size: int
    image_size: Tuple[int, int]
    xlim: Tuple[float, float]
    ylim: Tuple[float, float]
    schedule: str = "static"
    communication: str = "blocking"
    output: Optional[str] = None
    show_plot: bool = False
    save_data: bool = False
    log_experiment: bool = False
    save_plot: bool = False

    @property
    def width(self) -> int:
        return self.image_size[0]

    @property
    def height(self) -> int:
        return self.image_size[1]

    @property
    def total_chunks(self) -> int:
        return (self.width + self.chunk_size - 1) // self.chunk_size


def _parse_size(value: str) -> Tuple[int, int]:
    try:
        width, height = value.lower().split("x", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("size must look like WIDTHxHEIGHT") from exc
    return int(width), int(height)


def _parse_limits(value: str) -> Tuple[float, float]:
    try:
        lo, hi = value.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("limit must look like MIN:MAX") from exc
    return float(lo), float(hi)


def parse_args(argv: Optional[Sequence[str]] = None) -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Configure Mandelbrot MPI experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("chunk_size", type=int, nargs="?", default=10)
    parser.add_argument("size", type=_parse_size, nargs="?", default=(1000, 1000), help="image size WIDTHxHEIGHT")
    parser.add_argument("xlim", type=_parse_limits, nargs="?", default=(-2.2, 0.75), help="real axis limits MIN:MAX")
    parser.add_argument("ylim", type=_parse_limits, nargs="?", default=(-1.3, 1.3), help="imag axis limits MIN:MAX")
    parser.add_argument(
        "--schedule",
        choices=("static", "dynamic"),
        default="static",
        help="work scheduling strategy",
    )
    parser.add_argument(
        "--communication",
        choices=("blocking", "nonblocking"),
        default="blocking",
        help="result exchange strategy",
    )
    parser.add_argument("--output", type=str, help="path to save the rendered image")
    parser.add_argument(
        "--show",
        action="store_true",
        help="display the plot using matplotlib.show() on rank 0",
    )
    parser.add_argument(
        "--save-data",
        action="store_true",
        help="save the computed data array as .npy file",
    )
    parser.add_argument(
        "--log-experiment",
        action="store_true",
        help="log experiment data to CSV file",
    )
    parser.add_argument(
        "--save-plot",
        action="store_true",
        help="save plot as PDF file",
    )

    args = parser.parse_args(argv)
    return RunConfig(
        chunk_size=args.chunk_size,
        image_size=args.size,
        xlim=args.xlim,
        ylim=args.ylim,
        schedule=args.schedule,
        communication=args.communication,
        output=args.output,
        show_plot=args.show,
        save_data=args.save_data,
        log_experiment=args.log_experiment,
        save_plot=args.save_plot,
    )


def run(config: RunConfig) -> Optional[np.ndarray]:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Initialize experiment logger if requested
    logger = None
    if config.log_experiment:
        from mandelbrot.utils import ExperimentLogger
        logger = ExperimentLogger(config, comm)

    if config.schedule == "static":
        scheduler = StaticScheduler(config, comm.Get_size())
        image, worker_stats = communication.run_static(comm, config, scheduler, blocking=(config.communication == "blocking"), logger=logger)
    else:
        scheduler = DynamicScheduler(config)
        image, worker_stats = communication.run_dynamic(comm, config, scheduler, blocking=(config.communication == "blocking"), logger=logger)

    # Ensure all ranks complete computation before logging
    comm.barrier()
    
    # Log worker stats if logger exists
    if logger and worker_stats:
        for worker_rank, stats in worker_stats.items():
            logger.log_worker_stats(
                worker_rank, 
                stats['chunks_processed'],
                stats['computation_time'],
                stats['communication_time'],
                stats['chunk_ids']
            )
    
    if logger:
        # All ranks participate in finalize for data gathering
        csv_path = logger.finalize(image if rank == 0 else None, 
                                 save_plot=config.save_plot, save_data=config.save_data)
        if rank == 0 and csv_path:
            print(f"Experiment logged to {csv_path}")
    elif rank == 0 and image is not None:
        _maybe_render(image, config)
    
    return image if rank == 0 else None


def _maybe_render(image: np.ndarray, config: RunConfig) -> None:
    # Save data array if requested
    if config.save_data:
        output_file = f"modular_{config.schedule}_{config.communication}_{config.image_size[0]}x{config.image_size[1]}_{config.xlim[0]}_{config.xlim[1]}_{config.ylim[0]}_{config.ylim[1]}.npy"
        np.save(output_file, image)
        print(f"Data saved to {output_file}")
    
    # Handle plotting
    if config.output is not None or config.show_plot:
        import matplotlib.pyplot as plt

        extent = (config.xlim[0], config.xlim[1], config.ylim[0], config.ylim[1])
        plt.imshow(image.T, extent=extent, origin="lower")
        plt.xlabel("x / Re(p_0)")
        plt.ylabel("y / Im(p_0)")

        if config.output is not None:
            output_path = Path(config.output)
            _ensure_parent(output_path)
            plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
            print(f"Plot saved to {config.output}")
        if config.show_plot:
            plt.show()
        else:
            plt.close()
    elif not config.save_data:
        # Don't save anything by default - just print completion message
        print("Computation completed successfully")


def _ensure_parent(path: Path) -> None:
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    config = parse_args()
    run(config)


if __name__ == "__main__":
    main()