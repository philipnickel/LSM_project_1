from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mandelbrot.config import default_run_config, load_named_sweep_configs
from mandelbrot.execution import run_single_experiment, run_sweep


def parse_args():
    parser = argparse.ArgumentParser(description="Run Mandelbrot MPI experiments.")
    parser.add_argument("--sweep", type=str, help="Path to sweep YAML file")
    parser.add_argument("--suite", type=str, help="Name of suite/experiment within sweep file")
    parser.add_argument("--list-suites", action="store_true", help="List suites in sweep file")
    parser.add_argument("--task-id", type=int, help="Run specific config index (for HPC arrays)")

    # Direct run parameters (hidden from help)
    parser.add_argument("--n-ranks", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--chunk-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--schedule", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--communication", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--image-size", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--xlim", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--ylim", type=str, help=argparse.SUPPRESS)

    return parser.parse_args()


def main():
    args = parse_args()

    # Handle sweep runs
    if args.sweep:
        sweep_path = Path(args.sweep)

        if args.list_suites:
            suites = load_named_sweep_configs(sweep_path)
            for name, configs in suites:
                label = name or sweep_path.stem
                print(f"{label}: {len(configs)} configurations")
            return 0

        if args.suite and not args.sweep:
            sys.exit("ERROR: --suite requires --sweep")

        if args.task_id is not None and args.suite is None:
            sys.exit("ERROR: --task-id requires --suite")

        suites = load_named_sweep_configs(sweep_path, args.suite) if args.suite else load_named_sweep_configs(sweep_path)

        exit_code = 0
        for suite_name, configs in suites:
            descriptor = f"{sweep_path}::{suite_name}" if suite_name else str(sweep_path)
            rc = run_sweep(sweep_path, args.task_id, suite_name, configs, descriptor)
            exit_code = exit_code or rc
        return exit_code

    # Handle direct CLI run - all args required
    if not all([args.n_ranks, args.chunk_size, args.schedule, args.communication, args.image_size]):
        sys.exit("ERROR: Direct run requires: --n-ranks, --chunk-size, --schedule, --communication, --image-size")

    xlim = tuple(map(float, args.xlim.split(":"))) if args.xlim else (-2.2, 0.75)
    ylim = tuple(map(float, args.ylim.split(":"))) if args.ylim else (-1.3, 1.3)

    config = default_run_config(
        n_ranks=args.n_ranks,
        chunk_size=args.chunk_size,
        schedule=args.schedule,
        communication=args.communication,
        image_size=args.image_size,
        xlim=xlim,
        ylim=ylim,
    )

    run_single_experiment(config, None)
    return 0


if __name__ == "__main__":
    sys.exit(main())

