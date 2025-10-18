"""Helper script to run MPI computation and save result."""

import sys
import numpy as np
from mpi4py import MPI
from mandelbrot.config import load_sweep_configs
from mandelbrot.mpi import run_mpi_computation

try:
    config_file = sys.argv[1]
    config_idx = int(sys.argv[2])
    output_file = sys.argv[3]

    configs = load_sweep_configs(config_file)
    config = configs[config_idx]

    report = run_mpi_computation(config)

    if MPI.COMM_WORLD.rank == 0:
        np.save(output_file, report.image)

    sys.exit(0)

except Exception as e:
    if MPI.COMM_WORLD.rank == 0:
        print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
