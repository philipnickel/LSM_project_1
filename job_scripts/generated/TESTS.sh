#!/bin/bash
#BSUB -J TESTS[1-4]
#BSUB -q hpcintro
#BSUB -n 1
#BSUB -W 00:05
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -o logs/TESTS_%J_%I.out
#BSUB -e logs/TESTS_%J_%I.err

module load python3/3.11.1
module load mpi/5.0.8-gcc-13.4.0-binutils-2.44

cd ""
mkdir -p logs/TESTS

uv sync
uv run python main.py --sweep configs/sweeps.yaml --suite TESTS --task-id $((LSB_JOBINDEX - 1))
