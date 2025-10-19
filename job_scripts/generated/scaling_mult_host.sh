#!/bin/bash
#BSUB -J scaling_mult_host[1-32]
#BSUB -q hpcintro
#BSUB -n 50
#BSUB -W 00:15
#BSUB -R "span[ptile=20]"
#BSUB -R "rusage[mem=3GB]"
#BSUB -o logs/scaling_mult_host_%J_%I.out
#BSUB -e logs/scaling_mult_host_%J_%I.err

module load python3/3.11.1
module load mpi/5.0.8-gcc-13.4.0-binutils-2.44

cd ""
mkdir -p logs/scaling_mult_host

uv sync
uv run python main.py --sweep configs/sweeps.yaml --suite scaling_mult_host --task-id $((LSB_JOBINDEX - 1))
