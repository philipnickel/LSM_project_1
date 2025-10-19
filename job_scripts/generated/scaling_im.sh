#!/bin/bash
#BSUB -J scaling_im[1-24]
#BSUB -q hpcintro
#BSUB -n 16
#BSUB -W 00:15
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5GB]"
#BSUB -o logs/scaling_im_%J_%I.out
#BSUB -e logs/scaling_im_%J_%I.err

module load python3/3.11.1
module load mpi/5.0.8-gcc-13.4.0-binutils-2.44

cd ""
mkdir -p logs/scaling_im

uv sync
uv run python main.py --sweep configs/sweeps.yaml --suite scaling_im --task-id $((LSB_JOBINDEX - 1))
