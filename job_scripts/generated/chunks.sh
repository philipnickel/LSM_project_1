#!/bin/bash
#BSUB -J chunks[1-20]
#BSUB -q hpcintro
#BSUB -n 8
#BSUB -W 00:15
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2GB]"
#BSUB -o logs/chunks_%J_%I.out
#BSUB -e logs/chunks_%J_%I.err

module load python3/3.11.1
module load mpi/5.0.8-gcc-13.4.0-binutils-2.44

cd ""
mkdir -p logs/chunks

uv sync
uv run python main.py --sweep configs/sweeps.yaml --suite chunks --task-id $((LSB_JOBINDEX - 1))
