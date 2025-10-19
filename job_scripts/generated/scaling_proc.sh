#!/bin/bash
#BSUB -J scaling_proc[1-28]
#BSUB -q hpcintro
#BSUB -n 24
#BSUB -W 00:10
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=3GB]"
#BSUB -o logs/scaling_proc_%J_%I.out
#BSUB -e logs/scaling_proc_%J_%I.err

module purge
module load python3/3.11.1
module load mpi/5.0.8-gcc-13.4.0-binutils-2.44


# optional safety if you had crashes:
export OMPI_MCA_hwloc_base_binding_policy=none
export OMPI_MCA_rmaps_base_oversubscribe=1


cd ""
mkdir -p logs/scaling_proc

uv sync
uv run python main.py --sweep configs/sweeps.yaml --suite scaling_proc --task-id $((LSB_JOBINDEX - 1))
