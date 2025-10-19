#!/bin/bash
#BSUB -J scaling_mult_host[1-8]
#BSUB -q hpcintro
#BSUB -n 70
#BSUB -W 00:10
#BSUB -R "span[ptile=10]"
#BSUB -R "rusage[mem=3GB]"
#BSUB -o logs/scaling_mult_host_%J_%I.out
#BSUB -e logs/scaling_mult_host_%J_%I.err

module purge
module load python3/3.11.1
module load mpi/5.0.8-gcc-13.4.0-binutils-2.44


# optional safety if you had crashes:
export OMPI_MCA_hwloc_base_binding_policy=none
export OMPI_MCA_rmaps_base_oversubscribe=1


cd ""
mkdir -p logs/scaling_mult_host

uv sync
uv run python main.py --sweep configs/sweeps.yaml --suite scaling_mult_host --task-id $((LSB_JOBINDEX - 1))
