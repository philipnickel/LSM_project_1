#!/bin/bash
#BSUB -J ${JOB_NAME}[${ARRAY_RANGE}]
#BSUB -q ${QUEUE}
#BSUB -n ${N_CORES}
#BSUB -W ${WALLTIME}
${SPAN_DIRECTIVE}
#BSUB -R "rusage[mem=${MEM_PER_CORE}]"
#BSUB -o logs/${JOB_NAME}_%J_%I.out
#BSUB -e logs/${JOB_NAME}_%J_%I.err

module purge
module load python3/3.11.1
module load mpi/5.0.8-gcc-13.4.0-binutils-2.44


# optional safety if you had crashes:
export OMPI_MCA_hwloc_base_binding_policy=none
export OMPI_MCA_rmaps_base_oversubscribe=1


cd "${LSB_SUBCWD}"
mkdir -p logs/${JOB_NAME}

uv sync
uv run python main.py --sweep configs/sweeps.yaml --suite ${SUITE} --task-id $((LSB_JOBINDEX - 1))
