#!/bin/bash
#BSUB -J blocking_dynamic
#BSUB -o blocking_dynamic_%J.out
#BSUB -e blocking_dynamic_%J.err
#BSUB -n 5
#BSUB -R "span[hosts=1]"
#BSUB -W 00:05
#BSUB -q hpcintro
#BSUB -R "rusage[mem=1GB]"

module load python3/3.11.8
module load mpi/5.0.8-gcc-13.4.0-binutils-2.44

source ~/myenv/bin/activate

mpirun python ./scripts/python_scripts/mandelbrot_mpi_blocking_dynamic.py 5 1000x3000