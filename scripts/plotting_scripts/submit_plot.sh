#!/bin/bash
#BSUB -J submit_plot
#BSUB -o submit_plot_%J.out
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -W 00:05
#BSUB -q hpcintro
#BSUB -R "rusage[mem=1GB]"

module load python3/3.11.8
module load mpi/5.0.8-gcc-13.4.0-binutils-2.44

source ~/myenv/bin/activate

python scripts/plot_chunk_size_test_2.py