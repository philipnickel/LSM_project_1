# Source this file on HPC: source env_hpc.sh
# Loads required modules for development/testing on HPC

module load python3/3.11.1
module load mpi/5.0.8-gcc-13.4.0-binutils-2.44

echo "✓ HPC modules loaded"
echo "  - Python 3.11.1"
echo "  - MPI 5.0.8"
