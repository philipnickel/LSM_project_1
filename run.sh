#!/bin/bash
# Wrapper script that loads modules if on HPC, then runs main.py

# Check if running on HPC (look for module command)
if command -v module &> /dev/null; then
    module load python3/3.11.1 2>/dev/null || true
    module load mpi/5.0.8-gcc-13.4.0-binutils-2.44 2>/dev/null || true
fi

# Run main.py with all arguments passed through
uv run python main.py "$@"
