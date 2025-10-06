#!/bin/bash
# Test runner that tests against all available baseline sizes

set -e  # Exit on any error

# Get parameters with defaults
N=${1:-1}                    # Number of MPI processes
SCHEDULE=${2:-static}        # Scheduling strategy
COMMUNICATION=${3:-blocking} # Communication pattern

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "Testing $SCHEDULE + $COMMUNICATION with $N MPI processes against all baselines"

# Find all baseline files
baseline_files=$(find tests/baseline_data -name "baseline_*.npy" 2>/dev/null || true)

if [ -z "$baseline_files" ]; then
    echo "ERROR: No baseline files found in tests/baseline_data/"
    echo "Run 'make baseline' first to generate baseline data"
    exit 1
fi

# Test against each baseline
for baseline_file in $baseline_files; do
    # Extract size from filename (baseline_200x150.npy -> 200x150)
    size=$(basename "$baseline_file" .npy | sed 's/baseline_//' | sed 's/x/x/')
    
    echo ""
    echo "Testing size: $size"
    
    # Run modular implementation
    echo "Running computation..."
    if ! mpirun -n $N python main.py 10 $size --schedule $SCHEDULE --communication $COMMUNICATION --save-data; then
        echo "ERROR: Computation failed for size $size"
        exit 1
    fi
    
    # Check if file was created
    expected_file="modular_${SCHEDULE}_${COMMUNICATION}_${size//x/x}.npy"
    if [ ! -f "$expected_file" ]; then
        echo "ERROR: Modular file not created: $expected_file"
        exit 1
    fi
    
    # Compare arrays
    echo "Comparing with baseline..."
    if ! python -c "
import numpy as np
baseline = np.load('$baseline_file')
modular = np.load('$expected_file')

if np.array_equal(baseline, modular):
    print('Arrays are identical for size $size')
else:
    print('ERROR: Arrays differ for size $size')
    exit(1)
"; then
        echo "ERROR: Array comparison failed for size $size"
        exit 1
    fi
done

echo ""
echo "All $SCHEDULE + $COMMUNICATION tests passed for all sizes!"
exit 0