#!/bin/bash
# Generate baseline data array for testing

set -e  # Exit on any error

# Get size parameter (default: 200x150)
SIZE=${1:-200x150}

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "Generating baseline data array..."

echo "Generating baseline for size: $SIZE"
cd tests/baseline_data
python ../Mandelbrot.py 10 $SIZE
cd ../..

# Check if file was created
expected_file="tests/baseline_data/baseline_${SIZE//x/x}.npy"
if [ ! -f "$expected_file" ]; then
    echo "ERROR: Baseline file not created: $expected_file"
    exit 1
fi
echo "Generated $expected_file"

echo "Baseline data array generated successfully!"