#!/bin/bash
# Generate baseline data array for testing

set -e  # Exit on any error

# Get size parameter (default: 200x150)
SIZE=${1:-200x150}
XLIM=${2:--2.2:0.75}    # xlim
YLIM=${3:--1.3:1.3}     # ylim

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "Generating baseline data array..."

echo "Generating baseline for size: $SIZE"
cd tests/baseline_data
python ../Mandelbrot.py 10 $SIZE $XLIM $YLIM
cd ../..

# Check if file was created
# XLIM, YLIM Input must be f.1 float number: x.x
# For integers, add .0 for example: 1 -> 1.0
xlim_encoded="${XLIM//:/_}"
ylim_encoded="${YLIM//:/_}"
expected_file="tests/baseline_data/baseline_${SIZE}_${xlim_encoded}_${ylim_encoded}.npy"
if [ ! -f "$expected_file" ]; then
    echo "ERROR: Baseline file not created: $expected_file"
    exit 1
fi
echo "Generated $expected_file"

echo "Baseline data array generated successfully!"