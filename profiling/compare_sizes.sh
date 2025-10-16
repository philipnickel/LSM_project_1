#!/bin/bash

set -e

echo "Timing 1000×1000 Mandelbrot Set"
echo "================================"
echo ""

echo "[1/2] Reference implementation..."
time uv run python profiling/Mandelbrot.py 10 1000x1000 -2.2:0.75 -1.3:1.3

echo ""
echo "[2/2] MPI implementation (with MLflow)..."
time uv run python main.py --n-ranks=1 --chunk-size=5 --schedule=static --communication=blocking --image-size=1000x1000 --xlim=-2.2:0.75 --ylim=-1.3:1.3

echo ""
echo "✓ Complete!" 
