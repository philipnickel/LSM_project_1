#!/usr/bin/env python3
"""Script to save Mandelbrot computation results as numpy arrays for comparison."""

import sys
import argparse
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mandelbrot.computation import compute_full_image
from mandelbrot.config import RunConfig
from Mandelbrot import generate_image as baseline_generate_image


def save_baseline_results(size, xlim, ylim, max_iter, output_file):
    """Save baseline computation results."""
    result = baseline_generate_image(size, xlim, ylim, max_iter)
    np.save(output_file, result)
    return result


def save_modular_results(size, xlim, ylim, max_iter, chunk_size, output_file):
    """Save modular computation results."""
    config = RunConfig(
        image_size=size,
        xlim=xlim,
        ylim=ylim,
        max_iter=max_iter,
        chunk_size=chunk_size,
        schedule="static",
        communication="blocking",
        output=None,
        show_plot=False
    )
    result = compute_full_image(config)
    np.save(output_file, result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Save Mandelbrot computation results")
    parser.add_argument("--size", type=str, default="50x50", help="Image size WIDTHxHEIGHT")
    parser.add_argument("--xlim", type=str, default="-2.2|0.75", help="Real axis limits MIN|MAX")
    parser.add_argument("--ylim", type=str, default="-1.3|1.3", help="Imag axis limits MIN|MAX")
    parser.add_argument("--max-iter", type=int, default=20, help="Maximum iterations")
    parser.add_argument("--chunk-size", type=int, default=10, help="Chunk size for modular computation")
    parser.add_argument("--baseline-output", type=str, required=True, help="Output file for baseline results")
    parser.add_argument("--modular-output", type=str, required=True, help="Output file for modular results")
    
    args = parser.parse_args()
    
    # Parse size
    width, height = args.size.lower().split("x", 1)
    size = (int(width), int(height))
    
    # Parse limits
    xlo, xhi = args.xlim.split("|", 1)
    xlim = (float(xlo), float(xhi))
    
    ylo, yhi = args.ylim.split("|", 1)
    ylim = (float(ylo), float(yhi))
    
    # Compute and save results
    print(f"Computing baseline results: {size}, {xlim}, {ylim}, max_iter={args.max_iter}")
    baseline_result = save_baseline_results(size, xlim, ylim, args.max_iter, args.baseline_output)
    
    print(f"Computing modular results: {size}, {xlim}, {ylim}, max_iter={args.max_iter}, chunk_size={args.chunk_size}")
    modular_result = save_modular_results(size, xlim, ylim, args.max_iter, args.chunk_size, args.modular_output)
    
    # Compare results
    if np.array_equal(baseline_result, modular_result):
        print("✓ Results are identical!")
        return 0
    else:
        print("✗ Results differ!")
        diff = np.abs(baseline_result - modular_result)
        print(f"  Max difference: {np.max(diff)}")
        print(f"  Mean difference: {np.mean(diff)}")
        print(f"  Non-zero differences: {np.count_nonzero(diff)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())