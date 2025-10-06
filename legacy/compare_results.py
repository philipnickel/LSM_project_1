#!/usr/bin/env python3
"""Script to compare saved Mandelbrot computation results."""

import sys
import argparse
import numpy as np
from pathlib import Path


def compare_results(file1, file2, tolerance=1e-10):
    """Compare two saved result files."""
    try:
        result1 = np.load(file1)
        result2 = np.load(file2)
    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        return False
    
    print(f"Comparing {file1} vs {file2}")
    print(f"  Shape 1: {result1.shape}, Shape 2: {result2.shape}")
    
    if result1.shape != result2.shape:
        print("✗ Shapes differ!")
        return False
    
    # Check if arrays are identical
    if np.array_equal(result1, result2):
        print("✓ Results are identical!")
        return True
    
    # Check if arrays are close within tolerance
    if np.allclose(result1, result2, atol=tolerance):
        print(f"✓ Results are close within tolerance {tolerance}")
        return True
    
    # Analyze differences
    diff = np.abs(result1 - result2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    nonzero_diff = np.count_nonzero(diff)
    
    print(f"✗ Results differ significantly:")
    print(f"  Max difference: {max_diff}")
    print(f"  Mean difference: {mean_diff}")
    print(f"  Non-zero differences: {nonzero_diff} out of {result1.size}")
    print(f"  Relative max difference: {max_diff / np.max(result1) if np.max(result1) > 0 else 'N/A'}")
    
    return False


def main():
    parser = argparse.ArgumentParser(description="Compare Mandelbrot computation results")
    parser.add_argument("file1", help="First result file (.npy)")
    parser.add_argument("file2", help="Second result file (.npy)")
    parser.add_argument("--tolerance", type=float, default=1e-10, help="Tolerance for comparison")
    parser.add_argument("--verbose", action="store_true", help="Show detailed comparison")
    
    args = parser.parse_args()
    
    if not Path(args.file1).exists():
        print(f"Error: File {args.file1} does not exist")
        return 1
    
    if not Path(args.file2).exists():
        print(f"Error: File {args.file2} does not exist")
        return 1
    
    success = compare_results(args.file1, args.file2, args.tolerance)
    
    if args.verbose and not success:
        # Load and show some sample differences
        result1 = np.load(args.file1)
        result2 = np.load(args.file2)
        diff = np.abs(result1 - result2)
        
        # Find locations with largest differences
        max_indices = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"\nLargest difference at position {max_indices}:")
        print(f"  Value 1: {result1[max_indices]}")
        print(f"  Value 2: {result2[max_indices]}")
        print(f"  Difference: {diff[max_indices]}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())