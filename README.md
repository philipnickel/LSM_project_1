# Mandelbrot MPI Project

A modular implementation of the Mandelbrot set computation using MPI (Message Passing Interface) for parallel processing. This project demonstrates different scheduling strategies (static/dynamic) combined with communication patterns (blocking/non-blocking) for distributed computing.

## Features

- **Modular Architecture**: Clean separation of computation, scheduling, and communication
- **Multiple Scheduling Strategies**: Static and dynamic work distribution
- **Communication Patterns**: Blocking and non-blocking MPI communication
- **Comprehensive Testing**: Automated testing with numerical validation
- **Flexible Process Count**: Test with any number of MPI processes
- **Baseline Comparison**: Results validated against reference implementation
- **Interactive Analysis**: Jupyter notebook for detailed performance analysis
- **Experiment Logging**: CSV-based experiment tracking with detailed metrics
- **Clean Default Behavior**: No files saved by default - only when explicitly requested

## Project Structure

```
├── main.py                    # Main entry point
├── helpers/                   # Modular helper scripts
│   ├── blocking.py            # Helper for blocking operations
│   ├── non_blocking.py        # Helper for non-blocking operations
├── tests/                     # Test suite
│   ├── baseline_data/         # Reference data arrays
│   ├── Mandelbrot.py          # Original baseline script
│   ├── generate_baseline.sh   # Baseline generation
│   └── test_runner.sh         # Universal test runner (tests all available sizes)
├── Assignment_description/    # Project assignment documents
├── Makefile                   # Build and test automation
└── requirements.txt           # Python dependencies
```

## Prerequisites

- Python 3.7+
- MPI implementation (OpenMPI, MPICH, or Intel MPI)
- Virtual environment support

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd LSM_project_1
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### 4. Verify MPI Installation

```bash
# Check if MPI is available
mpirun --version
```

## Usage

### Basic Commands

```bash
# Show help and available targets
python main.py --help
```

### Direct Script Usage

The `main.py` script is the entry point and allows users to select between blocking/non-blocking and static/dynamic scheduling. It delegates specific operations to the modular helper scripts.

Example usage:
```bash
# Blocking and static scheduling
mpirun -n 4 python main.py blocking static 10 1000x1000 -2.2:0.75 -1.3:1.3

# Non-blocking and dynamic scheduling
mpirun -n 4 python main.py non_blocking dynamic 10 1000x1000 -2.2:0.75 -1.3:1.3
```

### Command-Line Arguments

- `type`: Execution type (`blocking` or `non_blocking`)
- `scheduling`: Scheduling strategy (`static` or `dynamic`)
- `chunk_size`: Number of rows per chunk
- `image_size`: Image dimensions as `WIDTHxHEIGHT`
- `xlim`: X-axis limits as `X_MIN:X_MAX`
- `ylim`: Y-axis limits as `Y_MIN:Y_MAX`

### Example Configurations

```bash
# Small image, static scheduling (blocking)
mpirun -n 2 python main.py blocking static 10 100x100 -2.0:1.0 -1.5:1.5

# Large image, dynamic scheduling (non-blocking)
mpirun -n 8 python main.py non_blocking dynamic 20 500x400 -2.5:1.5 -2.0:2.0

# Generate plot
mpirun -n 4 python main.py blocking static 10 300x200 -2.2:0.75 -1.3:1.3
```

## Development

### Adding New Features

1. **New Scheduling Strategy**: Add to `helpers/`
2. **New Communication Pattern**: Extend `helpers/`
3. **New Test**: Create script in `tests/` and add test runner target

### Code Style

- Follow PEP 8 Python style guidelines
- Use type hints for function signatures
- Add docstrings for all public functions
- Maintain consistent naming conventions

## License

This project is part of the Large Scale Modelling course at DTU.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Quick Reference

### Most Common Commands

```bash
# Setup
python main.py blocking static 10 200x150 -2.0:1.0 -1.5:1.5

# Testing
mpirun -n 4 python main.py non_blocking dynamic 10 300x200 -2.0:2.0 -1.5:1.5

# Clean up
make clean
```
