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
├── mandelbrot/                 # Core modular package
│   ├── __init__.py
│   ├── config.py              # Configuration and argument parsing
│   ├── computation.py         # Mandelbrot computation algorithms
│   ├── runtime.py             # Main execution logic
│   ├── communication/         # MPI communication modules
│   │   ├── __init__.py
│   │   ├── blocking.py        # Blocking communication
│   │   └── nonblocking.py     # Non-blocking communication
│   └── scheduling/            # Work scheduling modules
│       ├── __init__.py
│       ├── static.py          # Static scheduling
│       └── dynamic.py         # Dynamic scheduling
├── tests/                     # Test suite
│   ├── baseline_data/         # Reference data arrays
│   ├── Mandelbrot.py          # Original baseline script
│   ├── generate_baseline.sh   # Baseline generation
│   └── test_runner.sh         # Universal test runner (tests all available sizes)
├── legacy/                    # Non-modular scripts
├── Assignment_description/    # Project assignment documents
├── main.py                    # Main entry point
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
make help

# Generate baseline data for testing
make baseline

# Run individual tests
make test-static-blocking      # Default: 1 MPI process
make test-dynamic-blocking N=4 # 4 MPI processes

# Run all tests with custom process count
make test-all N=2              # All combinations with 2 MPI processes

# Clean generated files
make clean
```

### Testing with Different Process Counts

```bash
# Test with 1 MPI process (default)
make test-static-blocking

# Test with 4 MPI processes
make test-static-blocking N=4

# Run all tests with 2 MPI processes
make test-all N=2
```

### Individual Test Targets

```bash
# Test specific scheduling/communication combinations
make test-static-blocking      # Static scheduling + blocking communication
make test-static-nonblocking   # Static scheduling + non-blocking communication
make test-dynamic-blocking     # Dynamic scheduling + blocking communication
make test-dynamic-nonblocking  # Dynamic scheduling + non-blocking communication

# All tests with custom process count
make test-all N=4              # All combinations with 4 MPI processes
```

### Direct Script Usage

```bash
# Run the modular implementation directly (no output by default)
mpirun -n 4 python main.py 10 200x150 --schedule dynamic --communication blocking

# Save computed data array
mpirun -n 4 python main.py 10 200x150 --schedule dynamic --communication blocking --save-data

# Generate plots
mpirun -n 2 python main.py 10 200x150 --schedule static --communication nonblocking --output plot.png

# Display plot interactively
mpirun -n 2 python main.py 10 200x150 --schedule static --communication nonblocking --show

# Run the test runner directly
./tests/test_runner.sh 4 dynamic nonblocking

# Run the original baseline script
python tests/Mandelbrot.py 10 200x150
```

## Configuration Options

### Command Line Arguments

- `chunk_size`: Number of rows per chunk (default: 10)
- `size`: Image dimensions as WIDTHxHEIGHT (default: 200x150)
- `--schedule`: Scheduling strategy (`static` or `dynamic`)
- `--communication`: Communication pattern (`blocking` or `nonblocking`)
- `--output`: Output file for plots (optional)
- `--show`: Display plot interactively (optional)
- `--save-data`: Save computed data array as .npy file (optional)

### Example Configurations

```bash
# Small image, static scheduling (no output)
mpirun -n 2 python main.py 5 100x100 --schedule static --communication blocking

# Large image, dynamic scheduling with data saving
mpirun -n 8 python main.py 20 500x400 --schedule dynamic --communication nonblocking --save-data

# Generate plot
mpirun -n 4 python main.py 10 300x200 --schedule static --communication blocking --output mandelbrot.png

# Display plot interactively
mpirun -n 2 python main.py 10 200x150 --schedule dynamic --communication blocking --show

# Test with different baselines (generates baselines first)
make baseline SIZE=100x100
make baseline SIZE=500x400
make test-static-blocking  # Tests against all available baselines
```

## Testing

The project includes comprehensive testing that validates numerical accuracy across all scheduling and communication combinations.

### Test System

The testing system uses a single universal test runner (`tests/test_runner.sh`) that:
- Takes 3 parameters: `N` (MPI processes), `SCHEDULE` (static/dynamic), `COMM` (blocking/nonblocking)
- Automatically discovers all available baseline sizes in `tests/baseline_data/`
- Tests each size with the specified scheduling/communication combination
- Compares results numerically with baseline data
- Provides clear visual feedback with status messages

### Available Test Targets

```bash
# Individual test combinations
make test-static-blocking      # Static + blocking (N=1 default)
make test-static-nonblocking   # Static + nonblocking (N=1 default)
make test-dynamic-blocking     # Dynamic + blocking (N=1 default)
make test-dynamic-nonblocking  # Dynamic + nonblocking (N=1 default)

# All combinations
make test-all N=2              # All 4 combinations with 2 processes

# Direct test runner usage
./tests/test_runner.sh 4 dynamic nonblocking
```

### Test Process

1. **Baseline Generation**: Creates reference data using the original script
2. **Modular Testing**: Runs modular implementation with different configurations
3. **Numerical Comparison**: Compares arrays using `numpy.array_equal()`
4. **Validation**: Ensures identical results regardless of process count or scheduling strategy

### Test Results

All tests verify that:
- ✅ Results are numerically identical to baseline
- ✅ Different process counts produce same results
- ✅ All scheduling/communication combinations work correctly
- ✅ Worker load distribution is visible in dynamic scheduling
- ✅ Clear visual feedback with status messages

## Performance Considerations

### Process Count Guidelines

- **Small images (≤200x150)**: 1-4 processes
- **Medium images (200x150-500x400)**: 2-8 processes  
- **Large images (>500x400)**: 4-16+ processes

### Memory Usage

- Each process loads the full image array
- Consider available RAM when choosing process count
- Use `make clean` to remove generated files

## Troubleshooting

### Common Issues

**MPI not found:**
```bash
# Install MPI (macOS with Homebrew)
brew install open-mpi

# Install MPI (Ubuntu/Debian)
sudo apt-get install libopenmpi-dev openmpi-bin

# Install MPI (CentOS/RHEL)
sudo yum install openmpi-devel
```

**Python module not found:**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

**Permission denied on scripts:**
```bash
# Make scripts executable
chmod +x tests/*.sh
```

**Memory issues with large images:**
```bash
# Use fewer processes
make test-all N=2

# Or use smaller image size
mpirun -n 4 python main.py 10 100x100 --schedule dynamic --communication blocking
```

## Development

### Adding New Features

1. **New Scheduling Strategy**: Add to `mandelbrot/scheduling/`
2. **New Communication Pattern**: Add to `mandelbrot/communication/`
3. **New Test**: Create script in `tests/` and add Makefile target

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
make baseline                    # Generate baseline data
make help                       # Show all available commands

# Testing
make test-static-blocking       # Quick test (1 process)
make test-all N=4               # All tests with 4 processes
make clean                      # Clean up generated files

# Direct usage
mpirun -n 2 python main.py 10 200x150 --schedule dynamic --communication blocking
./tests/test_runner.sh 2 dynamic nonblocking
```

### Test Matrix

| Target | Scheduling | Communication | Default N |
|--------|------------|---------------|-----------|
| `test-static-blocking` | Static | Blocking | 1 |
| `test-static-nonblocking` | Static | Non-blocking | 1 |
| `test-dynamic-blocking` | Dynamic | Blocking | 1 |
| `test-dynamic-nonblocking` | Dynamic | Non-blocking | 1 |

## Acknowledgments

- DTU Large Scale Modelling course
- MPI4Py library for Python MPI bindings
- NumPy for numerical computations
- Matplotlib for visualization