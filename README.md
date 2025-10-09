# Mandelbrot MPI Project


## Project Structure

```
├── main.py                    # Main entry point
├── Mandelbrot/                   # Modular helper scripts
│   ├── Communication.py            # Helper for Communication 
│   ├── scheduling.py        # Helper for scheduling
│   ├── computation.py        # Helper for Mandelbrot computations
├── tests/                     # Tests 
│   ├── baseline_data/         # Reference data arrays
│   ├── Mandelbrot.py          # Original baseline script
│   ├── generate_baseline.sh   # Baseline generation
│   └── test_runner.sh         # Universal test runner (tests all available sizes)
├── Assignment_description/    # Project assignment documents
├── Makefile                   # Build and test automation
└── requirements.txt           # Python dependencies
```
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
## Usage

### Basic Commands

```bash
# Show help and available targets
python main.py --help
```

### Direct Script Usage

The `main.py` script is the entry point and allows users to select between blocking/non-blocking and static/dynamic scheduling. It delegates specific operations to the modular helper scripts.

### Command-Line Arguments
### Example Configurations

```bash
mpirun -n $N python main.py $CHUNK_SIZE $size --schedule $SCHEDULE --communication $COMMUNICATION --save-data 

```
### Makefile

# Clean up
make clean
```
