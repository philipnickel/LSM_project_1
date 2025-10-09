# Mandelbrot MPI Metrics Guide

## Overview

We use a **streamlined metrics logging system** that captures only essential runtime data during experiments, then calculates all derived metrics during analysis for optimal performance and maximum flexibility.

## 📊 Core Logged Metrics (18 columns)

These metrics are captured during experiment execution and saved to CSV files:

### Experiment Configuration (10 columns)
```
experiment_id        - Unique identifier (e.g., "dynamic_blocking_500x500_abc123")
timestamp           - Unix timestamp when experiment started
schedule            - Scheduling strategy: "static" or "dynamic" 
communication       - Communication pattern: "blocking" or "nonblocking"
num_processes       - Number of MPI processes used
chunk_size          - Size of work units (rows per chunk)
image_width         - Image width in pixels
image_height        - Image height in pixels  
xlim_min, xlim_max  - Complex plane real axis bounds
ylim_min, ylim_max  - Complex plane imaginary axis bounds
```

### Performance Data (6 columns)
```
wall_clock_time     - Total execution time (seconds)
computation_time    - Time spent in Mandelbrot computation (per rank)
communication_time  - Time spent in MPI communication (per rank)
rank                - MPI process rank (0, 1, 2, ...)
chunks_processed    - Number of chunks processed by this rank
chunk_ids           - Comma-separated list of chunk IDs (for debugging)
```

### File Structure
Each experiment creates **one row per MPI rank**, so an N-process experiment has N rows with the same `experiment_id`.

## 🔄 Derived Metrics (Calculated During Analysis)

These metrics are calculated in-memory during analysis using `scripts/results_analysis.qmd`:

### Basic Derived (7 metrics)
```
total_chunks           - ceil(image_width / chunk_size)
image_size            - image_width * image_height  
image_size_display    - "500×500" format for visualization
computation_efficiency    - computation_time / wall_clock_time
communication_overhead   - communication_time / wall_clock_time
total_efficiency        - (comp_time + comm_time) / wall_clock_time
time_per_chunk         - computation_time / chunks_processed
```

### Experiment Summary (per-experiment aggregation)
```
parallel_efficiency      - total_computation_time / (wall_time × num_processes)
communication_overhead   - total_communication_time / total_computation_time
load_balance_std         - Standard deviation of chunks per worker
load_balance_range       - Max - min chunks per worker
```

## 🚀 Benefits of This Approach

### Runtime Performance
- **Faster experiments**: No complex calculations during MPI execution
- **Smaller CSV files**: 18 columns (essential data only)
- **Less memory usage**: Minimal data structures during parallel execution

### Analysis Flexibility  
- **Recalculate metrics**: Change formulas by modifying analysis notebook
- **Add new metrics**: Extend analysis without touching core experiment code
- **Full transparency**: All calculations visible and modifiable in notebook
- **No dependencies**: Self-contained analysis without external scripts

### Maintenance
- **Simpler architecture**: No separate post-processing scripts to maintain
- **Easier debugging**: Raw timing data preserved, analysis logic transparent
- **Version control friendly**: Analysis changes tracked in notebook

## 📁 Usage

### Running Experiments
```bash
# Experiments automatically log core metrics (18 columns)
mpirun -n 4 python main.py 50 500x500 --schedule dynamic --log-experiment
```

### Analysis
```bash
# All derived metrics calculated during analysis
cd scripts
quarto render results_analysis.qmd

# Interactive analysis
quarto preview results_analysis.qmd
```

## 🔍 Key Metric Definitions

### Parallel Efficiency  
```
efficiency = total_computation_time / (wall_clock_time * num_processes)
```
- 1.0 = Perfect scaling (no communication overhead, perfect load balance)
- 0.5 = 50% efficiency (half the potential parallelism achieved)

### Communication Overhead
```
overhead = total_communication_time / total_computation_time
```
- 0.0 = No communication cost
- 1.0 = Communication time equals computation time
- >1.0 = Communication dominates computation

### Load Balance Quality
```
quality = 1.0 - (std_dev_chunks / mean_chunks)
```
- 1.0 = Perfect balance (all workers have same chunks)
- Lower values = worse load balancing

## 📊 Data Flow

```
Experiments → Raw CSV (18 cols) → Analysis Notebook → Reports
    ↓              ↓                    ↓              ↓
  MPI runs    Essential data      Derived metrics   HTML/PDF
```

**Files:**
- Raw: `../experiments/*/experiment_*.csv` (18 columns)
- Output: `results_analysis.html`, `results_analysis.pdf`

This approach provides **maximum performance during experiments** and **maximum flexibility during analysis**!