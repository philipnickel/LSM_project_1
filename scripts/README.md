# Analysis Scripts

This directory contains the analysis notebook for the Mandelbrot MPI experiments.

## Files

### `results_analysis.qmd` 
**Comprehensive performance analysis notebook** using Quarto for reproducible research.

**Features:**
- Self-contained analysis (no external dependencies)
- Multi-dimensional performance analysis
- Interactive visualizations
- Key findings and recommendations
- Publication-ready output

**Usage:**
```bash
# Render to HTML
quarto render results_analysis.qmd

# Render to PDF  
quarto render results_analysis.qmd --to pdf

# Live preview
quarto preview results_analysis.qmd
```

## Workflow

### 1. Run Experiments
```bash
# Example experiments
mpirun -n 4 python main.py 50 500x500 --schedule dynamic --communication blocking --log-experiment
mpirun -n 8 python main.py 50 500x500 --schedule static --communication nonblocking --log-experiment
```

### 2. Analyze Results
```bash
cd scripts

# Interactive analysis
quarto preview results_analysis.qmd

# Generate report
quarto render results_analysis.qmd
```

## Data Flow

### Raw Experiment Data
- `../experiments/*/experiment_*.csv` - Individual experiment files (18 columns)
- Contains essential runtime metrics only (optimized for performance)

### Analysis Output
- `results_analysis.html` - Interactive HTML report
- `results_analysis.pdf` - Publication-ready PDF report
- All derived metrics calculated in-memory during analysis

## Key Metrics

### Essential Metrics (logged during experiments)
- **Timing**: `wall_clock_time`, `computation_time`, `communication_time`
- **Configuration**: `experiment_id`, `schedule`, `communication`, `num_processes`
- **Work Distribution**: `rank`, `chunks_processed`, `chunk_ids`
- **Problem Parameters**: `image_width`, `image_height`, `chunk_size`

### Derived Metrics (calculated during analysis)
- **Performance**: `parallel_efficiency`, `communication_overhead`, `speedup`
- **Load Balancing**: `load_balance_std`, `load_balance_range`
- **Scaling**: `time_per_chunk`, `total_efficiency`

## Analysis Sections

The updated analysis provides comprehensive performance evaluation structured as:

1. **Communication vs Computation Analysis** - Fundamental trade-off measurements
2. **Domain Decomposition** - Chunk size effects on performance and load balancing  
3. **Load Balancing Analysis** - Work distribution quality across scheduling strategies
4. **Performance Scaling** - Wall time, parallel efficiency, and speedup analysis
5. **Problem Size Scaling** - How performance scales with image dimensions
6. **Performance Summary** - Key findings and recommendations

## Key Features

- **Individual plots** for clear, focused analysis (no subplots)
- **Scienceplots styling** for publication-quality figures
- **Communication overhead measurements** - Detailed breakdown of computation vs communication time
- **Load balancing metrics** - Quantitative analysis of work distribution
- **Research-structured analysis** following experimental methodology best practices

## Dependencies

```bash
# Required Python packages
pip install pandas numpy matplotlib seaborn

# For Quarto notebook
# Install Quarto from https://quarto.org/docs/get-started/
```

## Features

- **✅ Self-contained**: No external processing scripts required
- **✅ Fast**: Processes data in-memory during analysis  
- **✅ Flexible**: Easy to modify analysis logic
- **✅ Transparent**: All calculations visible in notebook
- **✅ Professional**: Publication-ready visualizations and reports