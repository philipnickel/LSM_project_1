# Mandelbrot MPI

High-level workflow for the Mandelbrot MPI experiments with MLflow tracking.

## Workflow

- **Install UV**: `curl -LsSf https://astral.sh/uv/install.sh | sh` 

- **Install deps**: `uv sync`
- **IMPORTANT** run `module load mpi/5.0.8-gcc-13.4.0-binutils-2.44` before syncing 

- **Configure sweeps**: edit `configs/sweeps.yaml` (suites + resources).

- **Auth MLflow**: `uv run python setup.py` to cache Databricks creds
use host: https://dbc-6756e917-e5fc.cloud.databricks.com/ml/experiments/3399934008965459?o=2967813328041853
token: (find on Databricks under settings->developer->Access Token and generate new one)

- **Run tests**: `uv run pytest` to make sure everything works

- **Run locally**: `uv run python main.py --n-ranks 4 --chunk-size 20` or `uv run python main.py --sweep configs/sweeps.yaml --suite TESTS`.

- **Generate HPC job script**: `uv run python job_scripts/generate_jobscript.py --suite TESTS`, inspect `job_scripts/generated/TESTS.sh`, submit with `bsub < job_scripts/generated/TESTS.sh`.

- Or just use the existing ones (only need to rerun if changing resources in configs)

