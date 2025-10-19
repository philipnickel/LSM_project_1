#!/usr/bin/env python3
"""Generate LSF job script for a suite."""

import argparse
import os
import subprocess
from pathlib import Path

import yaml

from mandelbrot.config import load_named_sweep_configs

parser = argparse.ArgumentParser()
parser.add_argument("--suite", required=True, help="Suite name (TESTS, chunks, etc.)")
args = parser.parse_args()

# Load suite config
with open("configs/sweeps.yaml") as f:
    suites = {exp["name"]: exp for exp in yaml.safe_load(f)["experiments"]}

suite = suites[args.suite]
res = suite.get("resources", {})

# Count configs
n_configs = len(load_named_sweep_configs("configs/sweeps.yaml", args.suite)[0][1])

# Set environment variables for template substitution
env = os.environ.copy()
span_value = res.get("span")
env.update(
    {
        "JOB_NAME": args.suite,
        "ARRAY_RANGE": f"1-{n_configs}",
        "QUEUE": res.get("queue", "hpcintro"),
        "N_CORES": str(res.get("n_cores", 4)),
        "WALLTIME": res.get("walltime", "00:30"),
        "MEM_PER_CORE": res.get("mem_per_core", "2GB"),
        "SPAN_DIRECTIVE": f'#BSUB -R "span[{span_value}]"'
        if span_value
        else '#BSUB -R "span[ptile=10]"',
        "SUITE": args.suite,
    }
)

# Render template with envsubst
with open("job_scripts/job_template.sh", "rb") as f:
    result = subprocess.run(
        ["envsubst"], input=f.read(), stdout=subprocess.PIPE, check=True, env=env
    )

# Save to job_scripts/generated/
output_dir = Path("job_scripts/generated")
output_dir.mkdir(exist_ok=True)
output_file = output_dir / f"{args.suite}.sh"

with open(output_file, "wb") as f:
    f.write(result.stdout)

print(f"Generated: {output_file} ({n_configs} configs)")
print(f"\nTo submit: bsub < {output_file}")
