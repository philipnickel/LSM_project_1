"""Test numerical correctness of MPI implementation via subprocess."""

import subprocess
import tempfile
from pathlib import Path
import numpy as np
import pytest
from mandelbrot.baseline import compute_mandelbrot
from mandelbrot.config import load_sweep_configs

TEST_CONFIGS = load_sweep_configs("tests/test_configs.yaml")


@pytest.mark.parametrize("config", TEST_CONFIGS, ids=lambda c: c.run_name)
def test_mpi_matches_baseline(config):
    """Run MPI via subprocess and compare to baseline."""

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "result.npy"
        config_idx = TEST_CONFIGS.index(config)

        # Always run with mpirun
        cmd = ["mpirun", "-n", str(config.n_ranks), "python",
               "tests/run_mpi_and_save.py", "tests/test_configs.yaml",
               str(config_idx), str(output)]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, f"MPI failed: {result.stderr}"

        # Compare to baseline
        mpi_image = np.load(output)
        baseline = compute_mandelbrot((config.width, config.height),config.xlim,config.ylim)

        np.testing.assert_allclose(mpi_image, baseline, rtol=1e-10,
                                   err_msg=f"Mismatch: {config.run_name}")
