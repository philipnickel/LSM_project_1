"""End-to-end test via main.py."""
import subprocess
import os


def test_tests_suite():
    """Run TESTS suite end-to-end - should complete without errors."""
    result = subprocess.run([
        "python", "main.py",
        "--sweep", "configs/sweeps.yaml",
        "--suite", "TESTS",
    ], env={**os.environ, "SKIP_MLFLOW": "1"},
       capture_output=True, text=True, timeout=30)
    
    # Show output if failed
    assert result.returncode == 0, f"Suite failed:\n{result.stdout}\n{result.stderr}"

