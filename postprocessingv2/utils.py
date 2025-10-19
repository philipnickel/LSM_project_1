from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
import seaborn as sns

PLOTS_V2_DIR = Path("plots_v2")


def ensure_style() -> None:
    """Apply consistent plotting style."""
    plt.style.use("science")
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
    sns.set_context("talk")


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
