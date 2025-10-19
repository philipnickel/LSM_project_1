from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from postprocessing.utils import (
    PLOTS_DIR,
    config_palette,
    ensure_config_level,
    ensure_output_dir,
    ensure_style,
    load_suite_runs,
)

SUITE = "scaling_im"


def prepare_runs() -> pd.DataFrame:
    runs_idx = ensure_config_level(load_suite_runs(SUITE, as_index=True))
    runs = runs_idx.reset_index()
    runs["Pixels"] = runs["Image Size"].str.lower().str.split("x").apply(
        lambda parts: int(parts[0]) * int(parts[1])
    )
    runs = runs.sort_values(["Pixels", "Config"])
    runs["Time per Pixel"] = runs["Wall Time(s)"] / runs["Pixels"]
    return runs


def plot_wall_time(runs: pd.DataFrame, out_dir: Path) -> None:
    palette = config_palette(runs["Config"].unique())
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.lineplot(
        data=runs,
        x="Pixels",
        y="Wall Time(s)",
        hue="Config",
        style="Config",
        markers=True,
        dashes=False,
        ax=ax,
        palette=palette,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Pixels per image")
    ax.set_ylabel("Wall time [s]")
    ax.set_title("Wall Time vs Problem Size")
    fig.tight_layout()
    fig.savefig(out_dir / "4.1_wall_time_vs_pixels.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_style()
    runs = prepare_runs()
    out_dir = ensure_output_dir(PLOTS_DIR / "4_scaling_problem")
    plot_wall_time(runs, out_dir)


if __name__ == "__main__":
    main()
