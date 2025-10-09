import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

result_dir = Path("results")
files = sorted(result_dir.glob("timings_*.csv"))

for file in files:
    df = pd.read_csv(file)
    ranks = df["ranks"].iloc[0]

    # Filter for blocking runs only
    df = df[df["communication"] == "blocking"]

    # Compute mean computation and communication time per chunk size & schedule
    summary = (
        df.groupby(["chunk_size", "schedule"])
        [["computation_time", "communication_time"]]
        .mean()
        .reset_index()
    )

    chunk_sizes = sorted(summary["chunk_size"].unique())
    schedules = ["static", "dynamic"]
    x = np.arange(len(chunk_sizes))
    width = 0.35  # width of the bars

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {
        "static": ("#4CAF50", "#81C784"),  # dark/light green
        "dynamic": ("#2196F3", "#64B5F6"),  # dark/light blue
    }

    for i, sched in enumerate(schedules):
        subset = summary[summary["schedule"] == sched]
        comp = subset["computation_time"].values
        comm = subset["communication_time"].values

        ax.bar(
            x + (i - 0.5) * width,
            comp,
            width,
            label=f"{sched.capitalize()} (comp)",
            color=colors[sched][0],
        )
        ax.bar(
            x + (i - 0.5) * width,
            comm,
            width,
            bottom=comp,
            label=f"{sched.capitalize()} (comm)",
            color=colors[sched][1],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(chunk_sizes)
    ax.set_xlabel("Chunk size")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"Blocking Communication - Computation vs Communication (N={ranks})")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(result_dir / f"barchart_blocking_{ranks}.png", dpi=150)
    plt.close()