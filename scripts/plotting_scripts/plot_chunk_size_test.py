import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

result_dir = Path("results")
files = sorted(result_dir.glob("timings_*.csv"))

for file in files:
    df = pd.read_csv(file)
    ranks = df["ranks"].iloc[0]

    plt.figure(figsize=(8, 5))
    for (sched, comm), group in df.groupby(["schedule", "communication"]):
        plt.loglog(
            group["chunk_size"],
            group["run_time"],
            marker="o",
            label=f"{sched}-{comm} (run)"
        )

    plt.xlabel("Chunk size")
    plt.ylabel("Time (s)")
    plt.title(f"Runtime vs Chunk Size (N={ranks})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(result_dir / f"timing_plot_{ranks}.png")
    plt.close()
