# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
plt.style.use("science")

# %%
df = pd.read_csv("numerical_optimization/bench_results.csv")
df.columns


# %%
fig = plt.figure(figsize=(12, 8))
fig = sns.lineplot(data=df, x="Image Size", y="Time (s)", hue="Implementation", markers="o")
fig.set(yscale="log")
fig.set(xscale="log")

plt.savefig("Plots/Numba_vs_baseline.pdf")
