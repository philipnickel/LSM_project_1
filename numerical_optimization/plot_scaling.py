# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scienceplots

sns.set_style("whitegrid")
plt.style.use("science")

# %%
df = pd.read_csv("numerical_optimization/bench_results.csv")
df["Problem Size (Pixels)"] = df["Image Size"].str.split("x").apply(lambda x: int(x[0]) * int(x[1]))

# Sort before plotting (important for lines)
df = df.sort_values("Problem Size (Pixels)")
df.columns

# %%
df

# %%
fig = plt.figure(figsize=(12, 8))
fig = sns.lineplot(data=df, x="Problem Size (Pixels)", y="Time (s)", hue="Implementation", markers=True)
fig.set(yscale="log")
fig.set(xscale="log")

plt.savefig("Plots/Numba_vs_baseline.pdf")


# %% Plot

fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(
    data=df,
    x="Problem Size (Pixels)",
    y="Time (s)",
    hue="Implementation",
    markers=True,
    dashes=False,
    ax=ax,
)

# log–log axes
ax.set(xscale="log", yscale="log")
ax.set_xlabel(r"\textbf{Problem Size} ($N_\mathrm{pixels}$)")
ax.set_ylabel(r"\textbf{Wall Time} (s)")
ax.set_title(r"\textbf{Scaling of Mandelbrot Implementations}")

# --- Simple reference line (O(N) scaling) ---
x_ref = np.array([df["Problem Size (Pixels)"].min(), df["Problem Size (Pixels)"].max()])
y_ref = df["Time (s)"].mean() * (x_ref / x_ref[0]) ** 1  # linear reference
ax.plot(x_ref, y_ref, "--", color="gray", label=r"$O(N)$")

# Legend and save
ax.legend(title=r"\textbf{Implementation}", loc="best")
plt.tight_layout()
plt.savefig("Plots/Numba_vs_baseline.pdf", dpi=300)
