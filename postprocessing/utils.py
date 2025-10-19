from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import scienceplots  # noqa: F401
import seaborn as sns

CONFIG_ORDER: Tuple[Tuple[str, str], ...] = (
    ("static", "blocking"),
    ("static", "nonblocking"),
    ("dynamic", "blocking"),
    ("dynamic", "nonblocking"),
)
CONFIG_COLORS = {
    ("static", "blocking"): "#1f77b4",
    ("static", "nonblocking"): "#ff7f0e",
    ("dynamic", "blocking"): "#2ca02c",
    ("dynamic", "nonblocking"): "#d62728",
}

PLOTS_DIR = Path("Plots")
CACHE_DIR = Path("_exp_mlcache")
RUNS_IDX = CACHE_DIR / "runs_indexed.parquet"
RANKS_IDX = CACHE_DIR / "ranks_indexed.parquet"
CHUNKS_IDX = CACHE_DIR / "chunks_indexed.parquet"


def ensure_style() -> None:
    plt.style.use("science")
    sns.set_style("whitegrid")
    sns.set_context("talk")


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_label(schedule: str, communication: str) -> str:
    return f"{schedule.title()} / {communication}"


def config_palette(configs: Iterable[str]) -> dict[str, str]:
    palette: dict[str, str] = {}
    for label in configs:
        parts = [p.strip().lower() for p in label.split("/")]
        key = (parts[0], parts[1]) if len(parts) == 2 else (parts[0], "")
        color = CONFIG_COLORS.get(key)
        if color is None:
            color = sns.color_palette("tab10")[len(palette) % 10]
        palette[label] = color
    return palette


def _load_index(path: Path, levels: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run postprocessing_v3/data_loading.py first.")
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError(f"{path.name} must be a MultiIndex table")
    if df.index.names != levels:
        df.index = df.index.set_names(levels)
    return df


def load_suite_runs(suite: str) -> pd.DataFrame:
    levels = [
        "Schedule",
        "Communication",
        "N Ranks",
        "Chunk Size",
        "Domain",
        "Image Size",
        "Run Id",
        "Suite",
    ]
    runs_idx = _load_index(RUNS_IDX, levels)
    try:
        subset = runs_idx.xs(suite, level="Suite").reset_index()
    except KeyError as exc:
        raise ValueError(f"Suite '{suite}' not present in indexed runs") from exc
    return subset


def load_suite_ranks(suite: str) -> pd.DataFrame:
    levels = [
        "Schedule",
        "Communication",
        "N Ranks",
        "Chunk Size",
        "Domain",
        "Image Size",
        "Run Id",
        "Suite",
        "rank",
    ]
    ranks_idx = _load_index(RANKS_IDX, levels)
    try:
        subset = ranks_idx.xs(suite, level="Suite").reset_index()
    except KeyError as exc:
        raise ValueError(f"Suite '{suite}' not present in indexed ranks") from exc
    return subset


def load_suite_chunks(suite: str) -> pd.DataFrame:
    levels = [
        "Schedule",
        "Communication",
        "N Ranks",
        "Chunk Size",
        "Domain",
        "Image Size",
        "Run Id",
        "Suite",
        "rank",
        "chunk_id",
    ]
    chunks_idx = _load_index(CHUNKS_IDX, levels)
    try:
        subset = chunks_idx.xs(suite, level="Suite").reset_index()
    except KeyError as exc:
        raise ValueError(f"Suite '{suite}' not present in indexed chunks") from exc
    return subset
