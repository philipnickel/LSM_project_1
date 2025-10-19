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
    sns.set_style("whitegrid")
    plt.style.use("science")
    # sns.set_context("talk")


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
        raise FileNotFoundError(f"{path} not found. Run postprocessing/data_loading.py first.")
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError(f"{path.name} must be a MultiIndex table")
    if df.index.names != levels:
        df.index = df.index.set_names(levels)
    return df


def _slice_suite(df: pd.DataFrame, suite: str) -> pd.DataFrame:
    try:
        return df.xs(suite, level="Suite")
    except KeyError as exc:
        raise ValueError(f"Suite '{suite}' not present in indexed data") from exc


def load_suite_runs(suite: str, *, as_index: bool = False) -> pd.DataFrame:
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
    subset = _slice_suite(runs_idx, suite)
    return subset if as_index else subset.reset_index()


def load_suite_ranks(suite: str, *, as_index: bool = False) -> pd.DataFrame:
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
    subset = _slice_suite(ranks_idx, suite)
    return subset if as_index else subset.reset_index()


def load_suite_chunks(suite: str, *, as_index: bool = False) -> pd.DataFrame:
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
    subset = _slice_suite(chunks_idx, suite)
    return subset if as_index else subset.reset_index()


def ensure_config_column(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a human readable Config column based on schedule/communication."""
    if "Config" in df.columns:
        return df
    if isinstance(df.index, pd.MultiIndex) and {"Schedule", "Communication"}.issubset(
        set(df.index.names or [])
    ):
        schedules = df.index.get_level_values("Schedule")
        communications = df.index.get_level_values("Communication")
    else:
        schedules = df["Schedule"]
        communications = df["Communication"]
    labels = [config_label(s, c) for s, c in zip(schedules, communications)]
    result = df.copy()
    result["Config"] = labels
    return result


def ensure_config_level(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the MultiIndex dataframe includes a Config level for grouping."""
    if not isinstance(df.index, pd.MultiIndex):
        raise TypeError("ensure_config_level expects a MultiIndex dataframe")
    if "Config" in (df.index.names or []):
        return df
    schedules = df.index.get_level_values("Schedule")
    communications = df.index.get_level_values("Communication")
    labels = [config_label(s, c) for s, c in zip(schedules, communications)]
    augmented = df.copy()
    augmented["Config"] = labels
    augmented = augmented.set_index("Config", append=True)
    names = [name for name in augmented.index.names if name is not None]
    names_without = [name for name in names if name != "Config"]
    if "Communication" not in names_without:
        raise ValueError("MultiIndex is missing expected 'Communication' level")
    insert_pos = names_without.index("Communication") + 1
    reordered = names_without[:insert_pos] + ["Config"] + names_without[insert_pos:]
    augmented = augmented.reorder_levels(reordered)
    return augmented.sort_index()
