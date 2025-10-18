"""Configuration objects and YAML loading for Mandelbrot MPI experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml


@dataclass(frozen=True)
class RunConfig:
    """Runtime configuration for a single Mandelbrot experiment run."""

    n_ranks: int
    chunk_size: int
    schedule: str  # 'static' or 'dynamic'
    communication: str  # 'blocking' or 'nonblocking'
    width: int
    height: int
    xlim: Tuple[float, float] = (-2.2, 0.75)
    ylim: Tuple[float, float] = (-1.3, 1.3)

    @property
    def total_chunks(self) -> int:
        return (self.width + self.chunk_size - 1) // self.chunk_size

    @property
    def run_name(self) -> str:
        """Generate unique run name embedding all parameters."""
        return (
            f"{self.schedule}_{self.communication}_n{self.n_ranks}_"
            f"c{self.chunk_size}_{self.width}x{self.height}"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for MLflow logging."""
        return asdict(self)

    def to_cli_args(self) -> List[str]:
        """Convert config to CLI arguments."""
        return [
            f"--n-ranks={self.n_ranks}",
            f"--chunk-size={self.chunk_size}",
            f"--schedule={self.schedule}",
            f"--communication={self.communication}",
            f"--image-size={self.image_size}",
            f"--xlim={self.xlim[0]}:{self.xlim[1]}",
            f"--ylim={self.ylim[0]}:{self.ylim[1]}",
        ]

    @property
    def image_size(self) -> str:
        return f"{self.width}x{self.height}"


DEFAULT_RUN_CONFIG = RunConfig(
    n_ranks=4,
    chunk_size=10,
    schedule="static",
    communication="blocking",
    width=100,
    height=100,
    xlim=(-2.2, 0.75),
    ylim=(-1.3, 1.3),
)


def default_run_config(**overrides: object) -> RunConfig:
    """Return the canonical default config optionally overridden with kwargs."""
    return replace(DEFAULT_RUN_CONFIG, **_coerce_dimensions(overrides))


def load_sweep_configs(yaml_path: str | Path) -> List[RunConfig]:
    """Load YAML config and generate all parameter sweep combinations.

    Supports the original format with top-level ``sweep`` as well as
    the resource-group format that nests multiple experiments under ``experiments``.
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f) or {}

    global_defaults: Dict[str, object] = cfg.get("defaults", {}) or {}

    if "experiments" in cfg:
        experiments = cfg.get("experiments") or []
        configs: List[RunConfig] = []
        for exp in experiments:
            sweep = exp.get("sweep")
            if not sweep:
                continue
            exp_defaults = {**global_defaults, **(exp.get("defaults", {}) or {})}
            configs.extend(_expand_sweep(exp_defaults, sweep))
        return configs

    sweep: Dict[str, object] = cfg.get("sweep", {}) or {}
    return _expand_sweep(global_defaults, sweep)


def get_config_by_index(yaml_path: str | Path, index: int) -> RunConfig:
    """Get a specific config by index from sweep."""
    configs = load_sweep_configs(yaml_path)
    if index < 0 or index >= len(configs):
        raise ValueError(f"Config index {index} out of range [0, {len(configs) - 1}]")
    return configs[index]


def _build_run_config(raw_data: Dict[str, object]) -> RunConfig:
    data = dict(raw_data)
    data = _coerce_dimensions(data)
    if "xlim" in data:
        data["xlim"] = tuple(map(float, data["xlim"]))
    if "ylim" in data:
        data["ylim"] = tuple(map(float, data["ylim"]))
    return RunConfig(**data)  # type: ignore[arg-type]


def _expand_sweep(defaults: Dict[str, object], sweep: Dict[str, object]) -> List[RunConfig]:
    """Expand sweep definition into RunConfig instances."""
    configs: List[RunConfig] = []

    domains = sweep.get("domains")
    param_grid = {k: sweep[k] for k in sweep if k not in {"domains", "image_shape"}}
    shape_options = sweep.get("image_shape")

    if domains:
        keys = list(param_grid.keys())
        for domain in domains:
            xlim, ylim = domain
            combos = product(*[param_grid[k] for k in keys]) if keys else [()]
            for combo in combos:
                data = {**defaults, **dict(zip(keys, combo))}
                data["xlim"] = tuple(map(float, xlim))
                data["ylim"] = tuple(map(float, ylim))
                configs.extend(_expand_shapes(data, shape_options))
    else:
        keys = list(param_grid.keys())
        if not keys:
            configs.extend(_expand_shapes(defaults, shape_options))
        else:
            values = [param_grid[k] for k in keys]
            for combo in product(*values):
                data = {**defaults, **dict(zip(keys, combo))}
                configs.extend(_expand_shapes(data, shape_options))

    return configs


def load_named_sweep_configs(
    yaml_path: str | Path,
    suite: str | None = None,
) -> List[tuple[str, List[RunConfig]]]:
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f) or {}

    defaults: Dict[str, object] = cfg.get("defaults", {}) or {}
    experiments = cfg.get("experiments")
    results: List[tuple[str, List[RunConfig]]] = []

    if experiments:
        for exp in experiments:
            name = exp.get("name")
            if not name:
                continue
            if suite and name != suite:
                continue
            sweep = exp.get("sweep") or {}
            exp_defaults = {**defaults, **(exp.get("defaults", {}) or {})}
            configs = _expand_sweep(exp_defaults, sweep)
            results.append((name, configs))
        if suite and not results:
            raise ValueError(f"Suite '{suite}' not found in {yaml_path}")
        return results

    sweep: Dict[str, object] = cfg.get("sweep", {}) or {}
    configs = _expand_sweep(defaults, sweep)
    label = cfg.get("name") or Path(yaml_path).stem
    return [(label, configs)]


def parse_image_size(value: str) -> Tuple[int, int]:
    width_str, height_str = value.lower().split("x")
    return int(width_str.strip()), int(height_str.strip())


def _coerce_dimensions(data: Dict[str, object]) -> Dict[str, object]:
    result = dict(data)
    image = result.pop("image_size", None)
    if image is not None:
        width, height = _normalize_shape_entry(image)
        result.setdefault("width", width)
        result.setdefault("height", height)
    shape = result.pop("image_shape", None)
    if shape is not None:
        width, height = _normalize_shape_entry(shape)
        result.setdefault("width", width)
        result.setdefault("height", height)
    if "width" in result:
        result["width"] = int(result["width"])
    if "height" in result:
        result["height"] = int(result["height"])
    return result


def _normalize_shape_entry(entry: object) -> Tuple[int, int]:
    if isinstance(entry, dict):
        width = entry.get("width")
        height = entry.get("height")
        if width is None or height is None:
            raise ValueError("image_shape dict must include 'width' and 'height'")
        return int(width), int(height)
    if isinstance(entry, (list, tuple)) and len(entry) == 2:
        return int(entry[0]), int(entry[1])
    if isinstance(entry, str):
        return parse_image_size(entry)
    raise ValueError(f"Unsupported image shape specification: {entry!r}")


def _expand_shapes(base: Dict[str, object], shape_options: object) -> List[RunConfig]:
    if not shape_options:
        return [_build_run_config(base)]

    shapes: Iterable[Tuple[int, int]]
    if isinstance(shape_options, (list, tuple)):
        shapes = [_normalize_shape_entry(opt) for opt in shape_options]
    else:
        shapes = [_normalize_shape_entry(shape_options)]

    configs = []
    for width, height in shapes:
        data = {**base, "width": width, "height": height}
        configs.append(_build_run_config(data))
    return configs
