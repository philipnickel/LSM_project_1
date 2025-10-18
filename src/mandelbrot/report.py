"""Structured results returned from MPI execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class ChunkReport:
    """Container for outputs produced by ``run_mpi_computation``."""

    image: Optional[np.ndarray]
    timing: Dict[str, Any]
    chunks: Optional[List[Dict[str, Any]]]

    def copy_chunks(self) -> Optional[List[Dict[str, Any]]]:
        if self.chunks is None:
            return None
        return [record.copy() for record in self.chunks]
