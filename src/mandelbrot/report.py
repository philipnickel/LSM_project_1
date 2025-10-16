"""Structured results returned from MPI execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class ChunkReport:
    """Container for outputs produced by ``run_mpi_computation``."""

    image: Optional[np.ndarray]
    timing: Dict[str, float]
    chunks: Optional[List[Dict[str, Any]]]  # List of chunk records, not DataFrame

    def copy_chunks(self) -> Optional[List[Dict[str, Any]]]:
        """Return a defensive copy of the chunk records if present."""
        if self.chunks is None:
            return None
        return [record.copy() for record in self.chunks]
