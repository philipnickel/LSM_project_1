"""Work scheduling strategies for Mandelbrot computation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from main import RunConfig


@dataclass
class StaticScheduler:
    """Static work scheduling - pre-assigns chunks to ranks."""
    config: RunConfig
    world_size: int
    assignments: Dict[int, List[int]] = field(init=False)

    def __post_init__(self) -> None:
        self.assignments = {rank: [] for rank in range(self.world_size)}
        for chunk_id in range(self.config.total_chunks):
            rank = chunk_id % self.world_size
            self.assignments[rank].append(chunk_id)

    def chunks_for_rank(self, rank: int) -> List[int]:
        """Get the list of chunk IDs assigned to a specific rank."""
        return self.assignments.get(rank, [])


@dataclass
class DynamicScheduler:
    """Dynamic work scheduling - assigns chunks on-demand."""
    config: RunConfig
    next_chunk: int = 0

    def request_chunk(self) -> Optional[int]:
        """Request the next available chunk, or None if all chunks are assigned."""
        if self.next_chunk >= self.config.total_chunks:
            return None
        chunk_id = self.next_chunk
        self.next_chunk += 1
        return chunk_id