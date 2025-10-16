"""Simplified MPI communication patterns for Mandelbrot computation."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from mpi4py import MPI

from .computation import allocate_image, compute_chunk
from .config import RunConfig
from .report import ChunkReport
from .scheduling import DynamicScheduler, StaticScheduler

__all__ = ["run_mpi_computation"]

# MPI tags
REQUEST_TAG = 10
ASSIGN_TAG = 11
DATA_TAG = 20


def _rank_log(rank: int, message: str) -> None:
    """Emit a progress message from a given MPI rank."""
    print(f"[Rank {rank}] {message}", flush=True)


def _compute_chunk_timed(config: RunConfig, chunk_id: int) -> tuple[int, int, np.ndarray, float]:
    """Compute a chunk and return start/end rows along with elapsed time."""
    comp_start = MPI.Wtime()
    start, end, chunk = compute_chunk(config, chunk_id)
    return start, end, chunk, MPI.Wtime() - comp_start


def _receive_worker_chunk(comm: MPI.Intracomm, worker: int, image: np.ndarray) -> None:
    """Receive a chunk produced by a worker and write it into the image."""
    start, end = comm.recv(source=worker, tag=DATA_TAG)
    chunk = comm.recv(source=worker, tag=DATA_TAG)
    image[start:end, :] = chunk


def _assign_chunk(comm: MPI.Intracomm, scheduler: DynamicScheduler, worker: int) -> bool:
    """Assign the next chunk to a worker, or send shutdown if depleted."""
    chunk_id = scheduler.request_chunk()
    if chunk_id is not None:
        _rank_log(0, f"Assigning chunk {chunk_id} to worker {worker}")
        payload = chunk_id
    else:
        _rank_log(0, f"No more chunks - sending shutdown to worker {worker}")
        payload = -1
    comm.send(payload, dest=worker, tag=ASSIGN_TAG)
    return chunk_id is not None


def _send_chunk_payload(
    comm: MPI.Intracomm,
    dest: int,
    start: int,
    end: int,
    chunk: np.ndarray,
) -> None:
    """Send chunk metadata and payload to the destination rank."""
    comm.send((start, end), dest=dest, tag=DATA_TAG)
    comm.send(chunk, dest=dest, tag=DATA_TAG)


def _send_chunk_count(comm: MPI.Intracomm, count: int, mode: str) -> None:
    """Send the number of chunks a worker produced."""
    if mode == "nonblocking":
        buf = np.array(count, dtype=np.int32)
        req = comm.Isend(buf, dest=0, tag=DATA_TAG)
        req.Wait()
    else:
        comm.send(count, dest=0, tag=DATA_TAG)


def _iter_worker_counts(comm: MPI.Intracomm, size: int, mode: str) -> List[tuple[int, int]]:
    """Collect chunk counts from workers."""
    if mode == "nonblocking":
        requests = []
        for worker in range(1, size):
            buf = np.array(0, dtype=np.int32)
            req = comm.Irecv(buf, source=worker, tag=DATA_TAG)
            requests.append((req, buf, worker))
        pairs: List[tuple[int, int]] = []
        for req, buf, worker in requests:
            req.Wait()
            pairs.append((worker, int(buf)))
        return pairs

    return [(worker, comm.recv(source=worker, tag=DATA_TAG)) for worker in range(1, size)]


def _chunk_record(rank: int, chunk_id: int, start: int, end: int, comp_time: float) -> Dict:
    """Create a uniform chunk metadata record."""
    return {
        "rank": rank,
        "chunk_id": int(chunk_id),
        "start_row": int(start),
        "end_row": int(end - 1) if end > start else int(end),
        "comp_time": comp_time,
    }


def run_mpi_computation(config: RunConfig) -> Tuple[np.ndarray | None, Dict, List[Dict]]:
    """Execute MPI computation and return consolidated results."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    start_time = MPI.Wtime()
    
    if config.schedule == "static":
        image, rank_times, chunk_records = _run_static(comm, config, rank, size)
    else:
        image, rank_times, chunk_records = _run_dynamic(comm, config, rank, size)
    
    # Gather timing stats and per-chunk records to rank 0
    all_times = comm.gather(rank_times, root=0)
    all_chunks = comm.gather(chunk_records, root=0)

    total_time = MPI.Wtime() - start_time

    timing_stats: Dict[str, float] = {}
    chunk_records: List[Dict[str, Any]] = []
    if rank == 0:
        timing_stats = _aggregate_timing(all_times, total_time)
        chunk_records = [record for records in all_chunks for record in records]

    return ChunkReport(image if rank == 0 else None, timing_stats, chunk_records if chunk_records else None)


def _run_static(
    comm: MPI.Intracomm,
    config: RunConfig,
    rank: int,
    size: int,
) -> Tuple[np.ndarray | None, Dict, List[Dict]]:
    """Static scheduling: pre-assigned chunks."""
    scheduler = StaticScheduler(config, size)
    chunk_ids = scheduler.chunks_for_rank(rank)

    results: List[Tuple[int, int, np.ndarray]] = []
    chunk_details: List[Dict] = []
    comp_time = 0.0

    for cid in chunk_ids:
        start, end, chunk, single_comp = _compute_chunk_timed(config, cid)
        _rank_log(
            rank,
            f"Computing chunk {cid} (rows {start}:{end}) took {single_comp:.4f}s "
            f"[static]",
        )
        comp_time += single_comp
        results.append((start, end, chunk))
        chunk_details.append(_chunk_record(rank, cid, start, end, single_comp))

    # Gather results
    comm_start = MPI.Wtime()
    image = _gather_results(comm, config, results, rank, size, config.communication)
    comm_time = MPI.Wtime() - comm_start
    
    return image, {"comp": comp_time, "comm": comm_time, "chunks": len(chunk_ids)}, chunk_details


def _run_dynamic(
    comm: MPI.Intracomm,
    config: RunConfig,
    rank: int,
    size: int,
) -> Tuple[np.ndarray | None, Dict, List[Dict]]:
    """Dynamic scheduling: on-demand chunk assignment."""
    # Single process - compute all locally
    if size == 1:
        results: List[Tuple[int, int, np.ndarray]] = []
        chunk_details: List[Dict] = []
        total_comp = 0.0

        for chunk_id in range(config.total_chunks):
            start, end, chunk, single_comp = _compute_chunk_timed(config, chunk_id)
            _rank_log(
                rank,
                f"Computing chunk {chunk_id} (rows {start}:{end}) took {single_comp:.4f}s "
                f"[dynamic-single]",
            )
            total_comp += single_comp
            results.append((start, end, chunk))
            chunk_details.append(_chunk_record(rank, chunk_id, start, end, single_comp))

        image = allocate_image(config)
        for start, end, chunk in results:
            image[start:end, :] = chunk

        return image, {"comp": total_comp, "comm": 0.0, "chunks": len(results)}, chunk_details
    
    # Rank 0 is master, others are workers
    if rank == 0:
        image, stats, chunk_details = _master_dynamic(comm, config, size)
        return image, stats, chunk_details
    else:
        stats, chunk_details = _worker_dynamic(comm, config)
        return None, stats, chunk_details


def _master_dynamic(
    comm: MPI.Intracomm,
    config: RunConfig,
    size: int,
) -> Tuple[np.ndarray, Dict, List[Dict]]:
    """Master rank for dynamic scheduling."""
    scheduler = DynamicScheduler(config)
    image = allocate_image(config)
    
    total_comp = 0.0
    chunk_details: List[Dict] = []

    # Process master's own chunks while managing workers
    master_chunks = 0
    active_workers = size - 1
    
    # Initial assignment to all workers
    for worker in range(1, size):
        if not _assign_chunk(comm, scheduler, worker):
            active_workers -= 1

    status = MPI.Status()

    # Process requests
    while active_workers > 0:
        # Check for worker requests (non-blocking)
        if comm.iprobe(source=MPI.ANY_SOURCE, tag=REQUEST_TAG, status=status):
            worker = status.Get_source()
            comm.recv(source=worker, tag=REQUEST_TAG)  # Dummy receive

            # Receive worker's result
            _receive_worker_chunk(comm, worker, image)

            # Assign next chunk
            if not _assign_chunk(comm, scheduler, worker):
                active_workers -= 1
            continue

        # Do master's own work when chunks remain
        chunk_id = scheduler.request_chunk()
        if chunk_id is not None:
            start, end, chunk, single_comp = _compute_chunk_timed(config, chunk_id)
            _rank_log(
                0,
                f"Master computing chunk {chunk_id} (rows {start}:{end}) took {single_comp:.4f}s",
            )
            total_comp += single_comp
            image[start:end, :] = chunk
            master_chunks += 1
            chunk_details.append(_chunk_record(0, chunk_id, start, end, single_comp))
            continue

        # No master chunk available: block until a worker reports completion
        comm.probe(source=MPI.ANY_SOURCE, tag=REQUEST_TAG, status=status)
        worker = status.Get_source()
        comm.recv(source=worker, tag=REQUEST_TAG)
        _receive_worker_chunk(comm, worker, image)

        if not _assign_chunk(comm, scheduler, worker):
            active_workers -= 1

    return image, {"comp": total_comp, "comm": 0.0, "chunks": master_chunks}, chunk_details


def _worker_dynamic(
    comm: MPI.Intracomm,
    config: RunConfig,
) -> Tuple[Dict, List[Dict]]:
    """Worker rank for dynamic scheduling."""
    rank = comm.Get_rank()
    comp_time = 0.0
    comm_time = 0.0
    chunks_done = 0
    chunk_details: List[Dict] = []
    
    while True:
        # Get assignment
        comm_start = MPI.Wtime()
        chunk_id = comm.recv(source=0, tag=ASSIGN_TAG)
        comm_time += MPI.Wtime() - comm_start
        
        if chunk_id == -1:  # Done signal
            _rank_log(rank, f"Received shutdown signal after {chunks_done} chunks")
            break

        # Compute chunk
        start, end, chunk, single_comp = _compute_chunk_timed(config, chunk_id)
        _rank_log(
            rank,
            f"Computing chunk {chunk_id} (rows {start}:{end}) took {single_comp:.4f}s",
        )
        comp_time += single_comp

        # Send result and request more
        comm_start = MPI.Wtime()
        comm.send(None, dest=0, tag=REQUEST_TAG)
        _send_chunk_payload(comm, dest=0, start=start, end=end, chunk=chunk)
        comm_time += MPI.Wtime() - comm_start

        chunks_done += 1
        chunk_details.append(_chunk_record(rank, chunk_id, start, end, single_comp))

    return {"comp": comp_time, "comm": comm_time, "chunks": chunks_done}, chunk_details


def _gather_results(
    comm: MPI.Intracomm,
    config: RunConfig,
    results: List[Tuple[int, int, np.ndarray]],
    rank: int,
    size: int,
    mode: str,
) -> np.ndarray | None:
    """Gather chunk results back to the master rank using selected communication mode."""
    if rank == 0:
        image = allocate_image(config)
        for start, end, chunk in results:
            image[start:end, :] = chunk

        for worker, count in _iter_worker_counts(comm, size, mode):
            for _ in range(count):
                _receive_worker_chunk(comm, worker, image)

        return image

    _send_chunk_count(comm, len(results), mode)
    for start, end, chunk in results:
        _send_chunk_payload(comm, dest=0, start=start, end=end, chunk=chunk)
    return None


def _aggregate_timing(all_times: List[Dict], total_time: float) -> Dict:
    """Aggregate timing statistics from all ranks."""
    comp_times = [t["comp"] for t in all_times]
    comm_times = [t["comm"] for t in all_times]
    chunks = [t["chunks"] for t in all_times]

    return {
        "total_time": total_time,
        "total_comp_time": float(np.sum(comp_times)),
        "total_comm_time": float(np.sum(comm_times)),
        "comp_std": float(np.std(comp_times)) if comp_times else 0.0,
        "comm_std": float(np.std(comm_times)) if comm_times else 0.0,
        "total_chunks": sum(chunks),
        **{f"rank_{i}_comp": t["comp"] for i, t in enumerate(all_times)},
        **{f"rank_{i}_comm": t["comm"] for i, t in enumerate(all_times)},
        **{f"rank_{i}_chunks": t["chunks"] for i, t in enumerate(all_times)},
    }
