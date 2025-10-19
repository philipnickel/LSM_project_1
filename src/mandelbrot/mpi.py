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


def _init_rank_stats() -> Dict[str, float]:
    return {
        "comp": 0.0,
        "comm_send": 0.0,
        "comm_recv": 0.0,
        "chunks": 0.0,
    }


def _rank_log(rank: int, message: str) -> None:
    """Emit a progress message from a given MPI rank."""
    print(f"[Rank {rank}] {message}", flush=True)


def _compute_chunk_timed(config: RunConfig, chunk_id: int) -> tuple[int, int, np.ndarray, float]:
    """Compute a chunk and return start/end rows along with elapsed time."""
    comp_start = MPI.Wtime()
    start, end, chunk = compute_chunk(config, chunk_id)
    return start, end, chunk, MPI.Wtime() - comp_start


def _receive_worker_chunk(
    comm: MPI.Intracomm,
    worker: int,
    image: np.ndarray,
    stats: Dict[str, float] | None = None,
) -> None:
    """Receive a chunk produced by a worker and write it into the image."""
    t0 = MPI.Wtime()
    start, end = comm.recv(source=worker, tag=DATA_TAG)
    chunk = comm.recv(source=worker, tag=DATA_TAG)
    image[start:end, :] = chunk
    if stats is not None:
        stats["comm_recv"] += MPI.Wtime() - t0


def _assign_chunk(
    comm: MPI.Intracomm,
    scheduler: DynamicScheduler,
    worker: int,
    stats: Dict[str, float] | None = None,
) -> bool:
    """Assign the next chunk to a worker, or send shutdown if depleted."""
    chunk_id = scheduler.request_chunk()
    if chunk_id is not None:
        _rank_log(0, f"Assigning chunk {chunk_id} to worker {worker}")
        payload = chunk_id
    else:
        _rank_log(0, f"No more chunks - sending shutdown to worker {worker}")
        payload = -1
    t0 = MPI.Wtime()
    comm.send(payload, dest=worker, tag=ASSIGN_TAG)
    if stats is not None:
        stats["comm_send"] += MPI.Wtime() - t0
    return chunk_id is not None


def _send_chunk_payload(
    comm: MPI.Intracomm,
    dest: int,
    start: int,
    end: int,
    chunk: np.ndarray,
    stats: Dict[str, float] | None = None,
) -> None:
    """Send chunk metadata and payload to the destination rank."""
    t0 = MPI.Wtime()
    comm.send((start, end), dest=dest, tag=DATA_TAG)
    comm.send(chunk, dest=dest, tag=DATA_TAG)
    if stats is not None:
        stats["comm_send"] += MPI.Wtime() - t0


def _send_chunk_count(
    comm: MPI.Intracomm,
    count: int,
    mode: str,
    stats: Dict[str, float] | None = None,
) -> None:
    """Send the number of chunks a worker produced."""
    t0 = MPI.Wtime()
    if mode == "nonblocking":
        buf = np.array(count, dtype=np.int32)
        req = comm.Isend(buf, dest=0, tag=DATA_TAG)
        req.Wait()
    else:
        comm.send(count, dest=0, tag=DATA_TAG)
    if stats is not None:
        stats["comm_send"] += MPI.Wtime() - t0


def _iter_worker_counts(
    comm: MPI.Intracomm,
    size: int,
    mode: str,
    stats: Dict[str, float] | None = None,
) -> List[tuple[int, int]]:
    """Collect chunk counts from workers."""
    if mode == "nonblocking":
        requests = []
        for worker in range(1, size):
            buf = np.array(0, dtype=np.int32)
            req = comm.Irecv(buf, source=worker, tag=DATA_TAG)
            requests.append((req, buf, worker))
        pairs: List[tuple[int, int]] = []
        for req, buf, worker in requests:
            t0 = MPI.Wtime()
            req.Wait()
            if stats is not None:
                stats["comm_recv"] += MPI.Wtime() - t0
            pairs.append((worker, int(buf)))
        return pairs

    pairs = []
    for worker in range(1, size):
        t0 = MPI.Wtime()
        count = comm.recv(source=worker, tag=DATA_TAG)
        if stats is not None:
            stats["comm_recv"] += MPI.Wtime() - t0
        pairs.append((worker, count))
    return pairs


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

    # Excluding time spend gathering timings
    total_time = MPI.Wtime() - start_time

    # Gather timing stats and per-chunk records to rank 0
    all_times = comm.gather(rank_times, root=0)
    all_chunks = comm.gather(chunk_records, root=0)

    timing_stats: Dict[str, float] = {}
    chunk_records: List[Dict[str, Any]] = []
    if rank == 0:
        timing_stats = _aggregate_timing(all_times, total_time)
        chunk_records = [record for records in all_chunks for record in records]

    image_out = image if rank == 0 else None
    chunk_list = chunk_records if chunk_records else None
    return ChunkReport(image_out, timing_stats, chunk_list)


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
    stats = _init_rank_stats()

    for cid in chunk_ids:
        start, end, chunk, single_comp = _compute_chunk_timed(config, cid)
        _rank_log(
            rank,
            f"Computing chunk {cid} (rows {start}:{end}) took {single_comp:.4f}s [static]",
        )
        stats["comp"] += single_comp
        stats["chunks"] += 1
        results.append((start, end, chunk))
        chunk_details.append(_chunk_record(rank, cid, start, end, single_comp))

    # Gather results
    image = _gather_results(
        comm,
        config,
        results,
        rank,
        size,
        config.communication,
        stats,
    )

    return image, stats, chunk_details


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
        stats = _init_rank_stats()

        for chunk_id in range(config.total_chunks):
            start, end, chunk, single_comp = _compute_chunk_timed(config, chunk_id)
            _rank_log(
                rank,
                f"Computing chunk {chunk_id} (rows {start}:{end}) took {single_comp:.4f}s "
                f"[dynamic-single]",
            )
            stats["comp"] += single_comp
            stats["chunks"] += 1
            results.append((start, end, chunk))
            chunk_details.append(_chunk_record(rank, chunk_id, start, end, single_comp))

        image = allocate_image(config)
        for start, end, chunk in results:
            image[start:end, :] = chunk

        return image, stats, chunk_details

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

    stats = _init_rank_stats()
    chunk_details: List[Dict] = []

    active_workers = size - 1

    def handle_completion(worker: int) -> None:
        nonlocal active_workers
        recv_start = MPI.Wtime()
        comm.recv(source=worker, tag=REQUEST_TAG)
        stats["comm_recv"] += MPI.Wtime() - recv_start
        _receive_worker_chunk(comm, worker, image, stats)
        if not _assign_chunk(comm, scheduler, worker, stats):
            active_workers -= 1

    # Initial assignment to all workers
    for worker in range(1, size):
        if not _assign_chunk(comm, scheduler, worker, stats):
            active_workers -= 1

    status = MPI.Status()

    while active_workers > 0:
        while comm.iprobe(source=MPI.ANY_SOURCE, tag=REQUEST_TAG, status=status):
            worker = status.Get_source()
            handle_completion(worker)
            if active_workers <= 0:
                break

        if active_workers <= 0:
            break

        comm.probe(source=MPI.ANY_SOURCE, tag=REQUEST_TAG, status=status)
        worker = status.Get_source()
        handle_completion(worker)

    return image, stats, chunk_details


def _worker_dynamic(
    comm: MPI.Intracomm,
    config: RunConfig,
) -> Tuple[Dict, List[Dict]]:
    """Worker rank for dynamic scheduling."""
    rank = comm.Get_rank()
    stats = _init_rank_stats()
    chunk_details: List[Dict] = []

    while True:
        # Get assignment
        recv_start = MPI.Wtime()
        chunk_id = comm.recv(source=0, tag=ASSIGN_TAG)
        stats["comm_recv"] += MPI.Wtime() - recv_start

        if chunk_id == -1:  # Done signal
            _rank_log(rank, f"Received shutdown signal after {int(stats['chunks'])} chunks")
            break

        # Compute chunk
        start, end, chunk, single_comp = _compute_chunk_timed(config, chunk_id)
        _rank_log(
            rank,
            f"Computing chunk {chunk_id} (rows {start}:{end}) took {single_comp:.4f}s",
        )
        stats["comp"] += single_comp
        stats["chunks"] += 1

        # Send result and request more
        send_start = MPI.Wtime()
        comm.send(None, dest=0, tag=REQUEST_TAG)
        stats["comm_send"] += MPI.Wtime() - send_start
        _send_chunk_payload(
            comm,
            dest=0,
            start=start,
            end=end,
            chunk=chunk,
            stats=stats,
        )

        chunk_details.append(_chunk_record(rank, chunk_id, start, end, single_comp))

    return stats, chunk_details


def _gather_results(
    comm: MPI.Intracomm,
    config: RunConfig,
    results: List[Tuple[int, int, np.ndarray]],
    rank: int,
    size: int,
    mode: str,
    stats: Dict[str, float],
) -> np.ndarray | None:
    """Gather chunk results back to the master rank using selected communication mode."""
    if rank == 0:
        image = allocate_image(config)
        for start, end, chunk in results:
            image[start:end, :] = chunk

        for worker, count in _iter_worker_counts(comm, size, mode, stats):
            for _ in range(count):
                _receive_worker_chunk(comm, worker, image, stats)

        return image

    _send_chunk_count(comm, len(results), mode, stats)
    for start, end, chunk in results:
        _send_chunk_payload(
            comm,
            dest=0,
            start=start,
            end=end,
            chunk=chunk,
            stats=stats,
        )
    return None


def _aggregate_timing(all_times: List[Dict], total_time: float) -> Dict:
    """Aggregate wall-clock timing plus per-rank statistics."""
    rank_stats: List[Dict[str, float]] = []
    comp_total = 0.0
    comm_send_total = 0.0
    comm_recv_total = 0.0
    total_chunks = 0

    for rank, stats in enumerate(all_times):
        comp = float(stats.get("comp", 0.0))
        comm_send = float(stats.get("comm_send", 0.0))
        comm_recv = float(stats.get("comm_recv", 0.0))
        chunks = int(stats.get("chunks", 0))

        rank_stats.append(
            {
                "rank": int(rank),
                "comp_time": comp,
                "comm_time": comm_send + comm_recv,
                "comm_send_time": comm_send,
                "comm_recv_time": comm_recv,
                "chunks": chunks,
            }
        )

        comp_total += comp
        comm_send_total += comm_send
        comm_recv_total += comm_recv
        total_chunks += chunks

    return {
        "wall_time": float(total_time),
        "comp_total": comp_total,
        "comm_send_total": comm_send_total,
        "comm_recv_total": comm_recv_total,
        "comm_total": comm_send_total + comm_recv_total,
        "total_comp_time": comp_total,
        "total_comm_time": comm_send_total + comm_recv_total,
        "total_chunks": total_chunks,
        "rank_stats": rank_stats,
    }
