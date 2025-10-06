"""MPI communication patterns for Mandelbrot computation."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from mpi4py import MPI

from .computation import allocate_image, compute_chunk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from main import RunConfig
from .scheduling import DynamicScheduler, StaticScheduler
# MPI message tags
REQUEST_TAG = 10
ASSIGN_TAG = 11
COUNT_TAG = 12
META_TAG = 20
DATA_TAG = 21
DONE_TAG = 99


ChunkResult = Tuple[int, int, np.ndarray]
RequestEntry = Tuple[Optional[MPI.Request], int, Optional[np.ndarray]]
ResultEntry = Tuple[MPI.Request, MPI.Request, int, np.ndarray, np.ndarray]


def run_static(comm: MPI.Intracomm, config: RunConfig, scheduler: StaticScheduler, blocking: bool = True, logger = None) -> np.ndarray | None:
    """Run static scheduling with specified communication pattern."""
    rank = comm.Get_rank()

    # Compute local chunks
    local_results: List[ChunkResult] = []
    chunk_ids = scheduler.chunks_for_rank(rank)
    comp_start = time.time()
    for chunk_id in chunk_ids:
        local_results.append(compute_chunk(config, chunk_id))
    comp_time = time.time() - comp_start
    
    # Log computation time
    if logger:
        logger.log_computation_time(comp_time)
        logger.log_worker_stats(rank, len(local_results), comp_time, 0.0, chunk_ids)

    # Gather results
    comm_start = time.time()
    if blocking:
        result = _gather_blocking(comm, config, local_results)
    else:
        result = _gather_nonblocking(comm, config, local_results)
    comm_time = time.time() - comm_start
    
    # Log communication time
    if logger:
        logger.log_communication_time(comm_time)
        # Update worker stats with communication time and chunk_ids
        logger.log_worker_stats(rank, len(local_results), comp_time, comm_time, chunk_ids)
    
    return result


def run_dynamic(comm: MPI.Intracomm, config: RunConfig, scheduler: DynamicScheduler, blocking: bool = True, logger = None) -> np.ndarray | None:
    """Run dynamic scheduling with specified communication pattern."""
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    
    # Handle single process case - just compute all chunks locally
    if world_size == 1:
        local_results: List[ChunkResult] = []
        chunk_ids = list(range(config.total_chunks))
        comp_start = time.time()
        for chunk_id in chunk_ids:
            local_results.append(compute_chunk(config, chunk_id))
        comp_time = time.time() - comp_start
        
        if logger:
            logger.log_computation_time(comp_time)
            logger.log_worker_stats(rank, len(local_results), comp_time, 0.0, chunk_ids)
        
        # Assemble full image
        full_image = allocate_image(config)
        for start, end, chunk in local_results:
            full_image[start:end, :] = chunk
        return full_image
    
    if rank == 0:
        if blocking:
            return _master_loop_blocking(comm, config, scheduler, logger)
        else:
            return _master_loop_nonblocking(comm, config, scheduler, logger)
    else:
        if blocking:
            _worker_loop_blocking(comm, config, logger)
        else:
            _worker_loop_nonblocking(comm, config, logger)
    return None


def _gather_blocking(
    comm: MPI.Intracomm, config: RunConfig, local_results: List[ChunkResult]
) -> np.ndarray | None:
    """Gather results using blocking MPI calls."""
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        full_image = allocate_image(config)
        for start, end, chunk in local_results:
            full_image[start:end, :] = chunk

        for source in range(1, size):
            num_chunks = np.array(0, dtype=np.int32)
            comm.Recv(num_chunks, source=source, tag=COUNT_TAG)
            for _ in range(int(num_chunks)):
                metadata = np.zeros(2, dtype=np.int32)
                comm.Recv(metadata, source=source, tag=META_TAG)
                start_row, end_row = map(int, metadata)
                rows = end_row - start_row
                buffer = np.zeros((rows, config.height), dtype=np.float64)
                comm.Recv(buffer, source=source, tag=DATA_TAG)
                full_image[start_row:end_row, :] = buffer
        return full_image

    num_chunks = np.array(len(local_results), dtype=np.int32)
    comm.Send(num_chunks, dest=0, tag=COUNT_TAG)
    for start_row, end_row, chunk in local_results:
        metadata = np.array([start_row, end_row], dtype=np.int32)
        comm.Send(metadata, dest=0, tag=META_TAG)
        comm.Send(chunk, dest=0, tag=DATA_TAG)
    return None


def _gather_nonblocking(
    comm: MPI.Intracomm, config: RunConfig, local_results: List[ChunkResult]
) -> np.ndarray | None:
    """Gather results using non-blocking MPI calls."""
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        full_image = allocate_image(config)
        for start, end, chunk in local_results:
            full_image[start:end, :] = chunk

        pending_receives = []
        for source in range(1, size):
            num_chunks = np.array(0, dtype=np.int32)
            comm.Recv(num_chunks, source=source, tag=COUNT_TAG)
            for _ in range(int(num_chunks)):
                metadata = np.zeros(2, dtype=np.int32)
                comm.Recv(metadata, source=source, tag=META_TAG)
                start_row, end_row = map(int, metadata)
                buffer = np.zeros((config.chunk_size, config.height), dtype=np.float64)
                req = comm.Irecv(buffer, source=source, tag=DATA_TAG)
                pending_receives.append((req, start_row, end_row, buffer))
        for req, start_row, end_row, buffer in pending_receives:
            req.Wait()
            rows = end_row - start_row
            full_image[start_row:end_row, :] = buffer[:rows, :]
        return full_image

    num_chunks = np.array(len(local_results), dtype=np.int32)
    comm.Send(num_chunks, dest=0, tag=COUNT_TAG)
    send_requests = []
    for start_row, end_row, chunk in local_results:
        metadata = np.array([start_row, end_row], dtype=np.int32)
        comm.Send(metadata, dest=0, tag=META_TAG)
        padded = np.zeros((config.chunk_size, config.height), dtype=np.float64)
        rows = end_row - start_row
        padded[:rows, :] = chunk
        req = comm.Isend(padded, dest=0, tag=DATA_TAG)
        send_requests.append(req)
    if send_requests:
        MPI.Request.Waitall(send_requests)
    return None


def _master_loop_blocking(comm: MPI.Intracomm, config: RunConfig, scheduler: DynamicScheduler, logger = None) -> np.ndarray:
    """Master loop for dynamic scheduling with blocking communication."""
    full_image = allocate_image(config)
    world_size = comm.Get_size()
    worker_chunks: Dict[int, int] = {rank: 0 for rank in range(world_size)}
    worker_chunk_ids: Dict[int, List[int]] = {rank: [] for rank in range(world_size)}
    done_ranks: Set[int] = set()
    results_received = 0

    while results_received < config.total_chunks or len(done_ranks) < world_size - 1:
        status = MPI.Status()
        comm.Probe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()

        if tag == REQUEST_TAG:
            comm.recv(source=source, tag=REQUEST_TAG)
            chunk_id = scheduler.request_chunk()
            payload = np.array(-1 if chunk_id is None else chunk_id, dtype=np.int32)
            comm.Send(payload, dest=source, tag=ASSIGN_TAG)
            if chunk_id is not None:
                worker_chunk_ids[source].append(chunk_id)
        elif tag == META_TAG:
            metadata = np.zeros(2, dtype=np.int32)
            comm.Recv(metadata, source=source, tag=META_TAG)
            start_row, end_row = map(int, metadata)
            rows = end_row - start_row
            buffer = np.zeros((rows, config.height), dtype=np.float64)
            comm.Recv(buffer, source=source, tag=DATA_TAG)
            full_image[start_row:end_row, :] = buffer
            results_received += 1
            worker_chunks[source] += 1
        elif tag == DONE_TAG:
            count_buf = np.array(0, dtype=np.int32)
            comm.Recv(count_buf, source=source, tag=DONE_TAG)
            worker_chunks[source] = int(count_buf)
            done_ranks.add(source)
        else:
            comm.recv(source=source, tag=tag)  # Drain unexpected message

    for worker in range(1, world_size):
        print(f"Worker {worker} processed {worker_chunks[worker]} chunks", flush=True)
        if logger:
            logger.log_worker_stats(worker, worker_chunks[worker], 0.0, 0.0, worker_chunk_ids[worker])

    # Log master stats (rank 0 doesn't process chunks in dynamic scheduling)
    if logger:
        logger.log_worker_stats(0, 0, 0.0, 0.0, [])

    return full_image


def _master_loop_nonblocking(comm: MPI.Intracomm, config: RunConfig, scheduler: DynamicScheduler, logger = None) -> np.ndarray:
    """Master loop for dynamic scheduling with non-blocking communication."""
    full_image = allocate_image(config)
    world_size = comm.Get_size()
    worker_chunks: Dict[int, int] = {rank: 0 for rank in range(world_size)}
    done_ranks: Set[int] = set()
    results_received = 0

    while results_received < config.total_chunks or len(done_ranks) < world_size - 1:
        status = MPI.Status()
        comm.Probe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()

        if tag == REQUEST_TAG:
            comm.recv(source=source, tag=REQUEST_TAG)
            chunk_id = scheduler.request_chunk()
            payload = np.array(-1 if chunk_id is None else chunk_id, dtype=np.int32)
            comm.Send(payload, dest=source, tag=ASSIGN_TAG)
        elif tag == META_TAG:
            metadata = np.zeros(2, dtype=np.int32)
            comm.Recv(metadata, source=source, tag=META_TAG)
            start_row, end_row = map(int, metadata)
            rows = end_row - start_row
            buffer = np.zeros((config.chunk_size, config.height), dtype=np.float64)
            comm.Recv(buffer, source=source, tag=DATA_TAG)
            full_image[start_row:end_row, :] = buffer[:rows, :]
            results_received += 1
            worker_chunks[source] += 1
        elif tag == DONE_TAG:
            count_buf = np.array(0, dtype=np.int32)
            comm.Recv(count_buf, source=source, tag=DONE_TAG)
            worker_chunks[source] = int(count_buf)
            done_ranks.add(source)
        else:
            comm.recv(source=source, tag=tag)  # Drain unexpected message

    for worker in range(1, world_size):
        print(f"Worker {worker} processed {worker_chunks[worker]} chunks", flush=True)
        if logger:
            logger.log_worker_stats(worker, worker_chunks[worker], 0.0, 0.0, worker_chunk_ids[worker])

    # Log master stats (rank 0 doesn't process chunks in dynamic scheduling)
    if logger:
        logger.log_worker_stats(0, 0, 0.0, 0.0, [])

    return full_image


def _worker_loop_blocking(comm: MPI.Intracomm, config: RunConfig, logger = None) -> None:
    """Worker loop for dynamic scheduling with blocking communication."""
    chunks_processed = 0
    while True:
        comm.send(None, dest=0, tag=REQUEST_TAG)
        assignment = np.array(0, dtype=np.int32)
        comm.Recv(assignment, source=0, tag=ASSIGN_TAG)
        chunk_id = int(assignment)
        if chunk_id < 0:
            break
        start_row, end_row, chunk = compute_chunk(config, chunk_id)
        metadata = np.array([start_row, end_row], dtype=np.int32)
        comm.Send(metadata, dest=0, tag=META_TAG)
        comm.Send(chunk, dest=0, tag=DATA_TAG)
        chunks_processed += 1

    comm.Send(np.array(chunks_processed, dtype=np.int32), dest=0, tag=DONE_TAG)


def _worker_loop_nonblocking(comm: MPI.Intracomm, config: RunConfig, logger = None) -> None:
    """Worker loop for dynamic scheduling with non-blocking communication."""
    chunks_processed = 0
    while True:
        comm.send(None, dest=0, tag=REQUEST_TAG)
        assignment = np.array(0, dtype=np.int32)
        comm.Recv(assignment, source=0, tag=ASSIGN_TAG)
        chunk_id = int(assignment)
        if chunk_id < 0:
            break
        start_row, end_row, chunk = compute_chunk(config, chunk_id)
        metadata = np.array([start_row, end_row], dtype=np.int32)
        comm.Send(metadata, dest=0, tag=META_TAG)
        comm.Send(chunk, dest=0, tag=DATA_TAG)
        chunks_processed += 1

    comm.Send(np.array(chunks_processed, dtype=np.int32), dest=0, tag=DONE_TAG)