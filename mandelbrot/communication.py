"""MPI communication patterns for Mandelbrot computation."""

from __future__ import annotations

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


def run_static(comm: MPI.Intracomm, config: RunConfig, scheduler: StaticScheduler, blocking: bool = True) -> np.ndarray | None:
    """Run static scheduling with specified communication pattern."""
    rank = comm.Get_rank()

    local_results: List[ChunkResult] = []
    for chunk_id in scheduler.chunks_for_rank(rank):
        local_results.append(compute_chunk(config, chunk_id))

    if blocking:
        return _gather_blocking(comm, config, local_results)
    else:
        return _gather_nonblocking(comm, config, local_results)


def run_dynamic(comm: MPI.Intracomm, config: RunConfig, scheduler: DynamicScheduler, blocking: bool = True) -> np.ndarray | None:
    """Run dynamic scheduling with specified communication pattern."""
    rank = comm.Get_rank()
    if rank == 0:
        if blocking:
            return _master_loop_blocking(comm, config, scheduler)
        else:
            return _master_loop_nonblocking(comm, config, scheduler)
    if blocking:
        _worker_loop_blocking(comm, config)
    else:
        _worker_loop_nonblocking(comm, config)
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


def _master_loop_blocking(comm: MPI.Intracomm, config: RunConfig, scheduler: DynamicScheduler) -> np.ndarray:
    """Master loop for dynamic scheduling with blocking communication."""
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

    return full_image


def _master_loop_nonblocking(comm: MPI.Intracomm, config: RunConfig, scheduler: DynamicScheduler) -> np.ndarray:
    """Master loop for dynamic scheduling with non-blocking communication."""
    full_image = allocate_image(config)
    world_size = comm.Get_size()

    request_pool: List[RequestEntry] = []
    for worker_rank in range(1, world_size):
        buf = np.array(0, dtype=np.int32)
        req = comm.Irecv(buf, source=worker_rank, tag=REQUEST_TAG)
        request_pool.append((req, worker_rank, buf))

    result_pool: List[ResultEntry] = []
    for worker_rank in range(1, world_size):
        meta_buf = np.zeros(2, dtype=np.int32)
        data_buf = np.zeros((config.chunk_size, config.height), dtype=np.float64)
        req_meta = comm.Irecv(meta_buf, source=worker_rank, tag=META_TAG)
        req_data = comm.Irecv(data_buf, source=worker_rank, tag=DATA_TAG)
        result_pool.append((req_meta, req_data, worker_rank, meta_buf, data_buf))

    results_received = 0
    worker_chunks = {rank: 0 for rank in range(world_size)}
    done_ranks = set()

    while results_received < config.total_chunks or len(done_ranks) < world_size - 1:
        for idx, (req, worker_rank, buf) in enumerate(request_pool):
            if req is not None and req.Test():
                chunk_id = scheduler.request_chunk()
                assignment = np.array(-1 if chunk_id is None else chunk_id, dtype=np.int32)
                comm.Send(assignment, dest=worker_rank, tag=ASSIGN_TAG)
                if chunk_id is None:
                    request_pool[idx] = (None, worker_rank, None)
                else:
                    new_buf = np.array(0, dtype=np.int32)
                    new_req = comm.Irecv(new_buf, source=worker_rank, tag=REQUEST_TAG)
                    request_pool[idx] = (new_req, worker_rank, new_buf)

        for idx, (req_meta, req_data, worker_rank, meta_buf, data_buf) in enumerate(result_pool):
            if req_meta.Test() and req_data.Test():
                start_row, end_row = map(int, meta_buf)
                rows = end_row - start_row
                full_image[start_row:end_row, :] = data_buf[:rows, :]
                results_received += 1
                worker_chunks[worker_rank] += 1
                new_meta = np.zeros(2, dtype=np.int32)
                new_data = np.zeros((config.chunk_size, config.height), dtype=np.float64)
                result_pool[idx] = (
                    comm.Irecv(new_meta, source=worker_rank, tag=META_TAG),
                    comm.Irecv(new_data, source=worker_rank, tag=DATA_TAG),
                    worker_rank,
                    new_meta,
                    new_data,
                )

        status = MPI.Status()
        while comm.Iprobe(source=MPI.ANY_SOURCE, tag=DONE_TAG, status=status):
            source = status.Get_source()
            count_buf = np.array(0, dtype=np.int32)
            comm.Recv(count_buf, source=source, tag=DONE_TAG)
            worker_chunks[source] = int(count_buf)
            done_ranks.add(source)

    for worker in range(1, world_size):
        print(f"Worker {worker} processed {worker_chunks[worker]} chunks", flush=True)

    return full_image


def _worker_loop_blocking(comm: MPI.Intracomm, config: RunConfig) -> None:
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


def _worker_loop_nonblocking(comm: MPI.Intracomm, config: RunConfig) -> None:
    """Worker loop for dynamic scheduling with non-blocking communication."""
    chunks_processed = 0
    active_sends: List[Tuple[MPI.Request, MPI.Request]] = []

    assignment_buf = np.array(-1, dtype=np.int32)
    req_work = comm.Isend(np.array(0, dtype=np.int32), dest=0, tag=REQUEST_TAG)
    req_assign = comm.Irecv(assignment_buf, source=0, tag=ASSIGN_TAG)

    while True:
        req_assign.Wait()
        req_work.Wait()
        chunk_id = int(assignment_buf)
        if chunk_id < 0:
            break

        start_row, end_row, chunk = compute_chunk(config, chunk_id)
        rows = end_row - start_row
        metadata = np.array([start_row, end_row], dtype=np.int32)
        padded = np.zeros((config.chunk_size, config.height), dtype=np.float64)
        padded[:rows, :] = chunk
        req_meta = comm.Isend(metadata, dest=0, tag=META_TAG)
        req_data = comm.Isend(padded, dest=0, tag=DATA_TAG)
        active_sends.append((req_meta, req_data))
        chunks_processed += 1

        req_work = comm.Isend(np.array(0, dtype=np.int32), dest=0, tag=REQUEST_TAG)
        assignment_buf = np.array(-1, dtype=np.int32)
        req_assign = comm.Irecv(assignment_buf, source=0, tag=ASSIGN_TAG)

        active_sends = [
            (rm, rd) for rm, rd in active_sends if not (rm.Test() and rd.Test())
        ]

    for req_meta, req_data in active_sends:
        req_meta.Wait()
        req_data.Wait()

    comm.Send(np.array(chunks_processed, dtype=np.int32), dest=0, tag=DONE_TAG)