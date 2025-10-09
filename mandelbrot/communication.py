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


def run_static(comm: MPI.Intracomm, config: RunConfig, scheduler: StaticScheduler, blocking: bool = True, logger = None) -> Tuple[np.ndarray | None, Dict[int, Dict]]:
    """Run static scheduling with specified communication pattern.
    
    Returns:
        Tuple of (image, worker_stats) where worker_stats is a dict mapping rank -> stats dict
    """
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
    
    # Create worker stats
    worker_stats = {
        'chunks_processed': len(local_results),
        'computation_time': comp_time,
        'communication_time': comm_time,
        'chunk_ids': chunk_ids
    }
    
    return result, {rank: worker_stats}


def run_dynamic(comm: MPI.Intracomm, config: RunConfig, scheduler: DynamicScheduler, blocking: bool = True, logger = None) -> Tuple[np.ndarray | None, Dict[int, Dict]]:
    """Run dynamic scheduling with specified communication pattern.
    
    Returns:
        Tuple of (image, worker_stats) where worker_stats is a dict mapping rank -> stats dict
    """
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    worker_stats_dict = {}
    
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
        
        worker_stats_dict[0] = {
            'chunks_processed': len(local_results),
            'computation_time': comp_time, 
            'communication_time': 0.0,
            'chunk_ids': chunk_ids
        }
        
        # Assemble full image
        full_image = allocate_image(config)
        for start, end, chunk in local_results:
            full_image[start:end, :] = chunk
        return full_image, worker_stats_dict
    
    if rank == 0:
        if blocking:
            image, worker_stats_dict = _master_loop_blocking(comm, config, scheduler, logger)
        else:
            image, worker_stats_dict = _master_loop_nonblocking(comm, config, scheduler, logger)
        return image, worker_stats_dict
    else:
        try:
            if blocking:
                worker_stats = _worker_loop_blocking(comm, config, logger)
            else:
                worker_stats = _worker_loop_nonblocking(comm, config, logger)
                
            # Ensure stats are never None or empty
            if not worker_stats:
                worker_stats = {
                    'chunks_processed': 0,
                    'computation_time': 0.0,
                    'communication_time': 0.0,
                    'chunk_ids': []
                }
            
            return None, {rank: worker_stats}
        except Exception as e:
            print(f"ERROR: Worker {rank} failed: {e}", flush=True)
            # Return safe default stats
            return None, {rank: {
                'chunks_processed': 0,
                'computation_time': 0.0,
                'communication_time': 0.0,
                'chunk_ids': []
            }}


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
    """Gather results using non-blocking MPI calls (3-phase approach)."""
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Master: 3-phase non-blocking approach
        full_image = allocate_image(config)
        
        # Store own results
        for start, end, chunk in local_results:
            full_image[start:end, :] = chunk

        # Phase 1: Receive all metadata first
        all_metadata = []
        for source in range(1, size):
            num_chunks = np.array(0, dtype=np.int32)
            comm.Recv(num_chunks, source=source, tag=COUNT_TAG)
            for _ in range(int(num_chunks)):
                metadata = np.zeros(2, dtype=np.int32)
                comm.Recv(metadata, source=source, tag=META_TAG)
                start_row, end_row = map(int, metadata)
                all_metadata.append((source, start_row, end_row))

        # Phase 2: Post ALL non-blocking receives at once
        pending_receives = []
        for source, start_row, end_row in all_metadata:
            chunk_rows = end_row - start_row
            buffer = np.zeros((chunk_rows, config.height), dtype=np.float64)
            req = comm.Irecv(buffer, source=source, tag=DATA_TAG)
            pending_receives.append((req, start_row, end_row, buffer))

        # Phase 3: Wait for all receives and assemble image
        for req, start_row, end_row, buffer in pending_receives:
            req.Wait()
            full_image[start_row:end_row, :] = buffer
        
        return full_image
    else:
        # Worker: 2-phase approach
        # Phase 1: Send all metadata first
        num_chunks = np.array(len(local_results), dtype=np.int32)
        comm.Send(num_chunks, dest=0, tag=COUNT_TAG)
        
        for start_row, end_row, chunk in local_results:
            metadata = np.array([start_row, end_row], dtype=np.int32)
            comm.Send(metadata, dest=0, tag=META_TAG)

        # Phase 2: Send all data non-blockingly
        send_requests = []
        for start_row, end_row, chunk in local_results:
            req = comm.Isend(chunk, dest=0, tag=DATA_TAG)
            send_requests.append(req)
        
        if send_requests:
            MPI.Request.Waitall(send_requests)
        
        return None


def _master_loop_blocking(comm: MPI.Intracomm, config: RunConfig, scheduler: DynamicScheduler, logger = None) -> Tuple[np.ndarray, Dict[int, Dict]]:
    """Master loop for dynamic scheduling with blocking communication."""
    full_image = allocate_image(config)
    world_size = comm.Get_size()
    worker_chunks: Dict[int, int] = {rank: 0 for rank in range(world_size)}
    worker_chunk_ids: Dict[int, List[int]] = {rank: [] for rank in range(world_size)}
    worker_timing: Dict[int, Tuple[float, float]] = {}  # comp_time, comm_time
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
            # Don't increment worker_chunks here - rely on DONE message for accurate count
        elif tag == DONE_TAG:
            count_buf = np.array(0, dtype=np.int32)
            comm.Recv(count_buf, source=source, tag=DONE_TAG)
            worker_chunks[source] = int(count_buf)  # Use worker's reported count
            
            # Receive timing data
            timing_buf = np.zeros(2, dtype=np.float64)
            comm.Recv(timing_buf, source=source, tag=DONE_TAG + 1)
            worker_timing[source] = (float(timing_buf[0]), float(timing_buf[1]))
            
            done_ranks.add(source)
        else:
            comm.recv(source=source, tag=tag)  # Drain unexpected message

    # Prepare worker stats to return
    worker_stats_dict = {}
    
    for worker in range(1, world_size):
        comp_time, comm_time = worker_timing.get(worker, (0.0, 0.0))
        print(f"Worker {worker} processed {worker_chunks[worker]} chunks", flush=True)
        worker_stats_dict[worker] = {
            'chunks_processed': worker_chunks[worker],
            'computation_time': comp_time,
            'communication_time': comm_time,
            'chunk_ids': worker_chunk_ids[worker]
        }

    # Master stats (rank 0 doesn't process chunks in dynamic scheduling)
    worker_stats_dict[0] = {
        'chunks_processed': 0,
        'computation_time': 0.0,
        'communication_time': 0.0,
        'chunk_ids': []
    }

    return full_image, worker_stats_dict


def _master_loop_nonblocking(comm: MPI.Intracomm, config: RunConfig, scheduler: DynamicScheduler, logger = None) -> Tuple[np.ndarray, Dict[int, Dict]]:
    """Master loop for dynamic scheduling with non-blocking communication."""
    full_image = allocate_image(config)
    world_size = comm.Get_size()
    worker_chunks: Dict[int, int] = {rank: 0 for rank in range(world_size)}
    worker_chunk_ids: Dict[int, List[int]] = {rank: [] for rank in range(world_size)}
    worker_timing: Dict[int, Tuple[float, float]] = {}  # comp_time, comm_time
    results_received = 0
    
    # Pre-post receives for work requests from all workers
    request_pool = []
    for worker_rank in range(1, world_size):
        buf = np.array(0, dtype=np.int32)
        req = comm.Irecv(buf, source=worker_rank, tag=REQUEST_TAG)
        request_pool.append((req, worker_rank, buf))
    
    # Pre-post receives for results from all workers
    result_pool = []
    for worker_rank in range(1, world_size):
        meta_buf = np.zeros(2, dtype=np.int32)
        data_buf = np.zeros((config.chunk_size, config.height), dtype=np.float64)
        req_meta = comm.Irecv(meta_buf, source=worker_rank, tag=META_TAG)
        req_data = comm.Irecv(data_buf, source=worker_rank, tag=DATA_TAG)
        result_pool.append((req_meta, req_data, worker_rank, meta_buf, data_buf))
    
    while results_received < config.total_chunks:
        # Check for completed work requests
        for i, (req, worker_rank, buf) in enumerate(request_pool):
            if req.Test():
                # Assign work
                chunk_id = scheduler.request_chunk()
                if chunk_id is not None:
                    assignment = np.array(chunk_id, dtype=np.int32)
                    worker_chunk_ids[worker_rank].append(chunk_id)
                else:
                    assignment = np.array(-1, dtype=np.int32)
                
                comm.Send(assignment, dest=worker_rank, tag=ASSIGN_TAG)
                
                # Re-post receive for next request
                new_buf = np.array(0, dtype=np.int32)
                new_req = comm.Irecv(new_buf, source=worker_rank, tag=REQUEST_TAG)
                request_pool[i] = (new_req, worker_rank, new_buf)
        
        # Check for completed results
        for i, (req_meta, req_data, worker_rank, meta_buf, data_buf) in enumerate(result_pool):
            if req_meta.Test() and req_data.Test():
                # Extract result
                start_row, end_row = meta_buf
                rows = end_row - start_row
                full_image[start_row:end_row, :] = data_buf[:rows, :]
                results_received += 1
                worker_chunks[worker_rank] += 1
                
                # Re-post receives for next result
                new_meta = np.zeros(2, dtype=np.int32)
                new_data = np.zeros((config.chunk_size, config.height), dtype=np.float64)
                new_req_meta = comm.Irecv(new_meta, source=worker_rank, tag=META_TAG)
                new_req_data = comm.Irecv(new_data, source=worker_rank, tag=DATA_TAG)
                result_pool[i] = (new_req_meta, new_req_data, worker_rank, new_meta, new_data)
    
    # Send termination signals to all workers that might still be waiting
    for worker_rank in range(1, world_size):
        termination = np.array(-1, dtype=np.int32)
        comm.Send(termination, dest=worker_rank, tag=ASSIGN_TAG)
    
    # Prepare worker stats to return
    worker_stats_dict = {}
    
    # Wait for workers to finish and receive timing data
    for worker_rank in range(1, world_size):
        num_chunks = comm.recv(source=worker_rank, tag=DONE_TAG)
        
        # Receive timing data
        timing_buf = np.zeros(2, dtype=np.float64)
        comm.Recv(timing_buf, source=worker_rank, tag=DONE_TAG + 1)
        comp_time, comm_time = float(timing_buf[0]), float(timing_buf[1])
        worker_timing[worker_rank] = (comp_time, comm_time)
        
        print(f"Worker {worker_rank} processed {num_chunks} chunks", flush=True)
        worker_stats_dict[worker_rank] = {
            'chunks_processed': num_chunks,
            'computation_time': comp_time,
            'communication_time': comm_time,
            'chunk_ids': worker_chunk_ids[worker_rank]
        }

    # Master stats (rank 0 doesn't process chunks in dynamic scheduling)
    worker_stats_dict[0] = {
        'chunks_processed': 0,
        'computation_time': 0.0,
        'communication_time': 0.0,
        'chunk_ids': []
    }

    return full_image, worker_stats_dict


def _worker_loop_blocking(comm: MPI.Intracomm, config: RunConfig, logger = None) -> Dict:
    """Worker loop for dynamic scheduling with blocking communication."""
    chunks_processed = 0
    total_computation_time = 0.0
    total_communication_time = 0.0
    chunk_ids = []
    
    while True:
        # Communication: Request work
        comm_start = time.time()
        comm.send(None, dest=0, tag=REQUEST_TAG)
        assignment = np.array(0, dtype=np.int32)
        comm.Recv(assignment, source=0, tag=ASSIGN_TAG)
        chunk_id = int(assignment)
        comm_time = time.time() - comm_start
        total_communication_time += comm_time
        
        if chunk_id < 0:
            break
            
        chunk_ids.append(chunk_id)
        
        # Computation: Process chunk
        comp_start = time.time()
        start_row, end_row, chunk = compute_chunk(config, chunk_id)
        comp_time = time.time() - comp_start
        total_computation_time += comp_time
        
        # Communication: Send result
        comm_start = time.time()
        metadata = np.array([start_row, end_row], dtype=np.int32)
        comm.Send(metadata, dest=0, tag=META_TAG)
        comm.Send(chunk, dest=0, tag=DATA_TAG)
        comm_time = time.time() - comm_start
        total_communication_time += comm_time
        
        chunks_processed += 1

    # Send final stats to master
    comm.Send(np.array(chunks_processed, dtype=np.int32), dest=0, tag=DONE_TAG)
    
    # Send timing data to master (using new tags)
    timing_data = np.array([total_computation_time, total_communication_time], dtype=np.float64)
    comm.Send(timing_data, dest=0, tag=DONE_TAG + 1)
    
    # Return stats for this worker
    return {
        'chunks_processed': chunks_processed,
        'computation_time': total_computation_time,
        'communication_time': total_communication_time,
        'chunk_ids': chunk_ids
    }


def _worker_loop_nonblocking(comm: MPI.Intracomm, config: RunConfig, logger = None) -> Dict:
    """Worker loop for dynamic scheduling with non-blocking communication."""
    chunks_processed = 0
    active_sends = []
    total_computation_time = 0.0
    total_communication_time = 0.0
    chunk_ids = []
    
    # Pre-allocate buffer for assignment
    assignment_buf = np.array(-1, dtype=np.int32)
    
    # Initial work request and receive
    comm_start = time.time()
    req_work = comm.Isend(np.array(0, dtype=np.int32), dest=0, tag=REQUEST_TAG)
    req_assign = comm.Irecv(assignment_buf, source=0, tag=ASSIGN_TAG)
    
    while True:
        # Wait for assignment
        req_assign.Wait()
        req_work.Wait()
        comm_time = time.time() - comm_start
        total_communication_time += comm_time
        
        chunk_id = int(assignment_buf)
        
        if chunk_id < 0:
            break

        chunk_ids.append(chunk_id)

        # Compute
        comp_start = time.time()
        start_row, end_row, chunk = compute_chunk(config, chunk_id)
        comp_time = time.time() - comp_start
        total_computation_time += comp_time
        chunks_processed += 1
        
        # Prepare send buffers
        metadata = np.array([start_row, end_row], dtype=np.int32)
        padded_data = np.zeros((config.chunk_size, config.height), dtype=np.float64)
        rows = end_row - start_row
        padded_data[:rows, :] = chunk
        
        # Non-blocking sends (use META_TAG and DATA_TAG to match master expectations)
        comm_start = time.time()
        req_meta = comm.Isend(metadata, dest=0, tag=META_TAG)
        req_data = comm.Isend(padded_data, dest=0, tag=DATA_TAG)
        active_sends.append((req_meta, req_data, metadata, padded_data))
        
        # Request next work
        req_work = comm.Isend(np.array(0, dtype=np.int32), dest=0, tag=REQUEST_TAG)
        assignment_buf = np.array(-1, dtype=np.int32)  # New buffer
        req_assign = comm.Irecv(assignment_buf, source=0, tag=ASSIGN_TAG)
        comm_start = time.time()  # Start timing for next iteration
        
        # Cleanup completed sends
        active_sends = [(rm, rd, m, d) for rm, rd, m, d in active_sends 
                        if not (rm.Test() and rd.Test())]
    
    # Wait for all remaining sends to complete
    comm_start = time.time()
    for req_meta, req_data, _, _ in active_sends:
        req_meta.Wait()
        req_data.Wait()
    comm_time = time.time() - comm_start
    total_communication_time += comm_time
    
    comm.send(chunks_processed, dest=0, tag=DONE_TAG)
    
    # Send timing data to master (using new tags)
    timing_data = np.array([total_computation_time, total_communication_time], dtype=np.float64)
    comm.Send(timing_data, dest=0, tag=DONE_TAG + 1)
    
    # Return stats for this worker
    return {
        'chunks_processed': chunks_processed,
        'computation_time': total_computation_time,
        'communication_time': total_communication_time,
        'chunk_ids': chunk_ids
    }