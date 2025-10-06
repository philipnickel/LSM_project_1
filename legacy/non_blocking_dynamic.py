"""
MPI Blocking Implementation of Mandelbrot Set Calculation - Dynamic Scheduling
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

_help = f"""\
{sys.argv[0]} [chunk-size] [size widthXheight] [limits xmin:xmax ymin:ymax]

MPI Blocking Implementation of Mandelbrot Set Calculation

Here are some examples:

Call it with a chunk-size of 10
$ mpirun -n 4 {sys.argv[0]} 10

Call it with a chunk-size of 10 and image size of 100 by 500 pixels
$ mpirun -n 4 {sys.argv[0]} 10 100x500

Call it with a chunk-size of 10 and image size of 100 by 500 pixels
spanning the coordinates x \\in 0.1-0.3 and y \\in 0.2-0.3
$ mpirun -n 4 {sys.argv[0]} 10 100x500 0.1:0.3 0.2:0.3
"""

# Check for help request
for h in ("help", "-h", "-help", "--help"):
    if h in sys.argv:
        if rank == 0:
            print(_help)
        sys.exit(0)

# Defaults
chunk_size = 10
image_size = 1000, 1000
xlim = -2.2, 0.75
ylim = -1.3, 1.3

# Override defaults with command line arguments
argv = sys.argv[1:]
if argv:
    chunk_size = int(argv.pop(0))
if argv:
    image_size = tuple(map(int, argv.pop(0).split("x")))
if argv:
    xlim = tuple(map(float, argv.pop(0).split(":")))
if argv:
    ylim = tuple(map(float, argv.pop(0).split(":")))

# Print info (only rank 0)
if rank == 0:
    print(f"""
Calculating the Mandelbrot set with MPI Dynamic Scheduling:

Processes: {size}
{chunk_size = }
{image_size = }
{xlim = }
{ylim = }
""")

# Convert to numpy arrays
image_size = np.asarray(image_size)
xlim = np.asarray(xlim)
ylim = np.asarray(ylim)

# Full image container (master only)
full_image = np.zeros(image_size) if rank == 0 else None

# Constants
xconst = np.diff(xlim)[0] / image_size[0]
yconst = np.diff(ylim)[0] / image_size[1]

# Total number of chunks
total_chunks = (image_size[0] + chunk_size - 1) // chunk_size

# Maximum chunk dimensions
MAX_CHUNK_ROWS = chunk_size
MAX_CHUNK_COLS = image_size[1]


# Communication tags
REQUEST_TAG = 10
ASSIGN_TAG = 11
RESULT_TAG = 20
DONE_TAG = 99

# ====================================
# Mandelbrot calculation function
# ====================================
def calculate_chunk(chunk_id):
    start_row = chunk_id * chunk_size
    end_row = min(start_row + chunk_size, image_size[0])
    chunk_rows = end_row - start_row
    chunk_data = np.zeros((chunk_rows, image_size[1]))

    for local_x in range(chunk_rows):
        global_x = start_row + local_x
        cx = complex(xlim[0] + global_x * xconst, 0)
        for y in range(image_size[1]):
            c = cx + complex(0, ylim[0] + y * yconst)
            z = 0
            for i in range(100):
                z = z*z + c
                if abs(z) > 2:
                    chunk_data[local_x, y] = i
                    break

    print(f"Rank {rank} processed rows {start_row} to {end_row-1}", flush=True)
    return start_row, end_row, chunk_data


# ====================================
# Worker function
# ====================================
def worker_nonblocking():
    chunks_processed = 0
    active_sends = []
    
    # Pre-allocate buffer for assignment
    # -1: Initial placeholder value
    # Worker checks if chunk_id < 0: break to exit loop
    assignment_buf = np.array(-1, dtype=np.int32)
    
    # Initial work request and receive
    req_work = comm.Isend(np.array(0, dtype=np.int32), dest=0, tag=REQUEST_TAG)
    req_assign = comm.Irecv(assignment_buf, source=0, tag=ASSIGN_TAG)
    
    while True:
        # Wait for assignment
        req_assign.Wait()
        req_work.Wait()
        chunk_id = int(assignment_buf)
        
        if chunk_id < 0:
            break

        # Compute
        start_row, end_row, chunk_data = calculate_chunk(chunk_id)
        chunks_processed += 1
        
        # Prepare send buffers
        metadata = np.array([start_row, end_row], dtype=np.int32)
        padded_data = np.zeros((MAX_CHUNK_ROWS, MAX_CHUNK_COLS))
        rows = end_row - start_row
        padded_data[:rows, :] = chunk_data
        
        # Non-blocking sends
        req_meta = comm.Isend(metadata, dest=0, tag=RESULT_TAG)
        req_data = comm.Isend(padded_data, dest=0, tag=RESULT_TAG + 1)
        active_sends.append((req_meta, req_data, metadata, padded_data))
        
        # Request next work
        req_work = comm.Isend(np.array(0, dtype=np.int32), dest=0, tag=REQUEST_TAG)
        assignment_buf = np.array(-1, dtype=np.int32)  # New buffer
        req_assign = comm.Irecv(assignment_buf, source=0, tag=ASSIGN_TAG)
        
        # Cleanup
        active_sends = [(rm, rd, m, d) for rm, rd, m, d in active_sends 
                        if not (rm.Test() and rd.Test())]
    
    # Wait all
    for req_meta, req_data, _, _ in active_sends:
        req_meta.Wait()
        req_data.Wait()
    
    comm.send(chunks_processed, dest=0, tag=DONE_TAG)

# ====================================
# Master function
# ====================================
def master_nonblocking():
    """Non-blocking dynamic master"""
    next_chunk = 0
    results_received = 0
    # active_workers = size - 1
    
    # Pre-post receives for work requests from all workers
    request_pool = []
    for worker_rank in range(1, size):
        buf = np.array(0, dtype=np.int32)
        req = comm.Irecv(buf, source=worker_rank, tag=REQUEST_TAG)
        request_pool.append((req, worker_rank, buf))
    
    # Pre-post receives for results from all workers
    result_pool = []
    for worker_rank in range(1, size):
        meta_buf = np.zeros(2, dtype=np.int32)
        data_buf = np.zeros((MAX_CHUNK_ROWS, MAX_CHUNK_COLS))
        req_meta = comm.Irecv(meta_buf, source=worker_rank, tag=RESULT_TAG)
        req_data = comm.Irecv(data_buf, source=worker_rank, tag=RESULT_TAG + 1)
        result_pool.append((req_meta, req_data, worker_rank, meta_buf, data_buf))
    
    while results_received < total_chunks:
        # Check for completed work requests
        for i, (req, worker_rank, buf) in enumerate(request_pool):
            if req.Test():
                # Assign work
                if next_chunk < total_chunks:
                    assignment = np.array(next_chunk, dtype=np.int32)
                    next_chunk += 1
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
                
                # Re-post receives for next result
                new_meta = np.zeros(2, dtype=np.int32)
                new_data = np.zeros((MAX_CHUNK_ROWS, MAX_CHUNK_COLS))
                new_req_meta = comm.Irecv(new_meta, source=worker_rank, tag=RESULT_TAG)
                new_req_data = comm.Irecv(new_data, source=worker_rank, tag=RESULT_TAG + 1)
                result_pool[i] = (new_req_meta, new_req_data, worker_rank, new_meta, new_data)
    
    # Wait for workers to finish
    for worker_rank in range(1, size):
        num_chunks = comm.recv(source=worker_rank, tag=DONE_TAG)
        print(f"Worker {worker_rank} processed {num_chunks} chunks")

# ====================================
# Run master or worker
# ====================================
if rank == 0:
    master_nonblocking()
else:
    worker_nonblocking()

# ====================================
# Plot final image (rank 0 only)
# ====================================
if rank == 0:
    plt.imshow(full_image.T, extent=np.concatenate([xlim, ylim]), origin='lower')
    plt.xlabel(r"x / Re(p_0)")
    plt.ylabel(r"y / Im(p_0)")
    plt.title(f"Mandelbrot Set - MPI Non-Blocking Dynamic Scheduling ({size} processes)")
    plt.savefig("PLOTS/non_blocking_dynamic_mandelbrot.png", bbox_inches="tight", pad_inches=0)
    print("Mandelbrot image saved to PLOTS/non_blocking_dynamic_mandelbrot.png", flush=True)
