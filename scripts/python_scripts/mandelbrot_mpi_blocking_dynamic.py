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

# Communication tags
REQUEST_TAG = 10
ASSIGN_TAG = 11
RESULT_META_TAG = 20
RESULT_DATA_TAG = 21

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
def worker_loop():
    my_num_chunks = 0
    while True:
        # Request work
        comm.send(None, dest=0, tag=REQUEST_TAG)
        # Receive assigned chunk id
        chunk_id = comm.recv(source=0, tag=ASSIGN_TAG)
        if chunk_id < 0:
            break  # no more work
        # Compute chunk
        start_row, end_row, chunk_data = calculate_chunk(chunk_id)
        # Send metadata and data
        comm.send((start_row, end_row), dest=0, tag=RESULT_META_TAG)
        comm.Send(chunk_data, dest=0, tag=RESULT_DATA_TAG)
        my_num_chunks += 1
    comm.send(my_num_chunks, dest=0, tag=99)

# ====================================
# Master function
# ====================================
def master_loop():
    next_chunk = 0
    results_received = 0
    worker_chunks = np.zeros(size, dtype=int)

    while results_received < total_chunks:
        status = MPI.Status()
        msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()

        if tag == REQUEST_TAG:
            if next_chunk < total_chunks:
                comm.send(next_chunk, dest=source, tag=ASSIGN_TAG)
                next_chunk += 1
            else:
                comm.send(-1, dest=source, tag=ASSIGN_TAG)

        elif tag == RESULT_META_TAG:
            start_row, end_row = msg
            rows = end_row - start_row
            buf = np.empty((rows, image_size[1]), dtype=np.float64)
            comm.Recv(buf, source=source, tag=RESULT_DATA_TAG)
            full_image[start_row:end_row, :] = buf
            results_received += 1
            worker_chunks[source] += 1

    for r in range(1, size):
        print(f"Worker {r} processed {worker_chunks[r]} chunks")

# ====================================
# Run master or worker
# ====================================
if rank == 0:
    master_loop()
else:
    worker_loop()

# ====================================
# Plot final image (rank 0 only)
# ====================================
if rank == 0:
    plt.imshow(full_image.T, extent=np.concatenate([xlim, ylim]), origin='lower')
    plt.xlabel(r"x / Re(p_0)")
    plt.ylabel(r"y / Im(p_0)")
    plt.title(f"Mandelbrot Set - MPI Dynamic Scheduling ({size} processes)")
    plt.savefig("PLOTS/dynamic_mandelbrot.png", bbox_inches="tight", pad_inches=0)
    print("Mandelbrot image saved to PLOTS/dynamic_mandelbrot.png", flush=True)
