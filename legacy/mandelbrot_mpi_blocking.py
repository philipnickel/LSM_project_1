"""
MPI Blocking Implementation of Mandelbrot Set Calculation
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

for h in ("help", "-h", "-help", "--help"):
    if h in sys.argv:
        if rank == 0:
            print(_help)
        sys.exit(0)

# First we define all the defaults, then we let the arguments overwrite them.
chunk_size = 10
image_size = 1000, 1000
xlim = -2.2, 0.75
ylim = -1.3, 1.3

# Now grab the arguments
argv = sys.argv[1:]
if argv:
    chunk_size = int(argv.pop(0))
if argv:
    image_size = tuple(map(int, argv.pop(0).split("x")))
if argv:
    xlim = tuple(map(float, argv.pop(0).split(":")))
if argv:
    ylim = tuple(map(float, argv.pop(0).split(":")))

# Print info 
if rank == 0:
    print(f"""\
Calculating the Mandelbrot set with MPI Blocking implementation:

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

# Dimensions of the image
full_image = np.zeros(image_size)

# Calculate constants
xconst = np.diff(xlim)[0] / image_size[0]
yconst = np.diff(ylim)[0] / image_size[1]

#===================================
# Blocking MPI with chunk-size based distribution
#===================================

# Calculate total number of chunks
total_chunks = (image_size[0] + chunk_size - 1) // chunk_size  



# Static scheduler: each rank gets assigned chunks 
# Chunk i is assigned to rank (i % size)
my_chunks = []
for chunk_id in range(total_chunks):
    if chunk_id % size == rank:
        my_chunks.append(chunk_id)

# Calculate Mandelbrot set for assigned chunks
def calculate_chunk(chunk_id):
    """Calculate Mandelbrot set for a single chunk of rows"""
    start_row = chunk_id * chunk_size
    end_row = min(start_row + chunk_size, image_size[0])
    chunk_rows = end_row - start_row

    chunk_data = np.zeros((chunk_rows, image_size[1]))

    print(f"Rank {rank} processed rows {start_row} to {end_row-1}")



    for local_x in range(chunk_rows):
        global_x = start_row + local_x
        cx = complex(xlim[0] + global_x * xconst, 0)
        for y in range(image_size[1]):
            c = cx + complex(0, ylim[0] + y * yconst)
            z = 0
            for i in range(100):
                z = z*z + c
                if np.abs(z) > 2:
                    chunk_data[local_x, y] = i
                    break

    return start_row, end_row, chunk_data

# Process all assigned chunks
results = []
for chunk_id in my_chunks:
    results.append(calculate_chunk(chunk_id))

# Print info on what wark each rank processed

print(f"Rank {rank} processed total number of chunks: {len(my_chunks)}")
# Gather results at rank 0
if rank == 0:
    # Store own results
    for start_row, end_row, chunk_data in results:
        full_image[start_row:end_row, :] = chunk_data

    # Receive results from all other ranks using blocking receive
    for source_rank in range(1, size):
        # Receive number of chunks from this rank
        num_chunks = np.array(0, dtype=np.int32)
        comm.Recv(num_chunks, source=source_rank, tag=0)

        # Receive each chunk
        for _ in range(num_chunks):
            # Receive chunk metadata (start_row, end_row)
            metadata = np.zeros(2, dtype=np.int32)
            comm.Recv(metadata, source=source_rank, tag=1)
            start_row, end_row = metadata

            # Receive chunk data
            chunk_rows = end_row - start_row
            chunk_data = np.zeros((chunk_rows, image_size[1]))
            comm.Recv(chunk_data, source=source_rank, tag=2)

            full_image[start_row:end_row, :] = chunk_data
else:
    # Worker ranks: send results to rank 0
    num_chunks = np.array(len(results), dtype=np.int32)
    comm.Send(num_chunks, dest=0, tag=0)

    for start_row, end_row, chunk_data in results:
        # Send metadata
        metadata = np.array([start_row, end_row], dtype=np.int32)
        comm.Send(metadata, dest=0, tag=1)

        # Send chunk data
        comm.Send(chunk_data, dest=0, tag=2)

# rank 0 creates the plot
if rank == 0:
    # Increase font-size
    plt.imshow(full_image.T, extent=np.concatenate([xlim, ylim]))
    plt.xlabel(r"x / Re(p_0)")
    plt.ylabel(r"y / Im(p_0)")
    plt.title(f"Mandelbrot Set - MPI Blocking Implementation ({size} processes)")

    plt.savefig("Plots/blocking_mandelbrot.pdf", bbox_inches="tight", pad_inches=0)
