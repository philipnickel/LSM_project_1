#!/bin/bash
#BSUB -J test_chunk_size
#BSUB -o results/hpc_outputs/test_chunk_size_%J.out
#BSUB -e results/hpc_outputs/test_chunk_size_%J.err
#BSUB -n 16
#BSUB -R "span[ptile=2]"
#BSUB -W 01:00
#BSUB -q hpcintro
#BSUB -R "rusage[mem=1GB]"

module load python3/3.11.8
module load mpi/5.0.8-gcc-13.4.0-binutils-2.44

source ~/myenv/bin/activate

set -e  # Exit on any error

# Parameters
IMAGE_SIZE="1000x1000"
RANKS_LIST=(4 16)
SCHEDULES=("static" "dynamic")
COMMUNICATIONS=("blocking" "nonblocking")
CHUNK_SIZES=(4 16 32 64 128 256 512)

echo "=== Starting MPI benchmark combinations ==="
echo "Image size: ${IMAGE_SIZE}"
echo "Ranks: ${RANKS_LIST[@]}"
echo "Schedules: ${SCHEDULES[@]}"
echo "Communications: ${COMMUNICATIONS[@]}"
echo "Chunk sizes: ${CHUNK_SIZES[@]}"
echo "=========================================="

# Loop over all parameter combinations
for N in "${RANKS_LIST[@]}"; do
  for SCHEDULE in "${SCHEDULES[@]}"; do
    for COMM in "${COMMUNICATIONS[@]}"; do
      for CHUNK in "${CHUNK_SIZES[@]}"; do
        echo ""
        echo ">>> Running: ranks=$N | schedule=$SCHEDULE | comm=$COMM | chunk=$CHUNK <<<"
        echo "---------------------------------------------------------------"

        # Run the computation
        if ! mpirun -n $N python main.py $CHUNK $IMAGE_SIZE --schedule $SCHEDULE --communication $COMM --test-type chunk; then
            echo "ERROR: Computation failed for schedule=$SCHEDULE comm=$COMM chunk=$CHUNK ranks=$N"
            exit 1
        fi

        echo "=== Finished: ranks=$N, schedule=$SCHEDULE, comm=$COMM, chunk=$CHUNK ==="
        echo ""
      done
    done
  done
done

echo "=========================================="
echo "All combinations completed successfully!"
