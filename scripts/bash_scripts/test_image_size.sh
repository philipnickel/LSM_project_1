#!/bin/bash
#BSUB -J test_image_scaling
#BSUB -o results/hpc_outputs/test_image_scaling_%J.out
#BSUB -e results/hpc_outputs/test_image_scaling_%J.err
#BSUB -n 32
#BSUB -R "span[ptile=4]"
#BSUB -W 04:00
#BSUB -q hpcintro
#BSUB -R "rusage[mem=4GB]"

module load python3/3.11.8
module load mpi/5.0.8-gcc-13.4.0-binutils-2.44

source ~/myenv/bin/activate

set -e  # Exit on any error

# Parameters
CHUNK_SIZE=32
RANKS_LIST=(16 32)
IMAGE_SIZES=(1000 3000 5000 10000 15000 20000 25000)
SCHEDULES=("static" "dynamic")
COMMUNICATIONS=("blocking" "nonblocking")

echo "=== Starting MPI image size scaling tests ==="
echo "Chunk size: ${CHUNK_SIZE}"
echo "Ranks: ${RANKS_LIST[@]}"
echo "Image sizes: ${IMAGE_SIZES[@]}"
echo "Schedules: ${SCHEDULES[@]}"
echo "Communications: ${COMMUNICATIONS[@]}"
echo "=========================================="

for N in "${RANKS_LIST[@]}"; do
  for SIZE in "${IMAGE_SIZES[@]}"; do
    IMAGE_SIZE="${SIZE}x${SIZE}"
    for SCHEDULE in "${SCHEDULES[@]}"; do
      for COMM in "${COMMUNICATIONS[@]}"; do
        echo ""
        echo ">>> Running: image=$IMAGE_SIZE | ranks=$N | schedule=$SCHEDULE | comm=$COMM <<<"
        echo "---------------------------------------------------------------"

        # Run the computation
        if ! mpirun -n $N python main.py $CHUNK_SIZE $IMAGE_SIZE --schedule $SCHEDULE --communication $COMM --test-type image; then
            echo "ERROR: Computation failed for image=$IMAGE_SIZE ranks=$N schedule=$SCHEDULE comm=$COMM"
            exit 1
        fi

        echo "=== Finished: image=$IMAGE_SIZE, ranks=$N, schedule=$SCHEDULE, comm=$COMM ==="
        echo ""
      done
    done
  done
done

echo "=========================================="
echo "All image size scaling combinations completed successfully!"
