#!/bin/bash
#BSUB -J test_rank_scaling
#BSUB -o results/hpc_outputs/test_rank_scaling_%J.out
#BSUB -e results/hpc_outputs/test_rank_scaling_%J.err
#BSUB -n 100
#BSUB -R "span[ptile=4]"
#BSUB -W 02:00
#BSUB -q hpcintro
#BSUB -R "rusage[mem=2GB]"

module load python3/3.11.8
module load mpi/5.0.8-gcc-13.4.0-binutils-2.44

source ~/myenv/bin/activate

set -e  # Exit on any error

# Parameters
CHUNK_SIZE=32
IMAGE_SIZES=("1000x1000" "5000x5000")
RANKS_LIST=(2 4 8 16 32 48 64 80 100)
SCHEDULES=("static" "dynamic")
COMMUNICATIONS=("blocking" "nonblocking")

echo "=== Starting MPI rank scaling tests ==="
echo "Chunk size: ${CHUNK_SIZE}"
echo "Image sizes: ${IMAGE_SIZES[@]}"
echo "Ranks: ${RANKS_LIST[@]}"
echo "Schedules: ${SCHEDULES[@]}"
echo "Communications: ${COMMUNICATIONS[@]}"
echo "=========================================="


for IMAGE_SIZE in "${IMAGE_SIZES[@]}"; do
  for N in "${RANKS_LIST[@]}"; do
    for SCHEDULE in "${SCHEDULES[@]}"; do
      for COMM in "${COMMUNICATIONS[@]}"; do
        echo ""
        echo ">>> Running: image=$IMAGE_SIZE | ranks=$N | schedule=$SCHEDULE | comm=$COMM <<<"
        echo "---------------------------------------------------------------"

        # Run the computation
        if ! mpirun -n $N python main.py $CHUNK_SIZE $IMAGE_SIZE --schedule $SCHEDULE --communication $COMM --test-type ranks; then
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
echo "All rank scaling combinations completed successfully!"
