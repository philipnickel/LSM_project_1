

# Run for some different stuff 
N=(1 4 8)
CHUNK_SIZE=(128)
Size=("500x500") 
schedulers=("static" "dynamic")
communication=("blocking" "nonblocking")

for n in "${N[@]}"; do
  for chunk in "${CHUNK_SIZE[@]}"; do
    for size in "${Size[@]}"; do
      for schedule in "${schedulers[@]}"; do
        for comm in "${communication[@]}"; do
          mpirun -n $n python main.py $chunk $size --schedule $schedule --communication $comm --log-experiment
          echo "Completed: N=$n, chunk_size=$chunk, size=$size, schedule=$schedule, communication=$comm"
        done
      done
    done
  done
done
