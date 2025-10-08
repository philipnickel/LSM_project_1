#!/bin/bash
# Automated testing script for Mandelbrot MPI experiments

# Configuration
SIZES=("100x100" "200x200")
XLIMS=("-2.2:0.4")
YLIMS=("-1.3:1.3" "-0.1:2.5")

NUM_PROCESSORS=(4 8)
CHUNK_SIZES=(10 20)

# Setup logging
LOG_DIR="logs"
mkdir -p $LOG_DIR
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/experiment_$TIMESTAMP-log.txt"
SUMMARY_FILE="$LOG_DIR/experiment_$TIMESTAMP-summary.txt"
ERROR_FILE="$LOG_DIR/experiment_$TIMESTAMP-errors.txt"

passed=0
failed=0

make clean

# Generate baselines
echo "Generating baselines..." | tee -a $LOG_FILE
for size in "${SIZES[@]}"; do
    for xlim in "${XLIMS[@]}"; do
        for ylim in "${YLIMS[@]}"; do
            echo "Generating baseline: size=$size xlim=$xlim ylim=$ylim" | tee -a $LOG_FILE
            make baseline SIZE=$size XLIM=$xlim YLIM=$ylim 2>&1 | tee -a $LOG_FILE
            [ $? -ne 0 ] && echo "Baseline failed: $size $xlim $ylim" | tee -a $LOG_FILE $ERROR_FILE && exit 1
        done
    done
done

# Run experiments
total=$((${#NUM_PROCESSORS[@]} * ${#CHUNK_SIZES[@]}))
count=0

for n in "${NUM_PROCESSORS[@]}"; do
    for chunk in "${CHUNK_SIZES[@]}"; do
        count=$((count + 1))
        echo "[$count/$total] N=$n CHUNK=$chunk" | tee -a $LOG_FILE
        
        output=$(make test-all N=$n CHUNK_SIZE=$chunk 2>&1)
        echo "$output" | tee -a $LOG_FILE
        
        if [ $? -eq 0 ]; then
            passed=$((passed + 1))
        else
            failed=$((failed + 1))
            echo "FAILED: N=$n CHUNK=$chunk" >> $ERROR_FILE
            echo "$output" >> $ERROR_FILE
            echo "" >> $ERROR_FILE
        fi
        
        sleep 2
    done
done

# Write summary
echo "$passed/$total passed" | tee $SUMMARY_FILE
echo "Log: $LOG_FILE"
echo "Summary: $SUMMARY_FILE"
# Write when error occurs
[ $failed -gt 0 ] && echo "Errors: $ERROR_FILE"