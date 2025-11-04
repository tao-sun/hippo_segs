#!/bin/bash

# Define folds and views
folds=(2 3 4 5)
views=("sagittal" "coronal" "axial")

mkdir -p logs

# Get a single timestamp when the script starts
timestamp=$(date +"%Y%m%d_%H%M%S")

# Loop over folds (run in parallel)
for view in "${views[@]}"; do
{
    echo "=== Starting view ${view} at $(date) ==="
    for fold in "${folds[@]}"; do
        echo "Running fold=${fold}, view=${view} at $(date)"
        log_name="logs/snn_deep_fptt_${fold}_${view}_${timestamp}.log"
        python -u snn_fptt.py --val-fold "$fold" --view "$view" --model "deep" --lr "0.0005" > "$log_name" 2>&1
    done
    echo "=== Finished view ${view} at $(date) ==="
} &
done

# Wait for all background folds to finish
wait
echo "All folds completed at $(date)."
