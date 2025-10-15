#!/bin/bash

# Define folds and views
folds=(2 3 4 5)
views=("sagittal" "coronal" "axial")

mkdir -p logs

# Get a single timestamp when the script starts
timestamp=$(date +"%Y%m%d_%H%M%S")

# Loop over folds (run in parallel)
for fold in "${folds[@]}"; do
{
    echo "=== Starting fold ${fold} at $(date) ==="
    for view in "${views[@]}"; do
        echo "Running fold=${fold}, view=${view} at $(date)"
        log_name="logs/snn_fptt_${fold}_${view}_${timestamp}.log"
        python -u snn_fptt.py --val-fold "$fold" --view "$view" > "$log_name" 2>&1
    done
    echo "=== Finished fold ${fold} at $(date) ==="
} &
done

# Wait for all background folds to finish
wait
echo "All folds completed at $(date)."
