#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Get GPU Name
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i $CUDA_VISIBLE_DEVICES)

# Decide queue file based on GPU name
if echo "$GPU_NAME" | grep -qi "A6000"; then
    QUEUE_SUFFIX="A6000"
elif echo "$GPU_NAME" | grep -qi "3090"; then
    QUEUE_SUFFIX="3090"
elif echo "$GPU_NAME" | grep -qi "Blackwell" && echo "$GPU_NAME" | grep -qi "6000"; then
    QUEUE_SUFFIX="pro6k"
else
    QUEUE_SUFFIX="default"
fi

QUEUE_FILE="job_queue_${QUEUE_SUFFIX}.txt"
LOCK_FILE="job_queue_${QUEUE_SUFFIX}.lock"

touch "$QUEUE_FILE"

echo "Worker started on GPU $CUDA_VISIBLE_DEVICES ($GPU_NAME). Waiting for jobs in $QUEUE_FILE..."

while true; do
    job=""
    
    # Critical section to safely pop a job
    exec 200>"$LOCK_FILE"
    flock -x 200
    
    # Remove empty lines
    sed -i '/^[[:space:]]*$/d' "$QUEUE_FILE"
    
    if [ -s "$QUEUE_FILE" ]; then
        # Read the first line
        job=$(head -n 1 "$QUEUE_FILE")
        # Delete the first line
        sed -i '1d' "$QUEUE_FILE"
    fi
    
    flock -u 200
    exec 200>&-

    if [ -n "$job" ]; then
        echo "[GPU $CUDA_VISIBLE_DEVICES] Found job: $job"
        eval "$job"
        echo "[GPU $CUDA_VISIBLE_DEVICES] Job finished."
    else
        # Sleep for 10 seconds before polling again
        sleep 10
    fi
done