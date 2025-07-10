#!/bin/bash

# RunPod Serverless Entrypoint Script
echo "Starting RunPod MultiTalk serverless handler..."

# Check if models exist
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model path $MODEL_PATH does not exist!"
    echo "Please ensure network volume is mounted correctly."
    exit 1
fi

# List available models
echo "Available models in $MODEL_PATH:"
ls -la $MODEL_PATH/

# Start the Python handler
python -u /app/src/handler.py