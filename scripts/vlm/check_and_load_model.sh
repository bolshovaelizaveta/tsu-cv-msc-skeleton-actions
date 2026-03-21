#!/bin/bash

# =========================
# Model Check and Download Script
# =========================

MODEL_DIR="models/Qwen/Qwen3-VL-4B-Instruct"
MODEL_ID="Qwen/Qwen3-VL-4B-Instruct"

# Check if model exists
if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/model.safetensors.index.json" ]; then
    echo "Model already exists in $MODEL_DIR"
    exit 0
else
    echo "Model not found. Downloading..."
    
    # Create directory if it doesn't exist
    mkdir -p models
    
    # Download model using huggingface_hub
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$MODEL_ID',
    local_dir='$MODEL_DIR',
    local_dir_use_symlinks=False
)
print('Model downloaded successfully to $MODEL_DIR')
"
    
    if [ $? -eq 0 ]; then
        echo "Model download completed"
        exit 0
    else
        echo "Error downloading model"
        exit 1
    fi
fi