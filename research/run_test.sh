#!/bin/bash

# =========================
# Run Model Test Script
# =========================

cd /app

MODEL_PATH="models/Qwen/Qwen3-VL-4B-Instruct"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model not found at $MODEL_PATH"
    echo "Please run ./check_and_load_model.sh first"
    exit 1
fi

# Check if test image exists, if not create a dummy
if [ ! -f "test.jpg" ]; then
    echo "Creating test image..."
    python3 -c "from PIL import Image; img = Image.new('RGB', (448, 448), color='blue'); img.save('test.jpg'); print('Test image created: test.jpg')"
fi

# Run test
echo "Running model test..."
python3 scripts/vlm/test_model.py