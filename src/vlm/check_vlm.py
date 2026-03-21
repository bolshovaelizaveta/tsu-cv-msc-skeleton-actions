import os

# =========================
# CUDA FIXES
# =========================

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if torch.cuda.is_available():
    torch.cuda.init()
    torch.cuda.empty_cache()

# =========================
# Imports
# =========================

from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import cv2
import yaml
import glob
import json
import os
import re

# =========================
# Config
# =========================

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
INPUT_DIR = "data/input"

with open("config/vlm_config.yaml", "r") as f:
    vlm_config = yaml.safe_load(f)["vlm"]

print(f"Loading model: {MODEL_ID}")

# =========================
# Device
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    print("CUDA device:", torch.cuda.get_device_name(0))

# =========================
# Processor
# =========================

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

# =========================
# Model
# =========================

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    attn_implementation="sdpa"
)

model = model.to(device)
model.eval()

if device == "cuda":
    print(f"GPU memory used: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# =========================
# Load video
# =========================

videos = glob.glob(os.path.join(INPUT_DIR, "*.mp4"))

if not videos:
    print("No videos found")
    exit()

video_path = videos[0]

print("Processing:", os.path.basename(video_path))

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Could not read frame")
    exit()

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
image = Image.fromarray(frame)

# уменьшаем размер (важно для стабильности)
image = image.resize((448, 448))

# =========================
# Prompt
# =========================

prompt = vlm_config["prompts"]["moderation"]

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    }
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# =========================
# Inputs
# =========================

inputs = processor(
    text=[text],
    images=[image],
    return_tensors="pt"
)

inputs = {k: v.to(device) for k, v in inputs.items()}

# =========================
# Generate
# =========================

print("Generating...")

with torch.inference_mode():

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=vlm_config["generation"]["max_new_tokens"],
        do_sample=False
    )

# =========================
# Decode
# =========================

output_ids = generated_ids[0][inputs["input_ids"].shape[1]:]

response = processor.decode(
    output_ids,
    skip_special_tokens=True
)

print("\n" + "="*50)
print("MODEL RESPONSE:")
print("="*50)
print(response)

# =========================
# Extract JSON
# =========================

json_match = re.search(r"\{.*\}", response, re.DOTALL)

if json_match:
    try:

        result = json.loads(json_match.group())

        print("\n" + "="*50)
        print("PARSED JSON:")
        print("="*50)

        print(json.dumps(result, indent=2, ensure_ascii=False))

        output_file = os.path.splitext(video_path)[0] + "_analysis.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print("Saved to:", output_file)

    except json.JSONDecodeError as e:
        print("JSON parse error:", e)

else:
    print("No JSON found")

# =========================
# Cleanup
# =========================

if device == "cuda":
    torch.cuda.empty_cache()

print("="*50)