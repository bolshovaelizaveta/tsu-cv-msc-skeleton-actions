import os
import torch
import yaml
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import json
import re

# CUDA оптимизации
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Загрузка конфига
with open("configs/vlm/config.yaml", "r") as f:
    config = yaml.safe_load(f)["vlm"]

# Настройки
device = "cuda" if torch.cuda.is_available() and config["model"]["device"] == "cuda" else "cpu"
print(f"device: {device}")

model_path = config["model"].get("local_path", config["model"]["name"])

print(f"Loading model from: {model_path}")

# Загрузка модели
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_path, 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    trust_remote_code=True,
    local_files_only=True
).to(device).eval()

# Загрузка тестового изображения
test_image_path = config.get("test_image", "scripts/vlm/test.jpg")
if not os.path.exists(test_image_path):
    print(f"Test image not found, creating red square")
    image = Image.new('RGB', (448, 448), 'red')
else:
    image = Image.open(test_image_path).convert('RGB').resize((448, 448))

# Подготовка промпта
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": config["prompts"]["moderation"]}
]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Инференс
inputs = {k: v.to(device) for k, v in processor(text=[text], images=[image], return_tensors="pt").items()}
with torch.inference_mode():
    output = model.generate(**inputs, **config["generation"])

# Декодирование
response = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("\nMODEL RESPONSE:\n", response)


# Очистка
torch.cuda.empty_cache() if device == "cuda" else None
print("\nTest completed")