# src/vlm/vlm_api.py
import os
import torch
import yaml
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import json
import re
import base64
import io
from flask import Flask, request, jsonify

app = Flask(__name__)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

with open("configs/vlm/config.yaml", "r") as f:
    config = yaml.safe_load(f)["vlm"]

device = "cuda" if torch.cuda.is_available() and config["model"]["device"] == "cuda" else "cpu"
print(f"device: {device}")

model_path = config["model"].get("local_path", config["model"]["name"])
print(f"Loading model from: {model_path}")

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_path, 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    trust_remote_code=True,
    local_files_only=True
).to(device).eval()

print("Model loaded")

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    
    image_data = data['image'].split(',')[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('RGB')
    
    # Выбор промпта с подсказкой
    suggested = data.get('suggested_action')
    if suggested:
        prompt = config["prompts"]["keyframe_selection"].format(suggested_action=suggested)
    else:
        prompt = config["prompts"]["moderation"]
    
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt}
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = {k: v.to(device) for k, v in processor(text=[text], images=[image], return_tensors="pt").items()}
    with torch.inference_mode():
        output = model.generate(**inputs, **config["generation"])
    
    response = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            return jsonify({'success': True, **result})
        except:
            pass
    
    return jsonify({'success': False})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)