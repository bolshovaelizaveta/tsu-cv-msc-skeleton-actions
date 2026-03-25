# scripts/test_vlm.py
import sys
import cv2
sys.path.insert(0, '.')
from src.vlm.vlm_client import VLMClient

client = VLMClient()
frame = cv2.imread(sys.argv[1])
result = client.analyze(frame)
print(result)