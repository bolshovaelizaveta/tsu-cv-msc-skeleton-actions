# src/vlm_client.py
import requests
import cv2
import base64

class VLMClient:
    def __init__(self, host='vlm', port=5000):
        self.url = f'http://{host}:{port}/analyze'
    
    def analyze(self, frame, suggested_action=None):
        _, buffer = cv2.imencode('.jpg', frame)
        b64 = base64.b64encode(buffer).decode()
        
        try:
            resp = requests.post(self.url, json={
                'image': f'data:image/jpeg;base64,{b64}',
                'suggested_action': suggested_action
            }, timeout=10)
            
            if resp.status_code == 200:
                return resp.json()
        except:
            pass
        return None