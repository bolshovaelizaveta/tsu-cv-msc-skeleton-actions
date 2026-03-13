import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any

class PoseDetector:
    def __init__(self, model_path: str, device: str = 'mps', conf: float = 0.5):
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf
        self.min_area = 15000 

    def get_skeleton_data(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        results = self.model.track(
            frame, 
            persist=True, 
            device=self.device, 
            conf=self.conf, 
            verbose=False
        )
        
        r = results[0]
        detections = []

        if r.boxes is not None and r.boxes.id is not None:
            track_ids = r.boxes.id.int().cpu().tolist()
            bboxes = r.boxes.xyxy.cpu().numpy()
            keypoints = r.keypoints.data.cpu().numpy()

            for i, track_id in enumerate(track_ids):
                box = bboxes[i]
                area = (box[2] - box[0]) * (box[3] - box[1])
                if area < self.min_area:
                    continue

                detections.append({
                    "track_id": track_id,
                    "bbox": box.tolist(),
                    "keypoints": keypoints[i].tolist() 
                })
        
        return detections