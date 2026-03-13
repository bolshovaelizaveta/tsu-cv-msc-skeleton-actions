import cv2
import numpy as np
from typing import List, Dict, Any

class Visualizer:
    def __init__(self):
        self.skeleton_edges = [
            (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), 
            (5, 11), (6, 12), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
        ]

    def draw_frame(self, frame: np.ndarray, people: List[Dict[str, Any]]) -> np.ndarray:
        canvas = frame.copy()

        for person in people:
            kpts = np.array(person['keypoints'])
            track_id = person['track_id']
            action = person.get('action', 'unknown')

            # Рисуем линии 
            for edge in self.skeleton_edges:
                pt1, pt2 = kpts[edge[0]], kpts[edge[1]]
                # Проверяем уверенность (>0.3) И чтобы координаты не были нулевыми (x>5, y>5)
                if (pt1[2] > 0.3 and pt2[2] > 0.3 and 
                    pt1[0] > 5 and pt1[1] > 5 and pt2[0] > 5 and pt2[1] > 5):
                    cv2.line(canvas, (int(pt1[0]), int(pt1[1])), 
                             (int(pt2[0]), int(pt2[1])), (0, 255, 0), 2)

            # Рисуем точки
            for pt in kpts:
                if pt[2] > 0.3 and pt[0] > 5 and pt[1] > 5:
                    cv2.circle(canvas, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)

            # Красивый вывод текста 
            x1, y1 = int(person['bbox'][0]), int(person['bbox'][1])
            
            text_color = (0, 0, 255) if action in ['smoking', 'fighting'] else (255, 255, 0)
            
            # Рисуем черную подложку для текста 
            text = f"ID: {track_id} | {action}"
            cv2.putText(canvas, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4) 
            cv2.putText(canvas, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2) 
            
        return canvas