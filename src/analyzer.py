import math
import numpy as np
from typing import List, Dict, Any

class GroupAnalyzer:
    def _get_distance(self, pt1, pt2):
        return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

    def analyze(self, people_data: List[Dict[str, Any]]) -> List[str]:
        events = []
        n = len(people_data)
        centers = []
        for p in people_data:
            box = p['bbox']
            centers.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))

        # Формирование фигур
        if n == 3: events.append("triangle")
        elif n >= 4:
            centroid_x = sum(c[0] for c in centers) / n
            centroid_y = sum(c[1] for c in centers) / n
            distances = [self._get_distance(c, (centroid_x, centroid_y)) for c in centers]
            events.append("circle" if np.var(distances) < 1500 else "meeting")

        # Взаимодействия
        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = people_data[i], people_data[j]
                dist = self._get_distance(centers[i], centers[j])
                
                # Порог близости на основе ширины плеч
                s_width = self._get_distance(p1['keypoints'][5], p1['keypoints'][6])
                
                # Объятия (Очень близко, стоят на месте)
                if dist < s_width * 1.2:
                    if p1.get('action') == 'standing' and p2.get('action') == 'standing':
                        events.append("hugging")
                    else:
                        events.append("fighting") # Активное движение вплотную
                
                # Рукопожатие (Сближение кистей)
                p1_wrists = [p1['keypoints'][9], p1['keypoints'][10]]
                p2_wrists = [p2['keypoints'][9], p2['keypoints'][10]]
                for w1 in p1_wrists:
                    for w2 in p2_wrists:
                        if w1[2] > 0.3 and w2[2] > 0.3 and self._get_distance(w1, w2) < 40:
                            events.append("handshake")
                            break
        return list(set(events))