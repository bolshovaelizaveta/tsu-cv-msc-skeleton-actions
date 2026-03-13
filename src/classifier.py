import math
from collections import defaultdict

class ActionClassifier:
    def __init__(self):
        self.history = defaultdict(list)
        self.window_size = 15

    def _get_distance(self, pt1, pt2):
        return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

    def predict(self, person_data: dict) -> str:
        track_id = person_data.get('track_id')
        kpts = person_data.get('keypoints', [])
        
        if not track_id or len(kpts) < 17:
            return "unknown"

        self.history[track_id].append(kpts)
        if len(self.history[track_id]) > self.window_size:
            self.history[track_id].pop(0)

        if len(self.history[track_id]) < 5:
            return "buffering"

        curr_kpts = self.history[track_id][-1]
        past_kpts = self.history[track_id][0]

        l_shoulder, r_shoulder = curr_kpts[5], curr_kpts[6]
        shoulder_width = self._get_distance(l_shoulder, r_shoulder)
        if shoulder_width < 10: shoulder_width = 50

        # Анализ прыжков (Резкое изменение Y координат лодыжек)
        curr_l_ankle, past_l_ankle = curr_kpts[15], past_kpts[15]
        if curr_l_ankle[2] > 0.4 and past_l_ankle[2] > 0.4:
            # В OpenCV Y растет вниз. Прыжок вверх - это уменьшение Y.
            y_diff = past_l_ankle[1] - curr_l_ankle[1]
            if y_diff > (shoulder_width * 0.5): 
                return "jumping"

        # Анализ ходьбы
        if curr_l_ankle[2] > 0.3 and past_l_ankle[2] > 0.3:
            movement_ratio = self._get_distance(curr_l_ankle, past_l_ankle) / shoulder_width
            if movement_ratio > 0.3: 
                return "walking"

        # Анализ курения (Биомеханика локтя)
        curr_nose = curr_kpts[0]
        l_elbow, r_elbow = curr_kpts[7], curr_kpts[8]
        if curr_nose[2] > 0.3:
            if r_shoulder[2] > 0.3 and r_elbow[2] > 0.3:
                if (r_elbow[1] - r_shoulder[1]) < (shoulder_width * 0.5):
                    return "smoking"
            if l_shoulder[2] > 0.3 and l_elbow[2] > 0.3:
                if (l_elbow[1] - l_shoulder[1]) < (shoulder_width * 0.5):
                    return "smoking"

        # Анализ сидения
        l_hip, l_knee = curr_kpts[11], curr_kpts[13]
        if l_hip[2] > 0.3 and l_knee[2] > 0.3:
            if abs(l_hip[1] - l_knee[1]) < (shoulder_width * 0.7):
                return "sitting"

        return "standing"