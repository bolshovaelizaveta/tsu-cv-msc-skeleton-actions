import math
import numpy as np
from typing import List, Dict, Any

class GroupAnalyzer:
    def _get_distance(self, pt1, pt2):
        return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

    def analyze(self, people_data: List[Dict[str, Any]]) -> List[str]:
        events = []
        n = len(people_data)
        
        if n == 0:
            return events

        centers = []
        keypoints_list = []
        
        for p in people_data:
            box = p.get('bbox', [0, 0, 0, 0])
            centers.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
            keypoints_list.append(p.get('keypoints', []))

        # 1. Проверка геометрических фигур(Круг / Треугольник)
        if n == 3:
            events.append("triangle_formation")
            
        elif n >= 4:
            # Считаем центр масс всех людей
            centroid_x = sum(c[0] for c in centers) / n
            centroid_y = sum(c[1] for c in centers) / n
            
            # Считаем расстояние от каждого человека до центра
            distances = [self._get_distance(c, (centroid_x, centroid_y)) for c in centers]
            
            # Если разброс (дисперсия) расстояний маленький - они стоят ровным кругом
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            
            # Коэффициент вариации < 25% означает ровную фигуру (круг)
            if mean_dist > 0 and (std_dist / mean_dist) < 0.25:
                events.append("circle_formation")

        # 2. Перетягивание каната (Tug of War)
        if n >= 4:
            # Проверяем, выстроены ли люди примерно в одну горизонтальную линию
            y_coords = [c[1] for c in centers]
            if np.std(y_coords) < 150: # Люди примерно на одной линии по высоте кадра
                
                # Сортируем людей слева направо по X
                sorted_people = sorted(zip(centers, keypoints_list), key=lambda item: item[0][0])
                
                left_team_leaning_left = 0
                right_team_leaning_right = 0
                
                mid_point = n // 2
                
                # Анализируем векторы тел 
                # Точки YOLO: 5,6 - плечи, 11,12 - бедра
                for i, (center, kpts) in enumerate(sorted_people):
                    if len(kpts) < 13: continue
                    
                    # Центр груди и центр таза
                    chest_x = (kpts[5][0] + kpts[6][0]) / 2
                    hip_x = (kpts[11][0] + kpts[12][0]) / 2
                    
                    # Уверенность точек
                    conf = min(kpts[5][2], kpts[11][2])
                    if conf < 0.4: continue
                    
                    if i < mid_point: # Левая команда
                        if chest_x < hip_x: # Грудная клетка левее бедер - отклонен назад
                            left_team_leaning_left += 1
                    else: # Правая команда
                        if chest_x > hip_x: # Грудная клетка правее бедер - отклонен назад
                            right_team_leaning_right += 1
                            
                # Если хотя бы по одному человеку в командах явно отклонены назад 
                if left_team_leaning_left >= 1 and right_team_leaning_right >= 1:
                    events.append("tug_of_war_candidate")

        # 3. Митинг / толпа (Gathering/Rally)
        if n >= 6:
            # Если людей много и они находятся близко друг к другу
            events.append("rally_candidate")

        return list(set(events))