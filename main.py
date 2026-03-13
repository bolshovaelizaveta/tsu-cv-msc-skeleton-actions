import cv2
import time
import json
import glob
import os
import numpy as np
from src.detector import PoseDetector
from src.classifier import ActionClassifier
from src.analyzer import GroupAnalyzer
from src.visualizer import Visualizer

SCREENSHOT_DIR = "results/screenshots"
SAVE_INTERVAL = 30  
last_saved_frame = {}

def get_video_path(folder="data"):
    videos = glob.glob(f"{folder}/*.mp4")
    if not videos:
        raise FileNotFoundError(f"В папке {folder} нет файлов mp4.")
    return videos[-1]

def main():
    detector = PoseDetector(model_path="yolo11m-pose.pt", device='mps')
    classifier = ActionClassifier()
    analyzer = GroupAnalyzer()
    visualizer = Visualizer()

    video_path = get_video_path()
    cap = cv2.VideoCapture(video_path)
    
    w, h, fps_in = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    out = cv2.VideoWriter('results/final_analytics.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps_in, (w, h))

    # Словарь для каждого найденного действия
    stats_frames = {} 
    
    frame_idx = 0
    FRAME_SKIP = 5 

    print(f"Запуск аналитики: {video_path}")

    while cap.isOpened():
        start_time = time.time()
        success, frame = cap.read()
        if not success:
            break

        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            continue
        
        people = detector.get_skeleton_data(frame)
        
        # Анализ индивидуальных действий и подсчет экранного времени
        current_frame_actions = []
        for p in people:
            action = classifier.predict(p)
            p['action'] = action
            current_frame_actions.append(action)
            
            # Накапливаем кадры для каждого действия
            stats_frames[action] = stats_frames.get(action, 0) + 1
            
            # Скриншоты действий 
            if action not in ['standing', 'buffering', 'unknown']:
                if frame_idx - last_saved_frame.get(action, 0) > SAVE_INTERVAL:
                    path = os.path.join(SCREENSHOT_DIR, action)
                    os.makedirs(path, exist_ok=True)
                    cv2.imwrite(os.path.join(path, f"frame_{frame_idx}.jpg"), frame)
                    last_saved_frame[action] = frame_idx

        # Анализ групп и скриншоты событий
        group_events = analyzer.analyze(people)
        for event in group_events:
            stats_frames[event] = stats_frames.get(event, 0) + 1
            
            if frame_idx - last_saved_frame.get(event, 0) > SAVE_INTERVAL:
                path = os.path.join(SCREENSHOT_DIR, event)
                os.makedirs(path, exist_ok=True)
                cv2.imwrite(os.path.join(path, f"group_{frame_idx}.jpg"), frame)
                last_saved_frame[event] = frame_idx

        # Визуализация
        output_frame = visualizer.draw_frame(frame, people)

        # Отрисовка DASHBOARD
        overlay = output_frame.copy()
        cv2.rectangle(overlay, (w - 350, 0), (w, 300), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, output_frame, 0.4, 0, output_frame)
        
        cv2.putText(output_frame, "KION ANALYTICS", (w - 330, 40), 0, 0.7, (255, 255, 255), 2)
        
        # Выводим топ самых частых действий в этом видео
        y_pos = 80
        sorted_stats = sorted(stats_frames.items(), key=lambda x: x[1], reverse=True)
        
        for action_name, frames_count in sorted_stats[:6]:
            duration = (frames_count * FRAME_SKIP) / fps_in
            color = (0, 255, 0) if action_name not in ['smoking', 'fighting'] else (0, 0, 255)
            
            text = f"{action_name.capitalize()}: {duration:.1f}s"
            cv2.putText(output_frame, text, (w - 330, y_pos), 0, 0.6, color, 1)
            y_pos += 30

        # Групповые события внизу
        for i, event in enumerate(group_events):
            cv2.putText(output_frame, f"EVENT: {event.upper()}", (20, h - 40 - i*30), 0, 0.8, (0, 0, 255), 2)

        # FPS
        proc_fps = (1.0 / (time.time() - start_time)) * FRAME_SKIP
        cv2.putText(output_frame, f"FPS: {proc_fps:.1f}", (20, 50), 0, 1.2, (0, 255, 255), 3)

        out.write(output_frame)
        cv2.imshow("KION", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Скриншоты в {SCREENSHOT_DIR}")

if __name__ == "__main__":
    main()