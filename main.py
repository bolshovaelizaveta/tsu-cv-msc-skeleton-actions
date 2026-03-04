import cv2
import time
import json
import numpy as np
from src.detector import PoseDetector
from src.classifier import ActionClassifier
from src.analyzer import GroupAnalyzer
from src.visualizer import Visualizer

def main():
    # Инициализация модулей
    detector = PoseDetector(model_path="models/yolo11n-pose.pt", device='mps')
    classifier = ActionClassifier(model_path="models/stgcn_weights.pt")
    analyzer = GroupAnalyzer()
    visualizer = Visualizer()

    video_path = "data/test_video.mp4"
    cap = cv2.VideoCapture(video_path)

    final_results = []
    fps_history = [] # История скорости обработки 

    print(f"Необходимая скорость: 75 FPS")
    
    frame_idx = 0
    while cap.isOpened():
        start_time = time.time() 
        
        success, frame = cap.read()
        if not success:
            break

        frame_idx += 1
        
        # Детекция 
        people = detector.get_skeleton_data(frame)

        # Анализ действий (пока что заглушка) 
        for p in people:
            p['action'] = classifier.predict({})['action']

        # Анализ групп (пока что заглушка) 
        group_events = analyzer.analyze(people)

        # Расчет скорости (FPS) 
        frame_time = time.time() - start_time
        fps = 1.0 / frame_time if frame_time > 0 else 0
        fps_history.append(fps)

        # Логирование в консоль 
        if frame_idx % 10 == 0: 
            summary = ", ".join([f"ID {p['track_id']}: {p['action']}" for p in people])
            print(f"Кадр {frame_idx} | Скор.: {fps:.1f} FPS | Объектов: {len(people)} | {summary}")

        # Сохранение данных 
        final_results.append({
            "frame": frame_idx,
            "fps": round(fps, 1),
            "people": people
        })

        # Визуализация 
        output_frame = visualizer.draw_frame(frame, people)
        
        cv2.putText(output_frame, f"FPS: {fps:.1f}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        cv2.imshow("KION Video Analytics", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Итоги 
    cap.release()
    cv2.destroyAllWindows()

    avg_fps = np.mean(fps_history) if fps_history else 0
    
    output_json_path = "data/analytics_results.json"
    with open(output_json_path, 'w') as f:
        json.dump({
            "metadata": {
                "avg_fps": round(avg_fps, 1),
                "total_frames": frame_idx
            },
            "data": final_results
        }, f, indent=4)
    
    print(f"Средняя скорость: {avg_fps:.1f} FPS")
    print(f"Результаты сохранены в: {output_json_path}")

if __name__ == "__main__":
    main()