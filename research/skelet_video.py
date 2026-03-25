import cv2
import pandas as pd
from ultralytics import YOLO
import os

# Загружаем CSV с результатами 
results = pd.read_csv('benchmark_report.csv')
model = YOLO('yolo11n-pose.pt')

os.makedirs('final_videos', exist_ok=True)

for index, row in results.iterrows():
    video_name = row['file']
    label = row['pred']
    video_path = f"to_fix/{video_name}"
    
    if not os.path.exists(video_path): continue
    
    cap = cv2.VideoCapture(video_path)
    w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    out = cv2.VideoWriter(f'final_videos/{video_name}', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    print(f"Рисую скелеты на {video_name}...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Детектим позы и рисуем их
        res = model(frame, verbose=False)[0]
        frame = res.plot(labels=False, boxes=False) # Рисует ТОЛЬКО скелеты
        
        # Накладываем лейбл из бенчмарка
        color = (0, 255, 0) if row['gt'] == row['pred'] else (0, 0, 255)
        cv2.putText(frame, f"ACTION: {label.upper()}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        out.write(frame)
    
    cap.release()
    out.release()

print("Готово! Все видео со скелетами в папке final_videos")