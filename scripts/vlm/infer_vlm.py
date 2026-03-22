import os
import sys
import cv2
import torch
import json
import yaml
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import PoseDetector
from src.sequence_buffer_3d import SequenceBuffer3D
from src.skeleton_adapter_stgcnpp import SkeletonAdapterSTGCNPP
from src.classifiers.stgcnpp_classifier import STGCNPPClassifier
from src.utils.ntu60_labels import NTU60_CLASSES
from src.utils.action_mapping import map_ntu_to_target, resolve_target_class # Добавили resolve
from src.vlm.vlm_client import VLMClient
from src.analyzer import GroupAnalyzer 

POSE_MODEL = "models/yolo11m-pose.pt"
STGCNPP_CONFIG = "configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py"
STGCNPP_CHECKPOINT = "models/stgcnpp_ntu60_xsub.pth"
WINDOW_SIZE = 32
YOLO_SKIP = 5

with open("configs/vlm/config.yaml", "r") as f:
    vlm_config = yaml.safe_load(f)["vlm"]

os.makedirs(vlm_config.get("output_dir", "results/vlm"), exist_ok=True)

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/infer_with_vlm_simplified.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = PoseDetector(POSE_MODEL, device)
    adapter = SkeletonAdapterSTGCNPP()
    buffer = SequenceBuffer3D(WINDOW_SIZE)
    classifier = STGCNPPClassifier(STGCNPP_CONFIG, STGCNPP_CHECKPOINT, device)
    vlm = VLMClient()
    
    # Инициализация анализатора и таймеров
    group_analyzer = GroupAnalyzer()
    last_vlm_time = 0.0
    VLM_COOLDOWN = 5.0 
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # VLM анализ первого кадра 
    middle_frame_num = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_num)
    ret, middle_frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    width, height = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(f"results/{video_name}_output.mp4", 
                            cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    frame_num = 0
    raw_ntu_predictions = [] # Список для хранения всех предсказаний для финала
    vlm_result = None
    
    try:
        vlm_result = vlm.analyze(middle_frame)
    except:
        pass
    
    start_time = time.time()
    last_fps_time = start_time
    last_fps_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_num % YOLO_SKIP == 0:
            persons = detector.get_skeleton_data(frame)
        else:
            persons = []
            
        group_events = group_analyzer.analyze(persons) if persons else []
        current_time = time.time()
        
        for p in persons:
            track_id = p.get('track_id', 0)
            keypoints = p.get('keypoints')
            bbox = p.get('bbox', [0, 0, 0, 0])
            
            if keypoints is None: continue
            
            skeleton = adapter.adapt_yolo(keypoints)
            seq = buffer.update(track_id, skeleton)
            
            x, y = int(bbox[0]), int(bbox[1])
            
            if len(seq) >= WINDOW_SIZE:
                seq_tensor = torch.tensor(seq, dtype=torch.float32)
                idx, conf = classifier.predict_from_sequence(seq_tensor)
                ntu_class = NTU60_CLASSES[idx] if idx < len(NTU60_CLASSES) else 'unknown'
                
                # Собираем сырые данные для финального resolve_target_class
                raw_ntu_predictions.append(ntu_class)
                
                action = map_ntu_to_target(ntu_class)
                
                # Триггер VLM: Курение (отправляем кроп)
                if action == "smoking_candidate":
                    if current_time - last_vlm_time > VLM_COOLDOWN:
                        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                        if crop.size > 0:
                            try:
                                vlm_result = vlm.analyze(crop) 
                                print(f"[{frame_num}] VLM Smoking Verification: {vlm_result.get('action')}")
                            except: pass
                        last_vlm_time = current_time
                
                label = f"{action if action else ntu_class} ({conf:.2f})"
            else:
                label = f"buffer {len(seq)}/{WINDOW_SIZE}"
            
            # Отрисовка текста (без рамок по просьбе KION)
            cv2.putText(frame_with_actions := frame, label, (x+5, y-7), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        # Триггер VLM: Групповые действия
        for event in group_events:
            # Отрисовка (Круг/Треугольник/Митинг)
            if event in ["circle_formation", "triangle_formation", "rally_candidate"]:
                cv2.putText(frame, f"MATH: {event.upper()}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 255), 2)
            
            # Вызов VLM для всех подозрительных групп
            if event in ["tug_of_war_candidate", "rally_candidate", "circle_formation", "triangle_formation"]:
                if current_time - last_vlm_time > VLM_COOLDOWN:
                    print(f"[{frame_num}] Trigger VLM: Group action check ({event})")
                    try:
                        vlm_result = vlm.analyze(frame) # Группы анализируем по всему кадру
                    except Exception:
                        pass
                    last_vlm_time = current_time
        
        # Расчет FPS для экрана
        if frame_num - last_fps_frames >= 30:
            elapsed = time.time() - last_fps_time
            if elapsed > 0:
                current_fps = (frame_num - last_fps_frames) / elapsed
                cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            last_fps_time = time.time()
            last_fps_frames = frame_num
        
        writer.write(frame)
        frame_num += 1
    
    cap.release()
    writer.release()
    
    # Расчет Accuracy
    if raw_ntu_predictions:
        # Вызываем с учетом ответа VLM
        vlm_act = vlm_result.get('action') if vlm_result else None
        final_class, scores = resolve_target_class(raw_ntu_predictions)

        print("\n" + "="*40)
        print(f"SCENE ANALYSIS COMPLETE: {video_name}")
        print(f"FINAL PREDICTED CLASS: {final_class}")
        
        # Автоматическая проверка Accuracy
        ground_truth = video_name.split('_')[0].lower() # берем начало имени файла
        
        # Маппинг для честного сравнения имен
        is_correct = False
        if final_class.lower() == ground_truth: is_correct = True
        elif ground_truth == "smoking" and final_class == "smoking_candidate": is_correct = True
        elif ground_truth == "jumping" and final_class == "jump": is_correct = True
        elif ground_truth == "walking" and final_class == "walk": is_correct = True
        elif ground_truth == "sitting" and final_class == "sit": is_correct = True
        
        if is_correct:
            print(f">>> RESULT: [ CORRECT ] (GT: {ground_truth})")
        else:
            print(f">>> RESULT: [ WRONG ] (GT: {ground_truth}, Pred: {final_class})")
        print("="*40 + "\n")

if __name__ == "__main__":
    main()