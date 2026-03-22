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
from src.utils.action_mapping import map_ntu_to_target
from src.vlm.vlm_client import VLMClient
from src.analyzer import GroupAnalyzer # Анализатор групповых действий 

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
    
    # Инициализация 
    group_analyzer = GroupAnalyzer()
    last_vlm_time = 0.0
    VLM_COOLDOWN = 5.0 # Секунд между вызовами VLM
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    middle_frame_num = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_num)
    ret, middle_frame = cap.read()
    
    if not ret:
        return
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    width, height = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(f"outputs/{video_name}_output.mp4", 
                            cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    frame_num = 0
    predictions = []
    vlm_result = None
    processed_frames = 0
    
    try:
        vlm_result = vlm.analyze(middle_frame)
        if vlm_result and vlm_result.get('success') and vlm_config['logging'].get('save_responses', True):
            results_path = os.path.join(vlm_config['output_dir'], f"{video_name}_vlm_result.json")
            with open(results_path, 'w') as f:
                json.dump({
                    'video': video_name,
                    'middle_frame': middle_frame_num,
                    'result': vlm_result
                }, f, indent=2)
    except:
        pass
    
    start_time = time.time()
    fps_display_interval = 30
    last_fps_time = start_time
    last_fps_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_num % YOLO_SKIP == 0:
            persons = detector.get_skeleton_data(frame)
            processed_frames += 1
        else:
            persons = []
            
        # Вызов математики
        group_events = group_analyzer.analyze(persons) if persons else []
        current_time = time.time()
        
        frame_with_actions = frame.copy()
        
        for p in persons:
            track_id = p.get('track_id', 0)
            keypoints = p.get('keypoints')
            bbox = p.get('bbox', [0, 0])
            
            if keypoints is None:
                continue
            
            skeleton = adapter.adapt_yolo(keypoints)
            seq = buffer.update(track_id, skeleton)
            
            x, y = int(bbox[0]), int(bbox[1])
            
            if len(seq) >= WINDOW_SIZE:
                seq_tensor = torch.tensor(seq, dtype=torch.float32)
                idx, conf = classifier.predict_from_sequence(seq_tensor)
                ntu_class = NTU60_CLASSES[idx] if idx < len(NTU60_CLASSES) else 'unknown'
                action = map_ntu_to_target(ntu_class)
                
                # Триггер VLM на курение
                if action == "smoking_candidate":
                    if current_time - last_vlm_time > VLM_COOLDOWN:
                        print(f"[{frame_num}] Trigger VLM: Smoking check!")
                        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                        if crop.size > 0:
                            try:
                                vlm_result = vlm.analyze(crop) # Обновляем vlm_result 
                            except Exception:
                                pass
                        last_vlm_time = current_time
                
                if action:
                    predictions.append(action)
                    label = f"{action} ({conf:.2f})"
                else:
                    label = ntu_class
            else:
                label = f"buffer {len(seq)}/{WINDOW_SIZE}"
            
            # Убрала BBox для KION, они просили без рамок
            # cv2.rectangle(frame_with_actions, (x, y-25), (x+300, y), (0,0,0), -1)
            cv2.putText(frame_with_actions, label, (x+5, y-7), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        # Отрисовка групп и триггер на канат
        for event in group_events:
            # Рисуем математические фигуры (Круг/Треугольник/Толпа)
            if event in ["circle_formation", "triangle_formation", "rally_candidate"]:
                cv2.putText(frame_with_actions, f"MATH: {event.upper()}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 255), 2)
            
            # Если математика нашла канат - шлем весь кадр в VLM
            if event == "tug_of_war_candidate":
                if current_time - last_vlm_time > VLM_COOLDOWN:
                    print(f"[{frame_num}] Trigger VLM: Tug of War check!")
                    try:
                        vlm_result = vlm.analyze(frame) # Обновляем vlm_result
                    except Exception:
                        pass
                    last_vlm_time = current_time

        if vlm_result and vlm_result.get('success'):
            cv2.putText(frame_with_actions, f"VLM: {vlm_result['action']}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        
        current_time_fps = time.time()
        if frame_num - last_fps_frames >= fps_display_interval:
            elapsed = current_time_fps - last_fps_time
            if elapsed > 0:
                current_fps = (frame_num - last_fps_frames) / elapsed
                cv2.putText(frame_with_actions, f"FPS: {current_fps:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                print(f"Processing FPS: {current_fps:.1f} | Frame: {frame_num}/{total_frames}")
            last_fps_time = current_time_fps
            last_fps_frames = frame_num
        
        writer.write(frame_with_actions)
        frame_num += 1
    
    cap.release()
    writer.release()
    
    total_time = time.time() - start_time
    avg_fps = frame_num / total_time if total_time > 0 else 0
    
    print(f"\nAverage FPS: {avg_fps:.1f}")
    print(f"Total time: {total_time:.1f}s")
    
    if predictions:
        counter = Counter(predictions)
        print(f"Most common action: {counter.most_common(1)[0][0]}")
        print(f"Total predictions: {len(predictions)}")
    
    if vlm_result and vlm_result.get('success'):
        print(f"VLM action: {vlm_result['action']} (conf={vlm_result['confidence']:.2f})")

if __name__ == "__main__":
    main()