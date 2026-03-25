# scripts/infer_with_vlm.py
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

# Конфиг
POSE_MODEL = "models/yolo11m-pose.pt"
STGCNPP_CONFIG = "configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py"
STGCNPP_CHECKPOINT = "models/stgcnpp_ntu60_xsub.pth"
WINDOW_SIZE = 32
VLM_INTERVAL = 5  # секунд между вызовами VLM
YOLO_SKIP = 5  # Обрабатывать каждый 5й кадр

with open("configs/vlm/config.yaml", "r") as f:
    vlm_config = yaml.safe_load(f)["vlm"]

os.makedirs(vlm_config.get("output_dir", "results/vlm"), exist_ok=True)
os.makedirs(vlm_config.get("keyframes_dir", "results/keyframes"), exist_ok=True)

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/infer_with_vlm.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    clf_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    detector = PoseDetector(POSE_MODEL, clf_device)
    adapter = SkeletonAdapterSTGCNPP()
    buffer = SequenceBuffer3D(WINDOW_SIZE)
    classifier = STGCNPPClassifier(STGCNPP_CONFIG, STGCNPP_CHECKPOINT, clf_device)
    vlm = VLMClient()
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video info: {total_frames} frames, {fps:.2f} fps, duration: {total_duration:.2f}s")
    print(f"VLM will be called every {VLM_INTERVAL} seconds")
    print(f"Expected VLM calls: ~{int(total_duration / VLM_INTERVAL) + 1}\n")
    
    width, height = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(f"outputs/{video_name}_output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    frame_num = 0
    predictions = []
    vlm_results = []
    last_vlm_time = -VLM_INTERVAL  # Инициализация для первого вызова
    vlm_call_count = 0
    
    # FPS замер
    start_time = time.time()
    processed_frames = 0  # Реально обработанные YOLO кадры
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = frame_num / fps
        
        # YOLO только каждый YOLO_SKIP кадр
        if frame_num % YOLO_SKIP == 0:
            persons = detector.get_skeleton_data(frame)
            processed_frames += 1
        else:
            persons = []
        
        # Рисуем на frame_for_viz (с надписями)
        frame_for_viz = frame.copy()
        
        # Отслеживаем действия для каждого человека
        current_actions = []
        
        for p in persons:
            track_id = p.get('track_id', 0)
            keypoints = p.get('keypoints')
            bbox = p.get('bbox', [0,0])
            
            if keypoints is None:
                continue
            
            skeleton = adapter.adapt_yolo(keypoints)
            seq = buffer.update(track_id, skeleton)
            
            if len(seq) >= WINDOW_SIZE:
                seq_tensor = torch.tensor(seq, dtype=torch.float32)
                idx, conf = classifier.predict_from_sequence(seq_tensor)
                ntu_class = NTU60_CLASSES[idx] if idx < len(NTU60_CLASSES) else 'unknown'
                action = map_ntu_to_target(ntu_class)
                
                if action:
                    current_actions.append(action)
                    predictions.append(action)
                    label = f"{action} ({conf:.2f})"
                else:
                    label = ntu_class
            else:
                label = f"buffer {len(seq)}/{WINDOW_SIZE}"
            
            x, y = int(bbox[0]), int(bbox[1])
            cv2.rectangle(frame_for_viz, (x, y-25), (x+300, y), (0,0,0), -1)
            cv2.putText(frame_for_viz, label, (x+5, y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        # VLM вызов с проверкой на интервал
        # Используем более надежное сравнение с эпсилон
        time_since_last_vlm = timestamp - last_vlm_time
        should_call_vlm = time_since_last_vlm >= (VLM_INTERVAL - 0.01)  # Добавляем небольшой допуск
        
        if should_call_vlm and timestamp < total_duration:
            print(f"\n[DEBUG] VLM CALL #{vlm_call_count + 1}")
            print(f"  timestamp={timestamp:.2f}s, last_vlm={last_vlm_time:.2f}s, diff={time_since_last_vlm:.2f}s")
            
            last_vlm_time = timestamp
            vlm_call_count += 1
            
            # Определяем предложенное действие на основе последних предсказаний
            suggested = None
            if current_actions:
                # Используем текущие действия вместо predictions[-10:]
                suggested = Counter(current_actions).most_common(1)[0][0]
            elif predictions:
                suggested = Counter(predictions[-10:]).most_common(1)[0][0]
            
            print(f"  Suggested action: {suggested if suggested else 'none'}")
            
            # Вызываем VLM
            try:
                result = vlm.analyze(frame, suggested_action=suggested)
                
                if result and result.get('success'):
                    print(f"  VLM result: {result['action']} (conf={result['confidence']:.2f}, people={result['participants']})")
                    cv2.putText(frame_for_viz, f"VLM: {result['action']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                    
                    vlm_results.append({
                        'timestamp': timestamp,
                        'action': result['action'],
                        'confidence': result['confidence'],
                        'participants': result.get('participants', 0),
                        'reasoning': result.get('reasoning', ''),
                        'suggested': suggested,
                        'frame_num': frame_num
                    })
                    
                    if vlm_config['logging'].get('save_keyframes', True):
                        keyframe_path = os.path.join(vlm_config['keyframes_dir'], f"{video_name}_{timestamp:.1f}s.jpg")
                        cv2.imwrite(keyframe_path, frame)
                        print(f"  Keyframe saved: {keyframe_path}")
                else:
                    print(f"  VLM failed: {result.get('error', 'Unknown error') if result else 'No result'}")
            except Exception as e:
                print(f"  VLM exception: {e}")
        
        # Добавляем FPS на кадр
        if frame_num % 30 == 0 and frame_num > 0:
            elapsed = time.time() - start_time
            real_fps = processed_frames / elapsed if elapsed > 0 else 0
            effective_fps = real_fps * YOLO_SKIP
            cv2.putText(frame_for_viz, f"FPS: {effective_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        
        # Показываем информацию о VLM на кадре
        cv2.putText(frame_for_viz, f"VLM calls: {vlm_call_count}", (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        
        writer.write(frame_for_viz)
        frame_num += 1
        
        if frame_num % 100 == 0:
            print(f"Progress: {frame_num}/{total_frames} frames ({frame_num/total_frames*100:.1f}%)")
    
    cap.release()
    writer.release()
    
    # Итоговый FPS
    total_time = time.time() - start_time
    real_fps = processed_frames / total_time if total_time > 0 else 0
    effective_fps = real_fps * YOLO_SKIP
    
    print(f"\n{'='*50}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*50}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Processed frames (YOLO): {processed_frames}")
    print(f"Real YOLO FPS: {real_fps:.2f}")
    print(f"Effective FPS (full video): {effective_fps:.2f}")
    print(f"Total VLM calls: {vlm_call_count}")
    print(f"Expected VLM calls: ~{int(total_duration / VLM_INTERVAL) + 1}")
    
    if vlm_results and vlm_config['logging'].get('save_responses', True):
        results_path = os.path.join(vlm_config['output_dir'], f"{video_name}_vlm_results.json")
        with open(results_path, 'w') as f:
            json.dump(vlm_results, f, indent=2)
        print(f"\nSaved VLM results to: {results_path}")
    
    if predictions:
        counter = Counter(predictions)
        print("\n" + "="*50)
        print("CLASSIFICATION RESULTS")
        print("="*50)
        for action, count in counter.most_common(10):
            print(f"{action:20}: {count:4} ({count/len(predictions)*100:5.1f}%)")
        print(f"\nMOST COMMON: {counter.most_common(1)[0][0]}")
    
    if vlm_results:
        print("\n" + "="*50)
        print("VLM RESULTS")
        print("="*50)
        for res in vlm_results:
            print(f"  {res['timestamp']:.1f}s: {res['action']} (conf={res['confidence']:.2f}) [suggested: {res.get('suggested', 'none')}]")

if __name__ == "__main__":
    main()