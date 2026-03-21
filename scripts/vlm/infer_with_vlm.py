# scripts/infer_with_vlm.py
import os
import sys
import cv2
import torch
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
VLM_INTERVAL = 30  # секунд

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/infer_with_vlm.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Инициализация
    clf_device = "cuda" if torch.cuda.is_available() else "cpu"
    pose_device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    detector = PoseDetector(POSE_MODEL, pose_device)
    adapter = SkeletonAdapterSTGCNPP()
    buffer = SequenceBuffer3D(WINDOW_SIZE)
    classifier = STGCNPPClassifier(STGCNPP_CONFIG, STGCNPP_CHECKPOINT, clf_device)
    vlm = VLMClient()
    
    # Видео
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))
    
    writer = cv2.VideoWriter(f"outputs/output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    frame_num = 0
    predictions = []
    last_vlm = -VLM_INTERVAL
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = frame_num / fps
        persons = detector.get_skeleton_data(frame)
        
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
                    predictions.append(action)
                    label = f"{action} ({conf:.2f})"
                else:
                    label = ntu_class
            else:
                label = f"buffer {len(seq)}/{WINDOW_SIZE}"
            
            # Рисуем
            x, y = int(bbox[0]), int(bbox[1])
            cv2.rectangle(frame, (x, y-25), (x+300, y), (0,0,0), -1)
            cv2.putText(frame, label, (x+5, y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        # VLM
        if timestamp - last_vlm >= VLM_INTERVAL:
            last_vlm = timestamp
            result = vlm.analyze(frame)
            if result and result.get('success'):
                print(f"\n[{timestamp:.1f}s] VLM: {result['action']} (conf={result['confidence']:.2f}, people={result['participants']})")
                cv2.putText(frame, f"VLM: {result['action']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        
        writer.write(frame)
        frame_num += 1
        
        if frame_num % 100 == 0:
            print(f"Frame: {frame_num}")
    
    cap.release()
    writer.release()
    
    # Результаты
    if predictions:
        counter = Counter(predictions)
        print("\n" + "="*50)
        print("RESULTS")
        for action, count in counter.most_common():
            print(f"{action:15}: {count} ({count/len(predictions)*100:.1f}%)")
        print(f"\nFINAL: {counter.most_common(1)[0][0]}")

if __name__ == "__main__":
    main()