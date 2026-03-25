import os
import sys
import cv2
import time
import math
import json
import csv
import torch
import argparse
import numpy as np
from collections import Counter

# Вычисляем корень проекта 
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.detector import PoseDetector
from src.utils.sequence_buffer_3d import SequenceBuffer3D
from src.utils.skeleton_adapter_stgcnpp import SkeletonAdapterSTGCNPP
from src.classifiers.stgcnpp_classifier import STGCNPPClassifier
from src.utils.ntu60_labels import NTU60_CLASSES
from src.utils.action_mapping import map_ntu_to_target, resolve_target_class, compute_motion_energy
from src.analyzer import GroupAnalyzer

try:
    from src.vlm.vlm_client import VLMClient
except ImportError:
    VLMClient = None

POSE_MODEL = os.path.join(ROOT_DIR, "models/yolo11m-pose.pt")
STGCNPP_CONFIG = os.path.join(ROOT_DIR, "configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py")
STGCNPP_CHECKPOINT = os.path.join(ROOT_DIR, "models/stgcnpp_ntu60_xsub.pth")
WINDOW_SIZE = 32
YOLO_SKIP = 5

GT_MAPPING = {
    "walking": "walk", "sitting": "sit", "jumping": "jump", "smoking": "smoking_candidate",
    "dance": "dance", "fight": "fight", "handshake": "handshake", "hug": "hug",
    "rally": "meeting", "circle": "circle_triangle", "tug_of_war": "tug_of_war"
}

CLASSES_LIST = list(set(GT_MAPPING.values()))

class UnifiedBenchmarkSuite:
    def __init__(self, data_dir="data", output_dir="results", use_vlm=True, save_video=False):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.save_video = save_video
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        os.makedirs(self.output_dir, exist_ok=True)
        if self.save_video:
            os.makedirs(os.path.join(self.output_dir, "videos"), exist_ok=True)

        self.detector = PoseDetector(POSE_MODEL, self.device)
        self.adapter = SkeletonAdapterSTGCNPP()
        self.classifier = STGCNPPClassifier(STGCNPP_CONFIG, STGCNPP_CHECKPOINT, self.device)
        self.base_group_analyzer = GroupAnalyzer()
        
        self.vlm = VLMClient() if (use_vlm and VLMClient) else None
        print(f"Инициализация завершена. VLM: {'ВКЛ' if self.vlm else 'ВЫКЛ'} Device: {self.device}\n")

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        buffer = SequenceBuffer3D(WINDOW_SIZE)
        raw_ntu_predictions = []
        
        # Копилка для Топ-3 кадров аудита VLM
        audit_items = [] 

        frame_num = 0
        frames_processed = 0
        start_time = time.time()

        writer = None
        if self.save_video:
            fps_v = cap.get(cv2.CAP_PROP_FPS)
            w, h = int(cap.get(3)), int(cap.get(4))
            out_p = os.path.join(self.output_dir, "videos", os.path.basename(video_path))
            writer = cv2.VideoWriter(out_p, cv2.VideoWriter_fourcc(*'mp4v'), fps_v, (w, h))

        last_draw_data = {"boxes": [], "labels": [], "events": []}

        while cap.isOpened():
            # Оптимизация Grab 
            if not self.save_video and frame_num % YOLO_SKIP != 0:
                if not cap.grab(): break
                frame_num += 1
                continue
                
            ret, frame = cap.read()
            if not ret: break
            frames_processed += 1
            
            if frame_num % YOLO_SKIP == 0:
                persons = self.detector.get_skeleton_data(frame)
                # Математика групп
                group_events = self.base_group_analyzer.analyze(persons) if persons else []
                last_draw_data["events"] = group_events
                last_draw_data["boxes"].clear(); last_draw_data["labels"].clear()
                
                for p in persons:
                    skeleton = self.adapter.adapt_yolo(p['keypoints'])
                    seq = buffer.update(p['track_id'], skeleton)
                    
                    if len(seq) >= WINDOW_SIZE:
                        idx, conf = self.classifier.predict_from_sequence(torch.tensor(seq, dtype=torch.float32))
                        ntu_class = NTU60_CLASSES[idx]
                        raw_ntu_predictions.append(ntu_class)
                        action = map_ntu_to_target(ntu_class)
                        
                        if action:
                            last_draw_data["boxes"].append(p['bbox'])
                            last_draw_data["labels"].append(f"{action} ({conf:.2f})")
                            
                            # Отбор кадров для аудита в vlm (без вызовы vlm)
                            if action in ["smoking_candidate", "fight", "hug", "handshake", "jump", "dance"]:
                                crop = None
                                if action == "smoking_candidate":
                                    x1, y1, x2, y2 = map(int, p['bbox'])
                                    crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                                
                                audit_items.append({
                                    "conf": conf,
                                    "frame": frame.copy(),
                                    "crop": crop,
                                    "type": 'crop' if action == "smoking_candidate" else 'full'
                                })
                                # Храним только 15 лучших по конфиденсу
                                audit_items = sorted(audit_items, key=lambda x: x['conf'], reverse=True)[:15]
                
                # Если математика нашла группу - это повод добавить кадр в аудит
                if group_events and len(audit_items) < 10:
                    audit_items.append({"conf": 0.99, "frame": frame.copy(), "crop": None, "type": 'full'})

            if writer:
                for b, l in zip(last_draw_data["boxes"], last_draw_data["labels"]):
                    cv2.putText(frame, l, (int(b[0])+5, int(b[1])-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                for i, e in enumerate(last_draw_data["events"]):
                    cv2.putText(frame, f"MATH: {e.upper()}", (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 255), 2)
                writer.write(frame)
            frame_num += 1

        vlm_results = []
        if self.vlm and audit_items:
            # Выбираем 3 разных момента: лучший, средний и последний из топа
            indices = [0, len(audit_items)//2, len(audit_items)-1]
            unique_indices = sorted(list(set(indices)))
            
            for idx in unique_indices:
                item = audit_items[idx]
                try:
                    img = item['crop'] if (item['type'] == 'crop' and item['crop'] is not None) else item['frame']
                    res = self.vlm.analyze(img)
                    if res and res.get('action'):
                        vlm_results.append(res.get('action').lower())
                except: pass

        cap.release()
        if writer: writer.release()
        
       # Большинство голосов
        mapped_gcn = [map_ntu_to_target(c) for c in raw_ntu_predictions if map_ntu_to_target(c)]
        final_class = Counter(mapped_gcn).most_common(1)[0][0] if mapped_gcn else "unknown"

        # Оверрайд на основе аудита VLM 
        if self.vlm and vlm_results:
            # Считаем, какое действие VLM называла чаще всего
            vlm_counts = Counter(vlm_results)
            best_vlm_vote, vote_count = vlm_counts.most_common(1)[0]
            
            # Верим VLM только если она подтвердила действие минимум в 2-х кадрах из 3-х
            if vote_count >= 2:
                va = best_vlm_vote.lower()
                if any(s in va for s in ["fight", "punch", "kick", "combat"]): final_class = "fight"
                elif any(s in va for s in ["hug", "embrace"]): final_class = "hug"
                elif any(s in va for s in ["handshake", "shake"]): final_class = "handshake"
                elif any(s in va for s in ["smoke", "cigarette", "vape"]): final_class = "smoking_candidate"
                elif any(s in va for s in ["rally", "meeting", "protest", "crowd"]): final_class = "meeting"
                elif any(s in va for s in ["tug", "war", "rope"]): final_class = "tug_of_war"
                elif any(s in va for s in ["circle", "formation"]): final_class = "circle_triangle"
                elif any(s in va for s in ["walk", "run"]): final_class = "walk"
                elif any(s in va for s in ["sit", "seat"]): final_class = "sit"
                elif any(s in va for s in ["jump", "hop"]): final_class = "jump"
                elif any(s in va for s in ["dance"]): final_class = "dance"
            else:
                # Если VLM каждый раз говорит разное (1, 1, 1), мы игнорируем её 
                # и оставляем решение за ST-GCN. 
                pass

        proc_time = time.time() - start_time
        fps = frame_num / proc_time if proc_time > 0 else 0.0
        return final_class or "unknown", fps

    def run(self):
        y_true, y_pred, total_fps = [], [], []
        report = []
        for root, _, files in os.walk(self.data_dir):
            for f in sorted(files):
                if not f.lower().endswith(('.mp4', '.avi', '.mov')): continue
                gt = GT_MAPPING.get(os.path.basename(root).lower(), "unknown")
                if gt == "unknown": continue
                
                print(f"Обработка: {f} | GT: {gt} ... ", end="", flush=True)
                res_cls, fps = self.process_video(os.path.join(root, f))
                y_true.append(gt); y_pred.append(res_cls); total_fps.append(fps)
                print(f"Pred: {res_cls} {'✅' if gt==res_cls else '❌'} (FPS: {fps:.1f})")
                report.append({"file": f, "gt": gt, "pred": res_cls, "fps": fps})
        
        self._save_metrics(y_true, y_pred, np.mean(total_fps), report)

    def _save_metrics(self, y_true, y_pred, avg_fps, report):
        # Сохранение CSV
        csv_p = os.path.join(self.output_dir, "benchmark_report.csv")
        with open(csv_p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["file", "gt", "pred", "fps"])
            w.writeheader(); w.writerows(report)
        
        # Сохранение JSON 
        metrics_dict = {}
        for cls in CLASSES_LIST:
            tp = sum(1 for g, p in zip(y_true, y_pred) if g == cls and p == cls)
            fp = sum(1 for g, p in zip(y_true, y_pred) if g != cls and p == cls)
            fn = sum(1 for g, p in zip(y_true, y_pred) if g == cls and p != cls)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            metrics_dict[cls] = {"f1": round(f1, 4), "precision": round(precision, 4), "recall": round(recall, 4)}

        with open(os.path.join(self.output_dir, "benchmark_report.json"), "w") as f:
            json.dump({"overall_avg_fps": round(avg_fps, 2), "per_class": metrics_dict}, f, indent=4)

        print(f"\nСредний FPS: {avg_fps:.1f}")
        print(f"Отчеты сохранены в папку: {self.output_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--no-vlm", action="store_true")
    p.add_argument("--save-video", action="store_true")
    args = p.parse_args()
    
    suite = UnifiedBenchmarkSuite(
        data_dir=args.data_dir, 
        output_dir=args.output_dir, 
        use_vlm=not args.no_vlm, 
        save_video=args.save_video
    )
    suite.run()