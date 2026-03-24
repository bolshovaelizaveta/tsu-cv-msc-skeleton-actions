import argparse
import os
import sys
import cv2
import time
import math 
import torch
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.detector import PoseDetector
from src.sequence_buffer_3d import SequenceBuffer3D
from src.skeleton_adapter_stgcnpp import SkeletonAdapterSTGCNPP
from src.classifiers.stgcnpp_classifier import STGCNPPClassifier
from src.utils.ntu60_labels import NTU60_CLASSES
from src.utils.action_mapping import map_ntu_to_target, resolve_target_class, compute_motion_energy
from src.analyzer import GroupAnalyzer

try:
    from src.vlm.vlm_client import VLMClient
except ImportError:
    VLMClient = None

POSE_MODEL = "models/yolo11m-pose.pt"
STGCNPP_CONFIG = "configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py"
STGCNPP_CHECKPOINT = "models/stgcnpp_ntu60_xsub.pth"
WINDOW_SIZE = 32
YOLO_SKIP = 5
VLM_COOLDOWN_FRAMES = 90  

# Маппинг папок датасета к нашим целевым классам
GT_MAPPING = {
    "walking": "walk",
    "sitting": "sit",
    "jumping": "jump",
    "smoking": "smoking_candidate", 
    "dance": "dance",
    "fight": "fight",
    "handshake": "handshake",
    "hug": "hug",
    "rally": "meeting", 
    "circle": "circle_triangle",
    "tug_of_war": "tug_of_war"
}

CLASSES_LIST = list(set(GT_MAPPING.values()))

class UnifiedBenchmarkSuite:
    def __init__(self, data_dir="data", use_vlm=True):
        self.data_dir = data_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[*] Инициализация моделей на {self.device.upper()} (Это произойдет только 1 раз!)...")
        self.detector = PoseDetector(POSE_MODEL, self.device)
        self.adapter = SkeletonAdapterSTGCNPP()
        self.classifier = STGCNPPClassifier(STGCNPP_CONFIG, STGCNPP_CHECKPOINT, self.device)
        self.base_group_analyzer = GroupAnalyzer()
        
        self.vlm = None
        if use_vlm:
            try:
                if VLMClient is not None:
                    self.vlm = VLMClient()
                    print("[*] VLM успешно подключена.")
                else:
                    print("Модуль VLMClient не найден. Работаем в GCN-only.")
            except Exception as e:
                print(f"ВНИМАНИЕ: Ошибка инициализации VLM ({e}). Работаем в GCN-only.")
        else:
            print("Режим GCN-only. VLM принудительно отключена.")
            
        print("Инициализация завершена.\n")


    def _get_strict_group_events(self, persons):
        """
        Обертка над GroupAnalyzer с более жесткими эвристиками плотности.
        Не трогаем исходный код src.analyzer, фильтруем результаты здесь.
        """
        events = self.base_group_analyzer.analyze(persons)
        
        if not events:
            return []

        strict_events = []
        centers = []
        for p in persons:
            box = p.get('bbox', [0, 0, 0, 0])
            centers.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))

        for event in events:
            if event == "rally_candidate":
                # Ужесточаем проверку плотности для митингов (rally)
                all_dists = []
                n = len(centers)
                for i in range(n):
                    for j in range(i + 1, n):
                        all_dists.append(math.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1]))
                
                if len(all_dists) > 0 and np.mean(all_dists) < 180:
                    strict_events.append(event)
            else:
                strict_events.append(event)
                
        return strict_events

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        buffer = SequenceBuffer3D(WINDOW_SIZE)
        raw_ntu_predictions = []
        vlm_responses = []
        
        frame_num = 0
        frames_since_vlm = VLM_COOLDOWN_FRAMES
        
        last_sequence = None
        start_time = time.time()
        frames_processed = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frames_processed += 1
            frames_since_vlm += 1
            
            if frame_num % YOLO_SKIP == 0:
                persons = self.detector.get_skeleton_data(frame)
            else:
                persons = []

            group_events = self._get_strict_group_events(persons) if persons else []
            current_gcn_actions = []
            
            # ST-GCN++ Оценка 
            for p in persons:
                track_id = p.get('track_id', 0)
                keypoints = p.get('keypoints')
                bbox = p.get('bbox', [0, 0, 0, 0])
                
                if keypoints is None: continue
                
                skeleton = self.adapter.adapt_yolo(keypoints)
                seq = buffer.update(track_id, skeleton)
                last_sequence = seq # Сохраняем для motion energy
                
                if len(seq) >= WINDOW_SIZE:
                    seq_tensor = torch.tensor(seq, dtype=torch.float32)
                    idx, conf = self.classifier.predict_from_sequence(seq_tensor)
                    ntu_class = NTU60_CLASSES[idx] if idx < len(NTU60_CLASSES) else 'unknown'
                    
                    raw_ntu_predictions.append(ntu_class)
                    action = map_ntu_to_target(ntu_class)
                    if action: current_gcn_actions.append((action, bbox))
            
            # VLM Триггеры (Индивидуальные и Парные) 
            if frames_since_vlm >= VLM_COOLDOWN_FRAMES:
                triggered = False
                
                for act, box in current_gcn_actions:
                    # Курение: шлем crop
                    if act == "smoking_candidate":
                        x1, y1, x2, y2 = map(int, box)
                        crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                        if crop.size > 0:
                            try:
                                res = self.vlm.analyze(crop)
                                if res and res.get('action'): vlm_responses.append(res.get('action'))
                                triggered = True
                            except: pass
                            
                    # Объятия и Рукопожатия: шлем весь кадр для контекста
                    elif act in ["hug", "handshake"]:
                        try:
                            res = self.vlm.analyze(frame)
                            if res and res.get('action'): vlm_responses.append(res.get('action'))
                            triggered = True
                        except: pass
                        
                    if triggered: break # Один успешный триггер на кадр 

            # VLM Триггеры (Групповые) 
            if frames_since_vlm >= VLM_COOLDOWN_FRAMES and group_events:
                try:
                    res = self.vlm.analyze(frame)
                    if res and res.get('action'): vlm_responses.append(res.get('action'))
                    frames_since_vlm = 0
                except: pass

            frame_num += 1

        cap.release()
        
        proc_time = time.time() - start_time
        fps = frames_processed / proc_time if proc_time > 0 else 0.0

        # Финальный Resolve с Dance и Circle 
        # Выбираем самый частый ответ VLM
        best_vlm_action = Counter(vlm_responses).most_common(1)[0][0] if vlm_responses else None
        
        # Если VLM увидел "круг", но ST-GCN зафиксировал быстрое движение (танцы), глушим "круг"
        if best_vlm_action and any(s in best_vlm_action.lower() for s in ["circle", "formation", "round"]):
            if last_sequence is not None:
                motion = compute_motion_energy(last_sequence)
                if motion > 0.08: # Быстро двигаются
                    best_vlm_action = "dance" # Переписываем ответ VLM

        final_class, _ = resolve_target_class(raw_ntu_predictions, last_sequence=last_sequence, vlm_action=best_vlm_action)
        
        if final_class is None:
            final_class = "unknown"

        return final_class, fps

    def run(self):
        y_true = []
        y_pred = []
        total_fps_list = []
        
        print(f"{'='*80}\nЗАПУСК БЕНЧМАРКА KION\n{'='*80}")
        
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if not file.lower().endswith(('.mp4', '.avi', '.mov')):
                    continue
                    
                video_path = os.path.join(root, file)
                folder_name = os.path.basename(root).lower()
                
                # Получаем истинный класс
                gt_class = GT_MAPPING.get(folder_name, "unknown")
                if gt_class == "unknown":
                    continue
                    
                print(f"Обработка: {file} | GT: {gt_class} ... ", end="")
                sys.stdout.flush()
                
                try:
                    pred_class, fps = self.process_video(video_path)
                    
                    y_true.append(gt_class)
                    y_pred.append(pred_class)
                    total_fps_list.append(fps)
                    
                    mark = "✅" if gt_class == pred_class else "❌"
                    print(f"Pred: {pred_class} {mark} (FPS: {fps:.1f})")
                except Exception as e:
                    print(f"ОШИБКА: {e}")
                    
        self._print_metrics(y_true, y_pred, np.mean(total_fps_list) if total_fps_list else 0)

    def _print_metrics(self, y_true, y_pred, avg_fps):
        if not y_true:
            print("Нет данных для подсчета метрик")
            return
            
        print("\n" + "="*80)
        print("ФИНАЛЬНЫЕ МЕТРИКИ")
        print("="*80)
        
        tp_total = fp_total = fn_total = 0
        macro_acc = 0.0
        
        print(f"{'Класс':<20} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8}")
        print("-" * 65)
        
        for cls in CLASSES_LIST:
            tp = sum(1 for gt, pr in zip(y_true, y_pred) if gt == cls and pr == cls)
            fp = sum(1 for gt, pr in zip(y_true, y_pred) if gt != cls and pr == cls)
            fn = sum(1 for gt, pr in zip(y_true, y_pred) if gt == cls and pr != cls)
            tn = sum(1 for gt, pr in zip(y_true, y_pred) if gt != cls and pr != cls)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
            
            tp_total += tp
            fp_total += fp
            fn_total += fn
            macro_acc += acc
            
            print(f"{cls:<20} | {acc:.4f}   | {precision:.4f}    | {recall:.4f}   | {f1:.4f}")

        # Общие метрики
        micro_prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        micro_rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
        
        correct_overall = sum(1 for gt, pr in zip(y_true, y_pred) if gt == pr)
        multi_acc = correct_overall / len(y_true)
        macro_acc_final = macro_acc / len(CLASSES_LIST)

        print("-" * 65)
        print(f"Средний FPS пайплайна : {avg_fps:.1f} кадров/сек")
        print(f"Multiclass Accuracy   : {multi_acc:.4f} ({correct_overall}/{len(y_true)})")
        print(f"Macro-Accuracy        : {macro_acc_final:.4f}")
        print(f"Micro F1-Score        : {micro_f1:.4f}")
        print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KION Hackathon Benchmark")
    parser.add_argument("--data-dir", type=str, default="data", help="Путь к датасету")
    parser.add_argument("--no-vlm", action="store_true", help="Принудительно отключить VLM")
    args = parser.parse_args()
    
    # Запускаем бенчмарк
    benchmark = UnifiedBenchmarkSuite(data_dir=args.data_dir, use_vlm=not args.no_vlm)
    benchmark.run()