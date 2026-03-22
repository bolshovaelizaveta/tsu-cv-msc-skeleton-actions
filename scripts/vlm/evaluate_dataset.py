import os
import sys
import json
import time
import yaml
import argparse
from collections import Counter, defaultdict

import cv2
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.detector import PoseDetector
from src.sequence_buffer_3d import SequenceBuffer3D
from src.skeleton_adapter_stgcnpp import SkeletonAdapterSTGCNPP
from src.classifiers.stgcnpp_classifier import STGCNPPClassifier
from src.utils.ntu60_labels import NTU60_CLASSES
from src.utils.action_mapping import map_ntu_to_target, resolve_target_class
from src.vlm.vlm_client import VLMClient
from src.analyzer import GroupAnalyzer


VIDEO_EXTENSIONS = {".mp4"}

POSE_MODEL = "models/yolo11m-pose.pt"
STGCNPP_CONFIG = "configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py"
STGCNPP_CHECKPOINT = "models/stgcnpp_ntu60_xsub.pth"

WINDOW_SIZE = 32
YOLO_SKIP = 5
VLM_COOLDOWN = 5.0


def is_video_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in VIDEO_EXTENSIONS


def canonicalize_label(label: str) -> str:
    label = label.strip().lower()
    aliases = {
        "walking": "walk",
        "jumping": "jump",
        "sitting": "sit",
        "standing": "stand",
        "smoking": "smoking_candidate",
    }
    return aliases.get(label, label)


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def collect_dataset(data_dir: str):
    items = []
    class_names = []

    for entry in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, entry)
        if not os.path.isdir(class_dir):
            continue

        gt_class = canonicalize_label(entry)
        class_names.append(gt_class)

        for fname in sorted(os.listdir(class_dir)):
            fpath = os.path.join(class_dir, fname)
            if os.path.isfile(fpath) and is_video_file(fname):
                items.append({
                    "video_path": fpath,
                    "ground_truth": gt_class,
                    "file_name": fname,
                })

    return items, sorted(set(class_names))


def init_models(use_vlm: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pose_device = "mps" if torch.backends.mps.is_available() else device

    detector = PoseDetector(POSE_MODEL, pose_device)
    adapter = SkeletonAdapterSTGCNPP()
    classifier = STGCNPPClassifier(STGCNPP_CONFIG, STGCNPP_CHECKPOINT, device)
    group_analyzer = GroupAnalyzer()
    vlm = VLMClient() if use_vlm else None

    return {
        "device": device,
        "pose_device": pose_device,
        "detector": detector,
        "adapter": adapter,
        "classifier": classifier,
        "group_analyzer": group_analyzer,
        "vlm": vlm,
    }


def predict_video(video_path: str, models: dict, use_vlm: bool = True):
    detector = models["detector"]
    adapter = models["adapter"]
    classifier = models["classifier"]
    group_analyzer = models["group_analyzer"]
    vlm = models["vlm"]

    buffer = SequenceBuffer3D(WINDOW_SIZE)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "final_class": "unknown",
            "scores": {},
            "ntu_distribution": {},
            "target_distribution": {},
            "error": f"Failed to open video: {video_path}",
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    raw_ntu_predictions = []
    mapped_target_predictions = []
    last_sequence = None
    vlm_result = None
    last_vlm_time = 0.0

    middle_frame_num = total_frames // 2 if total_frames > 0 else 0
    if use_vlm and vlm is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_num)
        ok, middle_frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if ok:
            try:
                vlm_result = vlm.analyze(middle_frame)
            except Exception:
                vlm_result = None
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_num = 0

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

        for person in persons:
            keypoints = person.get("keypoints")
            bbox = person.get("bbox", [0, 0, 0, 0])

            if keypoints is None:
                continue

            skeleton = adapter.adapt_yolo(keypoints)
            track_id = person.get("track_id", 0)
            seq = buffer.update(track_id, skeleton)
            last_sequence = seq

            if len(seq) >= WINDOW_SIZE:
                seq_tensor = torch.tensor(seq, dtype=torch.float32)
                pred_idx, conf = classifier.predict_from_sequence(seq_tensor)
                ntu_class = NTU60_CLASSES[pred_idx] if pred_idx < len(NTU60_CLASSES) else "unknown"

                raw_ntu_predictions.append(ntu_class)

                mapped = map_ntu_to_target(ntu_class)
                if mapped is not None:
                    mapped_target_predictions.append(mapped)

                if use_vlm and vlm is not None and mapped == "smoking_candidate":
                    if current_time - last_vlm_time > VLM_COOLDOWN:
                        x1, y1, x2, y2 = map(int, bbox)
                        crop = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                        if crop.size > 0:
                            try:
                                vlm_result = vlm.analyze(crop)
                            except Exception:
                                pass
                        last_vlm_time = current_time

        if use_vlm and vlm is not None:
            for event in group_events:
                if event in ["tug_of_war_candidate", "rally_candidate", "circle_formation", "triangle_formation"]:
                    if current_time - last_vlm_time > VLM_COOLDOWN:
                        try:
                            vlm_result = vlm.analyze(frame)
                        except Exception:
                            pass
                        last_vlm_time = current_time

        frame_num += 1

    cap.release()

    if not raw_ntu_predictions:
        return {
            "final_class": "unknown",
            "scores": {},
            "ntu_distribution": {},
            "target_distribution": {},
            "error": None,
        }

    final_class, scores = resolve_target_class(raw_ntu_predictions, last_sequence)

    # Если захотите реально использовать VLM-override, можно добавить здесь
    # vlm_action = vlm_result.get("action") if vlm_result else None

    return {
        "final_class": canonicalize_label(final_class),
        "scores": scores,
        "ntu_distribution": dict(Counter(raw_ntu_predictions)),
        "target_distribution": dict(Counter(mapped_target_predictions)),
        "error": None,
    }


def compute_metrics(y_true, y_pred, labels):
    total = len(y_true)
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    accuracy = safe_div(correct, total)

    confusion = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        confusion[t][p] += 1

    per_class = {}
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0

    for cls in labels:
        tp = confusion[cls][cls]
        fp = sum(confusion[other][cls] for other in labels if other != cls)
        fn = sum(confusion[cls][other] for other in labels if other != cls)
        support = sum(confusion[cls].values())

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
        class_acc = safe_div(tp, support)

        per_class[cls] = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy_within_class": class_acc,
            "correct": tp,
            "wrong": support - tp,
        }

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

    n = len(labels) if labels else 1
    macro_precision /= n
    macro_recall /= n
    macro_f1 /= n

    micro_tp = correct
    micro_fp = total - correct
    micro_fn = total - correct
    micro_precision = safe_div(micro_tp, micro_tp + micro_fp)
    micro_recall = safe_div(micro_tp, micro_tp + micro_fn)
    micro_f1 = safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall) if (micro_precision + micro_recall) else 0.0

    return {
        "overall": {
            "num_samples": total,
            "correct": correct,
            "wrong": total - correct,
            "accuracy": accuracy,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
        },
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def print_report(metrics, labels):
    overall = metrics["overall"]
    print("\n" + "=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"Samples:          {overall['num_samples']}")
    print(f"Correct:          {overall['correct']}")
    print(f"Wrong:            {overall['wrong']}")
    print(f"Accuracy:         {overall['accuracy']:.4f}")
    print(f"Micro Precision:  {overall['micro_precision']:.4f}")
    print(f"Micro Recall:     {overall['micro_recall']:.4f}")
    print(f"Micro F1:         {overall['micro_f1']:.4f}")
    print(f"Macro Precision:  {overall['macro_precision']:.4f}")
    print(f"Macro Recall:     {overall['macro_recall']:.4f}")
    print(f"Macro F1:         {overall['macro_f1']:.4f}")

    print("\nPER-CLASS METRICS")
    print("-" * 60)
    for cls in labels:
        m = metrics["per_class"][cls]
        print(
            f"{cls:<12} "
            f"support={m['support']:<3} "
            f"correct={m['correct']:<3} "
            f"wrong={m['wrong']:<3} "
            f"acc={m['accuracy_within_class']:.3f}"
            f"prec={m['precision']:.3f} "
            f"rec={m['recall']:.3f} "
            f"f1={m['f1']:.3f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Root dir with class subfolders")
    parser.add_argument("--output-json", default="results/eval_report.json", help="Where to save JSON report")
    parser.add_argument("--disable-vlm", action="store_true", help="Run only pose+STGCN pipeline without VLM")
    args = parser.parse_args()

    dataset, gt_labels = collect_dataset(args.data_dir)
    if not dataset:
        print(f"No videos found in {args.data_dir}")
        sys.exit(1)

    print(f"Found {len(dataset)} videos in {args.data_dir}")
    print(f"Ground-truth classes: {gt_labels}")

    models = init_models(use_vlm=not args.disable_vlm)

    results = []
    y_true = []
    y_pred = []

    for i, item in enumerate(dataset, 1):
        video_path = item["video_path"]
        gt = item["ground_truth"]
        print(f"[{i}/{len(dataset)}] {video_path}")

        pred_result = predict_video(video_path, models, use_vlm=not args.disable_vlm)
        pred = canonicalize_label(pred_result["final_class"])

        result_row = {
            "video_path": video_path,
            "file_name": item["file_name"],
            "ground_truth": gt,
            "prediction": pred,
            "correct": pred == gt,
            "scores": pred_result.get("scores", {}),
            "ntu_distribution": pred_result.get("ntu_distribution", {}),
            "target_distribution": pred_result.get("target_distribution", {}),
            "error": pred_result.get("error"),
        }
        results.append(result_row)

        y_true.append(gt)
        y_pred.append(pred)

        status = "CORRECT" if pred == gt else "WRONG"
        print(f"  -> GT={gt}, PRED={pred} [{status}]")

    all_labels = sorted(set(gt_labels) | set(y_pred))
    metrics = compute_metrics(y_true, y_pred, all_labels)
    print_report(metrics, all_labels)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    report = {
        "dataset_dir": args.data_dir,
        "labels": all_labels,
        "results": results,
        "metrics": metrics,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nSaved report to: {args.output_json}")


if __name__ == "__main__":
    main()