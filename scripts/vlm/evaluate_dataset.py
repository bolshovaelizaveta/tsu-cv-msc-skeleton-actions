import os
import sys
import json
import time
import yaml
import argparse
from collections import Counter, defaultdict

import cv2
import torch

import os
import sys
import json
import time
import argparse
from collections import Counter

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
        "meeting": "rally",
        "circle_triangle": "dance",
    }
    return aliases.get(label, label)


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def extract_vlm_action(vlm_result):
    if not vlm_result:
        return None

    if isinstance(vlm_result, dict):
        action = vlm_result.get("action")
        if action is None:
            return None
        return str(action).strip()

    return str(vlm_result).strip()


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
            "vlm_action": None,
            "vlm_result": None,
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
            "vlm_action": extract_vlm_action(vlm_result),
            "vlm_result": vlm_result,
            "error": None,
        }

    vlm_action = extract_vlm_action(vlm_result) if use_vlm else None
    final_class, scores = resolve_target_class(
        raw_ntu_predictions,
        last_sequence=last_sequence,
        vlm_action=vlm_action,
    )

    return {
        "final_class": canonicalize_label(final_class),
        "scores": scores,
        "ntu_distribution": dict(Counter(raw_ntu_predictions)),
        "target_distribution": dict(Counter(mapped_target_predictions)),
        "vlm_action": vlm_action,
        "vlm_result": vlm_result,
        "error": None,
    }

# остальная часть (метрики, main) без изменений


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