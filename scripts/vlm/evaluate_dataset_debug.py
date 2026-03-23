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
from src.utils.action_mapping import map_ntu_to_target, resolve_target_class, compute_motion_energy
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
    return label.strip().lower()


def extract_vlm_action(vlm_result):
    if not vlm_result:
        return None
    if isinstance(vlm_result, dict):
        return vlm_result.get("action")
    return str(vlm_result)


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
        return None

    raw_ntu_predictions = []
    mapped_target_predictions = []
    last_sequence = None
    vlm_result = None

    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % YOLO_SKIP == 0:
            persons = detector.get_skeleton_data(frame)
        else:
            persons = []

        for person in persons:
            keypoints = person.get("keypoints")
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
                if mapped:
                    mapped_target_predictions.append(mapped)

        frame_num += 1

    cap.release()

    if not raw_ntu_predictions:
        return None

    vlm_action = extract_vlm_action(vlm_result) if use_vlm else None

    final_class, scores = resolve_target_class(
        raw_ntu_predictions,
        last_sequence=last_sequence,
        vlm_action=vlm_action,
    )

    motion = compute_motion_energy(last_sequence) if last_sequence is not None else None

    return {
        "final_class": final_class,
        "scores": scores,
        "ntu_distribution": Counter(raw_ntu_predictions),
        "target_distribution": Counter(mapped_target_predictions),
        "motion": motion,
        "vlm_action": vlm_action,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-json", default="results/eval_debug.json")
    parser.add_argument("--debug-wrong-only", action="store_true")
    parser.add_argument("--debug-all", action="store_true")
    parser.add_argument("--focus-class", type=str, default=None)
    args = parser.parse_args()

    dataset, class_names = collect_dataset(args.data_dir)
    models = init_models(use_vlm=False)

    results = []

    for i, item in enumerate(dataset):
        print(f"[{i+1}/{len(dataset)}] {item['video_path']}")

        pred = predict_video(item["video_path"], models, use_vlm=False)
        if pred is None:
            continue

        gt = item["ground_truth"]
        pred_class = pred["final_class"]

        correct = gt == pred_class
        print(f"  -> GT={gt}, PRED={pred_class} [{'CORRECT' if correct else 'WRONG'}]")

        should_debug = (
            args.debug_all or
            (args.debug_wrong_only and not correct) or
            (args.focus_class and gt == args.focus_class)
        )

        if should_debug:
            print("    --- DEBUG ---")
            print("    motion:", pred["motion"])
            print("    vlm_action:", pred["vlm_action"])

            print("    top_ntu:", pred["ntu_distribution"].most_common(8))
            print("    top_target:", pred["target_distribution"].most_common(5))

            sorted_scores = sorted(pred["scores"].items(), key=lambda x: x[1], reverse=True)
            print("    top_scores:", sorted_scores[:5])

        results.append({
            "video": item["video_path"],
            "gt": gt,
            "pred": pred_class,
            "correct": correct,
            "motion": pred["motion"],
            "ntu": pred["ntu_distribution"],
            "scores": pred["scores"],
        })

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
