import os
import sys
import cv2
import torch
from collections import Counter

from src.detector import PoseDetector
from src.sequence_buffer_3d import SequenceBuffer3D
from src.skeleton_adapter_stgcnpp import SkeletonAdapterSTGCNPP
from src.classifiers.stgcnpp_classifier import STGCNPPClassifier
from src.utils.ntu60_labels import NTU60_CLASSES


POSE_MODEL_PATH = "models/yolo11n-pose.pt"
STGCNPP_CONFIG = "configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py"
STGCNPP_CHECKPOINT = "models/stgcnpp_ntu60_xsub.pth"

WINDOW_SIZE = 32


def draw_label(frame, text, x, y):
    x = max(10, x)
    y = max(30, y)

    cv2.rectangle(frame, (x, y - 25), (x + 320, y), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (x + 5, y - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/infer_stgcnpp_example.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]

    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        sys.exit(1)

    for path in [POSE_MODEL_PATH, STGCNPP_CONFIG, STGCNPP_CHECKPOINT]:
        if not os.path.exists(path):
            print(f"Required file not found: {path}")
            sys.exit(1)

    clf_device = "cuda" if torch.cuda.is_available() else "cpu"
    pose_device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("Pose device:", pose_device)
    print("STGCN++ device:", clf_device)

    detector = PoseDetector(model_path=POSE_MODEL_PATH, device=pose_device, conf=0.5)
    adapter = SkeletonAdapterSTGCNPP(num_joints=17, conf_threshold=0.1)
    buffer = SequenceBuffer3D(window_size=WINDOW_SIZE)

    classifier = STGCNPPClassifier(
        config_path=STGCNPP_CONFIG,
        checkpoint_path=STGCNPP_CHECKPOINT,
        device=clf_device
    )

    cap = cv2.VideoCapture(video_path)

    os.makedirs("outputs", exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = f"outputs/stgcnpp_{os.path.basename(video_path)}"
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    # список для накопления предсказаний
    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        persons = detector.get_skeleton_data(frame)

        for person in persons:
            track_id = person.get("track_id", -1)
            keypoints = person.get("keypoints")
            bbox = person.get("bbox", None)

            if keypoints is None:
                continue

            skeleton = adapter.adapt_yolo(keypoints)
            seq = buffer.update(track_id, skeleton)

            if bbox is not None and len(bbox) >= 2:
                x1, y1 = int(bbox[0]), int(bbox[1])
            else:
                x1, y1 = 20, 40

            if len(seq) >= WINDOW_SIZE:
                seq_tensor = torch.tensor(seq, dtype=torch.float32)
                pred_idx, conf = classifier.predict_from_sequence(seq_tensor)

                if pred_idx < len(NTU60_CLASSES):
                    class_name = NTU60_CLASSES[pred_idx]
                else:
                    class_name = f"class_{pred_idx}"

                label = f"id={track_id} {class_name} ({conf:.2f})"
                print(f"[track {track_id}] -> {class_name} ({conf:.2f})")

                # 🔥 сохраняем предсказание
                predictions.append(class_name)

            else:
                label = f"id={track_id} buffering {len(seq)}/{WINDOW_SIZE}"

            draw_label(frame, label, x1, y1)

        writer.write(frame)
        cv2.imshow("STGCN++ Example", frame)

        key = cv2.waitKey(1)
        if key == 27 or key == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"Saved annotated video to: {out_path}")

    if len(predictions) > 0:
        counter = Counter(predictions)
        total = len(predictions)

        print("\n========================")
        print("CLASS DISTRIBUTION:")

        for cls, count in counter.most_common():
            freq = count / total
            print(f"{cls:<35} — {freq:.2%} ({count})")

        print("========================")
    else:
        print("No predictions made")

if __name__ == "__main__":
    main()