import os
import sys
import cv2
import torch

from src.detector import PoseDetector
from src.sequence_buffer_3d import SequenceBuffer3D
from src.skeleton_adapter_stgcnpp import SkeletonAdapterSTGCNPP
from src.classifiers.stgcnpp_classifier import STGCNPPClassifier


POSE_MODEL_PATH = "models/yolo11n-pose.pt"
STGCNPP_CONFIG = "configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-u100-80e_ntu60-xsub-keypoint-2d.py"
STGCNPP_CHECKPOINT = "models/stgcnpp_ntu60_xsub.pth"

# Это не ваши кастомные классы, а классы NTU60.
# Для демо можно хотя бы печатать индекс.
# Потом отдельно сделаем маппинг action_id -> название.
CLASS_NAMES = None

WINDOW_SIZE = 100


def draw_label(frame, text, x, y):
    cv2.rectangle(frame, (x, y - 25), (x + 260, y), (0, 0, 0), -1)
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

            if bbox is not None:
                x1, y1 = int(bbox[0]), int(bbox[1])
            else:
                x1, y1 = 20, 40

            if len(seq) >= WINDOW_SIZE:
                seq_tensor = torch.tensor(seq, dtype=torch.float32)
                pred_idx, conf = classifier.predict_from_sequence(seq_tensor)

                if CLASS_NAMES is None:
                    label = f"id={track_id} class={pred_idx} ({conf:.2f})"
                else:
                    label = f"id={track_id} {CLASS_NAMES[pred_idx]} ({conf:.2f})"
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


if __name__ == "__main__":
    main()
