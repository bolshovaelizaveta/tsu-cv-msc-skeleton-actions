import os
import sys
import cv2
import torch

from src.detector import PoseDetector
from src.sequence_buffer import SequenceBuffer
from src.skeleton_adapter import SkeletonAdapter
from src.classifiers.ntu_baseline import NTUBaselineClassifier


MODEL_PATH = "models/ntu_baseline.pt"

CLASS_NAMES = [
    "sit",
    "jump",
    "handshake",
    "hug",
    "fight"
]

WINDOW_SIZE = 30
NUM_JOINTS = 17
NUM_CLASSES = len(CLASS_NAMES)


def draw_label(frame, text, x, y):
    cv2.rectangle(frame, (x, y - 25), (x + 200, y), (0, 0, 0), -1)

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


def load_model(device):

    model = NTUBaselineClassifier(
        num_joints=NUM_JOINTS,
        num_classes=NUM_CLASSES
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    return model


def main():

    if len(sys.argv) < 2:
        print("Usage: python infer_video_modular.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]

    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(device)

    detector = PoseDetector("yolo11n-pose.pt")
    adapter = SkeletonAdapter(num_joints=NUM_JOINTS)
    buffer = SequenceBuffer(window_size=WINDOW_SIZE)

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs("outputs", exist_ok=True)

    output_path = "outputs/annotated_video.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video_writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    print("Saving annotated video to:", output_path)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        persons = detector.get_skeleton_data(frame)

        for person in persons:

            track_id = person.get("track_id", -1)
            keypoints = person.get("keypoints")

            if keypoints is None:
                continue

            kp = adapter.adapt_yolo(keypoints)

            seq = buffer.update(track_id, kp)

            bbox = person.get("bbox", None)

            if bbox is not None:
                x1, y1 = int(bbox[0]), int(bbox[1])
            else:
                x1, y1 = 20, 40

            if len(seq) >= WINDOW_SIZE:

                x = torch.tensor(seq).unsqueeze(0).float().to(device)

                with torch.no_grad():

                    logits = model(x)

                    probs = torch.softmax(logits, dim=1)

                    pred_idx = torch.argmax(probs, dim=1).item()

                    conf = probs[0, pred_idx].item()

                label = f"{CLASS_NAMES[pred_idx]} ({conf:.2f})"

            else:

                label = f"track {track_id}: buffering..."

            draw_label(frame, label, x1, y1)

        video_writer.write(frame)

        cv2.imshow("Skeleton Action Recognition", frame)

        key = cv2.waitKey(1)

        if key == 27 or key == ord("q"):
            break

    cap.release()
    video_writer.release()

    print("Annotated video saved.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()