import os
import numpy as np


NUM_FRAMES = 30
NUM_JOINTS = 17
NUM_COORDS = 2
SAMPLES_PER_CLASS = 40

CLASSES = ["sit", "jump", "handshake", "hug", "fight"]


def base_skeleton():
    """
    Простой 2D-скелет в координатах [x, y]
    Порядок условный, но стабильный.
    """
    sk = np.array([
        [0.00, 0.90],   # 0 head
        [0.00, 0.78],   # 1 neck
        [-0.10, 0.78],  # 2 l_shoulder
        [-0.18, 0.62],  # 3 l_elbow
        [-0.22, 0.48],  # 4 l_wrist
        [0.10, 0.78],   # 5 r_shoulder
        [0.18, 0.62],   # 6 r_elbow
        [0.22, 0.48],   # 7 r_wrist
        [0.00, 0.55],   # 8 hip_center
        [-0.08, 0.55],  # 9 l_hip
        [-0.08, 0.32],  # 10 l_knee
        [-0.08, 0.08],  # 11 l_ankle
        [0.08, 0.55],   # 12 r_hip
        [0.08, 0.32],   # 13 r_knee
        [0.08, 0.08],   # 14 r_ankle
        [-0.04, 0.86],  # 15 l_eye-ish / face support
        [0.04, 0.86],   # 16 r_eye-ish / face support
    ], dtype=np.float32)
    return sk


def add_noise(sk, scale=0.01):
    return sk + np.random.normal(0, scale, sk.shape).astype(np.float32)


def normalize_sequence(seq):
    """
    Центрируем по hip_center и масштабируем.
    """
    seq = seq.copy()
    for t in range(seq.shape[0]):
        center = seq[t, 8, :2]
        seq[t, :, 0] -= center[0]
        seq[t, :, 1] -= center[1]
        scale = np.max(np.linalg.norm(seq[t, :, :2], axis=1))
        scale = max(scale, 1e-6)
        seq[t, :, :2] /= scale
    return seq


def make_sit():
    seq = []
    for t in range(NUM_FRAMES):
        sk = base_skeleton()

        # плавное приседание / посадка
        alpha = min(1.0, t / (NUM_FRAMES * 0.55))

        # опускаем корпус
        sk[:9, 1] -= 0.22 * alpha

        # сгибаем колени
        sk[10, 1] += 0.07 * alpha
        sk[13, 1] += 0.07 * alpha
        sk[11, 0] += 0.05 * alpha
        sk[14, 0] -= 0.05 * alpha

        # чуть вперед корпус
        sk[:9, 0] += 0.03 * alpha

        seq.append(add_noise(sk, 0.008))
    return normalize_sequence(np.stack(seq))


def make_jump():
    seq = []
    for t in range(NUM_FRAMES):
        sk = base_skeleton()

        phase = np.sin(np.pi * t / (NUM_FRAMES - 1))  # 0 -> 1 -> 0
        lift = 0.28 * phase

        # весь скелет поднимается
        sk[:, 1] += lift

        # ноги слегка поджимаются в полете
        sk[10, 1] += 0.04 * phase
        sk[13, 1] += 0.04 * phase
        sk[11, 0] += 0.03 * phase
        sk[14, 0] -= 0.03 * phase

        # руки вверх
        sk[3, 1] += 0.08 * phase
        sk[6, 1] += 0.08 * phase
        sk[4, 1] += 0.15 * phase
        sk[7, 1] += 0.15 * phase

        seq.append(add_noise(sk, 0.01))
    return normalize_sequence(np.stack(seq))


def make_handshake():
    seq = []
    for t in range(NUM_FRAMES):
        sk = base_skeleton()

        # правая рука тянется вперед к центру
        phase = np.sin(np.pi * t / (NUM_FRAMES - 1))
        sk[6, 0] += 0.10 * phase
        sk[7, 0] -= 0.12 * phase
        sk[6, 1] += 0.03 * phase
        sk[7, 1] += 0.05 * phase

        # небольшое покачивание кисти
        wave = 0.02 * np.sin(4 * np.pi * t / (NUM_FRAMES - 1))
        sk[7, 1] += wave

        seq.append(add_noise(sk, 0.008))
    return normalize_sequence(np.stack(seq))


def make_hug():
    seq = []
    for t in range(NUM_FRAMES):
        sk = base_skeleton()

        alpha = min(1.0, t / (NUM_FRAMES * 0.5))

        # обе руки тянутся к центру тела
        sk[3, 0] += 0.08 * alpha
        sk[4, 0] += 0.16 * alpha
        sk[6, 0] -= 0.08 * alpha
        sk[7, 0] -= 0.16 * alpha

        sk[4, 1] += 0.10 * alpha
        sk[7, 1] += 0.10 * alpha

        # корпус слегка вперед
        sk[:9, 0] += 0.015 * alpha

        seq.append(add_noise(sk, 0.008))
    return normalize_sequence(np.stack(seq))


def make_fight():
    seq = []
    for t in range(NUM_FRAMES):
        sk = base_skeleton()

        # резкие движения правой рукой
        punch = np.sin(3 * np.pi * t / (NUM_FRAMES - 1))
        sk[6, 0] += 0.10 * punch
        sk[7, 0] += 0.18 * punch
        sk[6, 1] += 0.05 * np.sign(punch)
        sk[7, 1] += 0.08 * np.sign(punch)

        # небольшие смещения корпуса
        sk[:, 0] += 0.03 * np.sin(2 * np.pi * t / (NUM_FRAMES - 1))
        sk[:, 1] += 0.02 * np.sin(5 * np.pi * t / (NUM_FRAMES - 1))

        seq.append(add_noise(sk, 0.015))
    return normalize_sequence(np.stack(seq))


GENERATORS = {
    "sit": make_sit,
    "jump": make_jump,
    "handshake": make_handshake,
    "hug": make_hug,
    "fight": make_fight,
}


def main():
    np.random.seed(42)

    for cls in CLASSES:
        out_dir = os.path.join("data", "ntu_subset", cls)
        os.makedirs(out_dir, exist_ok=True)

        generator = GENERATORS[cls]

        for i in range(SAMPLES_PER_CLASS):
            seq = generator()
            out_path = os.path.join(out_dir, f"sample_{i:03d}.npy")
            np.save(out_path, seq)

    print("Dummy NTU-like dataset created in data/ntu_subset")


if __name__ == "__main__":
    main()