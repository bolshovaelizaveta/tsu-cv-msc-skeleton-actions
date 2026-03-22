import numpy as np


class SkeletonAdapterSTGCNPP:
    def __init__(self, num_joints=17, conf_threshold=0.1):
        self.num_joints = num_joints
        self.conf_threshold = conf_threshold

    def adapt_yolo(self, keypoints):
        """
        keypoints: [V, 3] -> x, y, conf
        output: [17, 3]
        """
        kp = np.array(keypoints, dtype=np.float32)

        if kp.ndim != 2 or kp.shape[1] < 3:
            raise ValueError(f"Expected [V,3], got {kp.shape}")

        if kp.shape[0] < self.num_joints:
            pad = np.zeros((self.num_joints - kp.shape[0], kp.shape[1]), dtype=np.float32)
            kp = np.concatenate([kp, pad], axis=0)
        elif kp.shape[0] > self.num_joints:
            kp = kp[:self.num_joints]

        xy = kp[:, :2].copy()
        conf = kp[:, 2:3].copy()

        visible = conf[:, 0] >= self.conf_threshold

        if visible.any():
            center = np.mean(xy[visible], axis=0)
            xy[:, 0] -= center[0]
            xy[:, 1] -= center[1]

            scale = np.max(np.linalg.norm(xy[visible], axis=1))
            scale = max(scale, 1e-6)
            xy /= scale
        else:
            xy[:] = 0.0

        # если сустав не виден — зануляем координаты и conf
        xy[~visible] = 0.0
        conf[~visible] = 0.0

        return np.concatenate([xy, conf], axis=1).astype(np.float32)
