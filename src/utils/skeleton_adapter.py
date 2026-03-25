import numpy as np


class SkeletonAdapter:
    def __init__(self, num_joints=17):
        self.num_joints = num_joints

    def normalize(self, keypoints):
        keypoints = np.array(keypoints, dtype=np.float32)

        if keypoints.shape[0] == 0:
            return np.zeros((self.num_joints, 2), dtype=np.float32)

        if keypoints.shape[1] >= 2:
            xy = keypoints[:, :2].copy()
        else:
            raise ValueError("Keypoints must have at least 2 coordinates per joint")

        center = np.mean(xy, axis=0)
        xy[:, 0] -= center[0]
        xy[:, 1] -= center[1]

        scale = np.max(np.linalg.norm(xy, axis=1))
        scale = max(scale, 1e-6)
        xy /= scale

        return xy.astype(np.float32)

    def adapt_yolo(self, keypoints):
        kp = np.array(keypoints, dtype=np.float32)

        if kp.shape[0] < self.num_joints:
            pad = np.zeros((self.num_joints - kp.shape[0], kp.shape[1]), dtype=np.float32)
            kp = np.concatenate([kp, pad], axis=0)
        elif kp.shape[0] > self.num_joints:
            kp = kp[:self.num_joints]

        return self.normalize(kp)

    def adapt_ntu(self, skeleton):
        sk = np.array(skeleton, dtype=np.float32)

        if sk.shape[0] < self.num_joints:
            pad = np.zeros((self.num_joints - sk.shape[0], sk.shape[1]), dtype=np.float32)
            sk = np.concatenate([sk, pad], axis=0)
        elif sk.shape[0] > self.num_joints:
            sk = sk[:self.num_joints]

        return self.normalize(sk)
