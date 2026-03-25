from collections import defaultdict
import numpy as np


class SequenceBuffer3D:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.buffer = defaultdict(list)

    def update(self, track_id, skeleton):
        seq = self.buffer[track_id]
        seq.append(skeleton)

        if len(seq) > self.window_size:
            seq.pop(0)

        return np.array(seq, dtype=np.float32)

    def get(self, track_id):
        return np.array(self.buffer.get(track_id, []), dtype=np.float32)

    def clear(self):
        self.buffer.clear()
