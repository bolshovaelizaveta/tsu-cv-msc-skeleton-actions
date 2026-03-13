from collections import defaultdict
import numpy as np


class SequenceBuffer:
    """
    Хранит временные последовательности скелетов
    """

    def __init__(self, window_size=30):
        self.window_size = window_size
        self.buffer = defaultdict(list)

    def update(self, track_id, skeleton):
        seq = self.buffer[track_id]
        seq.append(skeleton)

        if len(seq) > self.window_size:
            seq.pop(0)

        return np.array(seq)

    def get(self, track_id):
        return np.array(self.buffer.get(track_id, []))

    def clear(self):
        self.buffer.clear()
