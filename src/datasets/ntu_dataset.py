import os
import numpy as np
import torch
from torch.utils.data import Dataset


class NTUDataset(Dataset):

    def __init__(self, data_dir):
        self.samples = []

        classes = os.listdir(data_dir)

        for label, cls in enumerate(classes):

            cls_dir = os.path.join(data_dir, cls)

            for file in os.listdir(cls_dir):
                if file.endswith(".npy"):
                    self.samples.append(
                        (os.path.join(cls_dir, file), label)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        data = np.load(path)

        data = torch.tensor(data).float()

        return data, label
