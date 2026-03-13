import torch
import torch.nn as nn


class NTUBaselineClassifier(nn.Module):
    def __init__(self, num_joints=17, hidden=128, num_classes=10):
        super().__init__()

        input_size = num_joints * 2

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            batch_first=True
        )

        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        """
        x: [batch, frames, joints, coords]
        """
        B, T, J, C = x.shape
        x = x.reshape(B, T, J * C)

        out, _ = self.lstm(x)
        out = out[:, -1]

        return self.fc(out)
