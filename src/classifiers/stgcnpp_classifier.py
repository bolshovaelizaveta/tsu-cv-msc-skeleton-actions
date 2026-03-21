import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmaction.registry import MODELS
from mmaction.utils import register_all_modules


class STGCNPPClassifier:
    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cpu"):
        register_all_modules()

        self.device = device
        cfg = Config.fromfile(config_path)

        self.model = MODELS.build(cfg.model)
        load_checkpoint(self.model, checkpoint_path, map_location=device)
        self.model = self.model.to(device)
        self.model.eval()

    def predict_from_sequence(self, seq_tensor: torch.Tensor):
        """
        seq_tensor: [T, V, C]
        где:
          T - кадры
          V - суставы
          C - x,y,score
        """

        if seq_tensor.ndim != 3:
            raise ValueError(f"Expected [T, V, C], got {seq_tensor.shape}")

        # STGCN++ ожидает [N, M, T, V, C]
        x = seq_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feats = self.model.backbone(x)
            logits = self.model.cls_head(feats)

            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            conf = probs[0, pred_idx].item()

        return pred_idx, conf
