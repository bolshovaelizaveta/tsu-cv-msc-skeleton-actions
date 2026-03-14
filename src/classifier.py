import torch
from src.classifiers.ntu_baseline import NTUBaselineClassifier
from src.skeleton_adapter import SkeletonAdapter
from src.sequence_buffer import SequenceBuffer

class ActionClassifier:
    def __init__(self, model_path="models/ntu_baseline.pt", device='cpu'):
        self.device = torch.device(device)
        
        self.classes = ["sit", "jump", "handshake", "hug", "fight"]
        
        self.adapter = SkeletonAdapter(num_joints=17)
        self.buffer = SequenceBuffer(window_size=30)
        
        self.model = NTUBaselineClassifier(num_joints=17, num_classes=len(self.classes)).to(self.device)
        self.ml_ready = False
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.ml_ready = True
        except Exception as e:
            print(f"ML модель не загружена. Ошибка: {e}")

    def predict(self, person_data: dict) -> str:
        track_id = person_data.get('track_id')
        kpts = person_data.get('keypoints')
        
        if not track_id or not kpts:
            return "unknown"

        # Нормализация данных 
        try:
            norm_kpts = self.adapter.adapt_yolo(kpts)
        except:
            return "unknown"

        # Добавление в буфер 
        seq = self.buffer.update(track_id, norm_kpts)

        # Ждем, пока накопится 30 кадров
        if len(seq) < 30 or not self.ml_ready:
            return "buffering"

        # ML Инференс (Предсказание нейросети)
        x = torch.tensor(seq).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            conf = probs[0, pred_idx].item()

        # Если нейросеть уверена больше чем на 60% - выдаем действие
        if conf > 0.6:
            return self.classes[pred_idx]
            
        # Если модель не уверена, значит человек просто стоит/идет
        return "standing"