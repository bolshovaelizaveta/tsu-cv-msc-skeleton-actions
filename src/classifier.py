class ActionClassifier:
    def __init__(self, model_path: str):
        # Здесь будет загрузка ST-GCN
        pass

    def predict(self, history_buffer: dict) -> dict:
        """
        Принимает буфер координат и возвращает название действия.
        """
        # Пока возвращает 'unknown'
        return {"action": "waiting"}