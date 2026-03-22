### Распознавание действий по скелетным данным
Проект реализуется для онлайн-кинотеатра КИОН. Цель — разработка прототипа системы распознавания действий человека (индивидуальных и групповых) по скелетным данным (ключевым точкам).

### Предполагаемая структура проекта

```text
tsu-cv-msc-skeleton-actions/
├── data/                      # Видео, JSON-результаты, датасеты
├── models/                    # Веса нейросетей (.pt)
├── src/                       # Исходный код классов              
│   ├── detector.py            # Модуль извлечения скелетов (YOLOv11 + Tracker)
│   ├── classifier.py          # Модуль классификации индивидуальных действий (ST-GCN / LSTM/ST-GCN )
│   ├── analyzer.py            # Модуль геометрического анализа групповых действий
│   └── visualizer.py          # Модуль отрисовки скелетов и аналитики на видео
│   ├── sequence_buffer.py     # Буфер временных рядов 
│   ├── skeleton_adapter.py    # Нормализация скелетов 
│   ├── datasets/              # Классы для загрузки данных
│   │   └── ntu_dataset.py
│   └── classifiers/           #  Архитектуры нейросетей 
│       └── ntu_baseline.py
├── scripts/                   # Вспомогательные скрипты
│   ├── generate_dummy_ntu.py  # Генерация фейковых данных 
│   └── infer_video_modular.py # Альтернативный запуск 
├── training/                  # Скрипты для обучения моделей
│   ├── train_ntu.py           # Обучение LSTM 
│   └── utils.py               # Вспомогательные функции для лосса/метрик
├── requirements.txt           # Зависимости проекта
├── README.md                  # Описание проекта
└── main.py                    # Точка входа: сборка пайплайна и запуск аналитики
```

### Запуск проекта 

1. **Окружение:** Python 3.10+.
   ```bash
   python -m venv venv
   source venv/bin/activate  # для Mac/Linux
   pip install -r requirements.txt
   ```

2. **Настройка устройства**
   В файле `main.py` при инициализации `PoseDetector` укажите ваше устройство:
   - `device='mps'` — для Mac 
   - `device='cuda'` — для GPU
   - `device='cpu'` — если нет видеокарты

3. **Запуск:**
   ```bash
   python main.py
   ```


## Запуск проекта (Docker)

### Проект запускается через Docker с поддержкой GPU.

1. **Сборка и запуск контейнеров**

   ```bash
   docker compose up --build -d
   ```

   Будут подняты сервисы:

   - hakaton-kion-mmaction2
   - hakaton-kion-vlm

2. **Установка зависимостей**

   **MMACTION2 контейнер**
   ```bash
   docker exec -it hakaton-kion-mmaction2 bash
   pip install -r requirements.txt
   ```

   **VLM контейнер**

   ```bash
   docker exec -it hakaton-kion-vlm bash
   pip install -r requirements.txt
   ```

3. **Настройка VLM (Qwen3-VL-4B-Instruct)**

   Выполняется внутри контейнера **vlm**.

   Вход в контейнер

   ```bash
   docker exec -it hakaton-kion-vlm bash
   ```

   Выдача прав на скрипты

   ```bash
   chmod 755 scripts/vlm/check_and_load_model.sh
   chmod 755 scripts/vlm/run_test.sh
   ```

   Загрузка модели

   ```bash
   bash scripts/vlm/check_and_load_model.sh
   ```

   Скрипт автоматически:

   - проверит наличие модели
   - скачает Qwen3-VL-4B-Instruct при необходимости
   - подготовит окружение

4. **Проверка работы VLM**

   ```bash
   bash scripts/vlm/run_test.sh
   ```

5. **Запуск VLM API**

   ```bash
   docker exec -it hakaton-kion-vlm bash
   python src/vlm/vlm_api.py
   ```

   **После запуска:**

      - поднимается API-сервер
      - модель загружается в память
      - сервис готов принимать запросы от vlm_client.py

   **⚠️ Важно:**

      - сервер должен быть запущен до запуска основного пайплайна
      - порт и хост задаются внутри vlm_api.py

6. **Запуск основного пайплайна**

   **В отдельном терминале:**

   ```bash
   docker exec -it hakaton-kion-mmaction2 bash
   python scripts/vlm/infer_vlm.py
   ```

   **Взаимодействие сервисов**

   - mmaction2 — обработка видео и скелетов
   - vlm — VLM сервер (Qwen3-VL-4B-Instruct)
   - взаимодействие происходит через API (vlm_api.py)