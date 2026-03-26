# PAV Detector

Программный комплекс для детекта VPN/PROXY-соединений на основе машинного обучения по **flow-признакам** (без анализа полезной нагрузки).

## Технологический стек

- Python 3.10+
- CICFlowMeter (извлечение flow-признаков из PCAP/интерфейса)
- pandas / numpy / joblib / python-dotenv
- ONNX Runtime (приоритетный runtime инференса) + PyTorch fallback
- FastAPI + Uvicorn (realtime API)
- PostgreSQL + psycopg2 (хранилище событий)
- Streamlit (UI оператора)
- logging (JSON-логи)

## Архитектура

Система построена модульно:

1. **Flow Extraction**  
   Вход: PCAP/PCAPNG (offline) или уже агрегированные flow-объекты (realtime).  
   Для PCAP используется CICFlowMeter.

2. **Inference Engine** (`pav_detector/core/inference.py`)  
   - Загружает `scaler.pkl` (если есть).
   - Пытается загрузить `models/model.onnx` через ONNX Runtime.
   - Если ONNX нет — пытается `models/model.pt` (TorchScript).
   - Возвращает вероятности классов.

3. **Decision Logic** (`pav_detector/core/decision.py`)  
   - `LEGIT` => не алерт.
   - `VPN/PROXY` и `confidence >= THRESHOLD` => формирует событие.

4. **Storage** (`pav_detector/db/postgres.py`)  
   Хранит только подозрительные события:
   - тип события (`VPN`/`PROXY`)
   - confidence
   - сетевые метаданные (src/dst/protocol)
   - `raw_flow` в JSONB

5. **Operator UI** (`pav_detector/ui/streamlit_app.py`)  
   - фильтры по типу и сенсору
   - таблица событий
   - базовая статистика

Общий сервис `DetectionService` используется и для offline, и для realtime режима.

## Структура проекта

```text
src/pav_detector/
  api/app.py               # FastAPI realtime API
  core/
    inference.py           # загрузка модели + инференс
    decision.py            # правила порога/алерта
    preprocessing.py       # подготовка признаков
    service.py             # общий orchestration слой
  db/
    postgres.py            # init schema, insert/list events
    init_db.py             # инициализация БД
  offline/
    cicflow.py             # запуск CICFlowMeter
    run_offline.py         # offline классификация CSV/PCAP
  ui/streamlit_app.py      # UI оператора
  utils/logging_json.py    # JSON-формат логов
```

## Быстрый старт

### One-command setup (после `git clone`)

```bash
cd PAV_detector
bash scripts/setup.sh
```

Скрипт автоматически:
- создаёт `.env` из `.env.example` (если ещё нет),
- создаёт папки `models/` и `data/`,
- ставит зависимости (через `.venv`, а если `venv` недоступен — через `python3 -m pip --user`),
- создаёт БД PostgreSQL (если не существует),
- создаёт таблицы/индексы.

Опции:
- `USE_VENV=1 bash scripts/setup.sh` — строго использовать виртуальное окружение.
- `USE_VENV=0 bash scripts/setup.sh` — ставить в user-окружение без `.venv`.

### 1) Установка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Либо как пакет:

```bash
pip install -e .
```

### 2) Настройка

```bash
cp .env.example .env
```

Отредактируйте:
- `POSTGRES_DSN`
- `THRESHOLD`
- `MODEL_ONNX_PATH` / `MODEL_TORCH_PATH`
- `SCALER_PATH`
- `FEATURE_ORDER` (опционально, через запятую)

### 3) Подготовка БД

```bash
PYTHONPATH=src python -m pav_detector.db.init_db
```

или:

```bash
pav-init-db
```

### 4) Realtime API

```bash
PYTHONPATH=src uvicorn pav_detector.api.app:app --host 0.0.0.0 --port 8000
```

Проверка:

```bash
curl http://localhost:8000/health
```

Пример запроса:

```bash
curl -X POST http://localhost:8000/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_name": "sensor-a",
    "flow": {
      "Src IP": "10.0.0.1",
      "Dst IP": "8.8.8.8",
      "Protocol": "6",
      "Flow Duration": 12345,
      "Tot Fwd Pkts": 20,
      "Tot Bwd Pkts": 15
    }
  }'
```

### 5) Offline режим

Из уже готового CSV:

```bash
PYTHONPATH=src python -m pav_detector.offline.run_offline \
  --csv path/to/flows.csv \
  --sensor-name offline-lab \
  --output-json data/offline_results.json
```

Из PCAP через CICFlowMeter:

```bash
PYTHONPATH=src python -m pav_detector.offline.run_offline \
  --pcap path/to/capture.pcapng \
  --generated-csv data/generated_flows.csv
```

### 6) UI оператора

```bash
PYTHONPATH=src streamlit run src/pav_detector/ui/streamlit_app.py
```

## Важные замечания

- Модель и scaler в репозиторий не включены: положите артефакты в `models/`.
- Порядок классов в `.env` (`CLASSES`) должен совпадать с обучением модели.
- Если задаёте `FEATURE_ORDER`, он должен совпадать с feature engineering на этапе train.
- В БД сохраняются только подозрительные события (VPN/PROXY выше порога), а не весь трафик.
