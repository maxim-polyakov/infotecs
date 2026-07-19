# UEBA Prototype

Проект разделен на две части:

- `server/` — Python/FastAPI backend, внутри которого находится весь UEBA/ML-макет: сбор телеметрии, признаки, автоэнкодер, детектор и отчеты.
- `client/` — React UI для запуска demo, сбора, обучения, проверки текущего окна и просмотра аномалий.

Backend внутри `server/ueba_server` разложен по слоям в стиле `lotus_game`: `controller/`, `dto/`, `service/`, `config/`. ML-логика вынесена отдельно в `server/ueba_prototype`.

## Быстрый запуск

Docker Compose:

```powershell
docker compose up --build
```

После запуска:

- backend: `http://127.0.0.1:8000`
- API docs: `http://127.0.0.1:8000/docs`
- frontend: `http://127.0.0.1:5173`

Важно: при запуске в Docker сборщик `psutil` видит окружение контейнера. Для анализа именно локального Windows/Linux-хоста запускайте backend напрямую через Python.

Backend:

```powershell
cd server\ueba_server
python api.py
```

Сервер запустится на `http://127.0.0.1:8000`.

Если зависимости еще не установлены:

```powershell
cd server
python -m pip install -e ".[dev]"
cd ueba_server
python api.py
```

Ручной запуск:

```powershell
cd server
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m uvicorn ueba_server.api:app --reload --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
cd client
npm install
npm run dev
```

Откройте `http://127.0.0.1:5173`. Документация API доступна на `http://127.0.0.1:8000/docs`.

## Основные сценарии

Через React-интерфейс можно:

- проверить доступность backend и статус модели;
- запустить demo с синтетическими аномалиями;
- выполнить короткий тестовый сбор;
- обучить модель на `data/raw.csv`;
- проверить текущее окно активности;
- запустить детектор на ограниченное число окон;
- посмотреть последние отчеты и фоновые jobs.

Для финального сбора обучающей выборки используйте не менее 24 часов:

```bash
cd server
python -m ueba_prototype collect --duration-hours 24 --interval 5 --out data/raw.csv
python -m ueba_prototype train --data data/raw.csv --model-dir models/default
python -m ueba_prototype detect --model-dir models/default --interval 5 --reports reports/anomalies.jsonl
```

## Что анализируется

Backend собирает кроссплатформенную телеметрию через `psutil`: CPU/RAM/swap, процессы, новые процессы, потоки, открытые файлы, сетевые соединения, удаленные IP/порты, нестандартные порты, скорость передачи байт и пакетов.

## Проверка

```bash
cd server
python -m pytest
python -m ueba_prototype demo --output-dir reports/demo
```
