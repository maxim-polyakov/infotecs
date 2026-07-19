# UEBA Prototype Server

FastAPI backend for the UEBA prototype. It contains the telemetry collector, feature processing, autoencoder model, detector, reporting logic, and HTTP API used by the React client.

## Structure

The server is organized similarly to the layered backend in `lotus_game`:

- `ueba_server/controller/` - FastAPI routers, similar to Spring controllers.
- `ueba_server/dto/` - Pydantic request/response DTOs.
- `ueba_server/service/` - application services and background job orchestration.
- `ueba_server/config/` - FastAPI app and CORS configuration.
- `api.py` - simple launcher from the `server` folder.
- `ueba_server/api.py` - FastAPI application entrypoint and direct launcher from `server/ueba_server`.
- `ueba_prototype/` - UEBA domain/ML code: collector, features, model, detector, reporter, CLI.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Run API

```bash
cd ueba_server
python api.py
```

Open API docs: `http://127.0.0.1:8000/docs`.

## Docker

From the repository root:

```bash
docker compose up --build
```

Server-only build:

```bash
docker build -t ueba-server ./server
docker run --rm -p 8000:8000 ueba-server
```

## CLI

The original CLI is still available:

```bash
python -m ueba_prototype demo --output-dir reports/demo
python -m ueba_prototype collect --duration-hours 24 --interval 5 --out data/raw.csv
python -m ueba_prototype train --data data/raw.csv --model-dir models/default
python -m ueba_prototype detect --model-dir models/default --interval 5 --reports reports/anomalies.jsonl
```
