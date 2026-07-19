# UEBA Prototype Client

React UI for the local UEBA backend.

## Install

```bash
npm install
```

## Run

Start the backend first on `http://127.0.0.1:8000`, then run:

```bash
npm run dev
```

## Docker

From the repository root:

```bash
docker compose up --build
```

Client-only build:

```bash
docker build -t ueba-client ./client
docker run --rm -p 5173:5173 -e VITE_API_URL=http://127.0.0.1:8000 ueba-client
```

If the backend uses another URL, create `.env`:

```bash
VITE_API_URL=http://127.0.0.1:8000
```
