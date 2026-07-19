FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UEBA_HOST=0.0.0.0
ENV UEBA_PORT=8000

WORKDIR /app

COPY pyproject.toml README.md ./
COPY ueba_prototype ./ueba_prototype
COPY ueba_server ./ueba_server

RUN python -m pip install --upgrade pip \
    && python -m pip install .

EXPOSE 8000

CMD ["python", "ueba_server/api.py"]
