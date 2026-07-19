FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SIM_DATA_DIR=/data \
    SIM_PREPARED_DIR=/prepared \
    SIM_ML_URL=http://ml-service:8080

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

EXPOSE 8090

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8090/health/live', timeout=2)"]

CMD ["uvicorn", "market_simulator.main:app", "--host", "0.0.0.0", "--port", "8090"]
