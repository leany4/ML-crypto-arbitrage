FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ML_CONFIG_DIR=/app/config \
    ML_MODELS_DIR=/models \
    ML_STATE_DB=/state/paper.sqlite3 \
    ML_DEVICE=auto

WORKDIR /app

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && if [ -n "$TORCH_INDEX_URL" ]; then \
         pip install "torch>=2.2,<3" --index-url "$TORCH_INDEX_URL"; \
       else \
         pip install "torch>=2.2,<3"; \
       fi

COPY src ./src
COPY config ./config
COPY scripts ./scripts
COPY models ./models-example
COPY README.md ./

RUN pip install --no-deps -e . \
    && mkdir -p /models /state \
    && useradd --create-home --uid 10001 app \
    && chown -R app:app /app /state

USER app

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
    CMD curl --fail http://127.0.0.1:8080/health/live || exit 1

CMD ["uvicorn", "ml_service.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
