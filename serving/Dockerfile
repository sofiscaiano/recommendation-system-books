FROM python:3.13-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        libgomp1 \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY main.py ./
RUN pip install --no-cache-dir ".[serve]"
COPY serving ./serving

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    SERVING_ARTIFACT=/models/serving.joblib \
    SERVING_DB=/data/data.db

EXPOSE 8000
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
