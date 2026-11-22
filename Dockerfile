FROM python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# Hash-locked, CPU-only dependencies (see scripts/lock_deps.sh).
COPY requirements.lock .
RUN pip install --no-cache-dir --no-deps --require-hashes \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.lock

# Bake the pinned model revisions into the image; no HF access at runtime.
COPY app/__init__.py app/config.py ./app/
COPY scripts/download_models.py ./scripts/
RUN python -m scripts.download_models
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY app ./app
COPY manifests/v3.json ./manifests/v3.json

# Data and indices are NOT baked in; mount them read-only at runtime
# (see docker-compose.yml). Startup verifies artifacts before serving.
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# --no-proxy-headers: rate limiting keys unauthenticated clients on the TCP
# peer address; never let a client-supplied X-Forwarded-For replace it.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-proxy-headers", "--no-access-log"]
