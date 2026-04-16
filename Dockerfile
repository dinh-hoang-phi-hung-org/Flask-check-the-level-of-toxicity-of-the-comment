# Stage 1: Install dependencies + pre-download HuggingFace model into image
FROM python:3.9-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m venv /venv && \
    /venv/bin/pip install --upgrade pip && \
    /venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    /venv/bin/pip install -r requirements.txt

# Bake vinai/phobert-base-v2 into the image at build time so there is no
# network download on container startup.
RUN /venv/bin/python -c "\
from transformers import AutoModel, AutoTokenizer; \
AutoModel.from_pretrained('vinai/phobert-base-v2'); \
AutoTokenizer.from_pretrained('vinai/phobert-base-v2'); \
print('PhoBERT model cached successfully')"

# Stage 2: Lean runtime image
FROM python:3.9-slim AS final

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_OFFLINE=1 \
    PATH="/venv/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /venv /venv

# Copy the pre-downloaded HuggingFace model cache
COPY --from=builder /app/.cache /app/.cache
COPY . .

EXPOSE 5050

# No --preload: model loads inside each worker via background thread (see app.py).
# This lets gunicorn start workers immediately without blocking on model load.
CMD ["gunicorn", "--bind", "0.0.0.0:5050", "--workers", "1", "--timeout", "300", "app:app"]
