# ============================================================
# GeoVision Backend — Hugging Face Spaces (Docker)
# Port: 7860 (required by HF Spaces)
# ============================================================

FROM python:3.11-slim

# System libs needed by rasterio / pyproj pip wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 \
    libgomp1 \
    curl \
    git \
    git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── 1. Install Python deps (cached layer — only rebuilds when requirements change) ──
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── 2. Copy backend source (no model files) ──
COPY server/ .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# ── 3. Non-sensitive runtime defaults (secrets injected via HF Space env vars) ──
ENV FLASK_ENV=production \
    FLASK_DEBUG=False \
    FLASK_HOST=0.0.0.0 \
    FLASK_PORT=7860 \
    LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1 \
    # Disable TF GPU detection noise
    TF_CPP_MIN_LOG_LEVEL=3 \
    CUDA_VISIBLE_DEVICES=""

EXPOSE 7860

# Entrypoint downloads models at startup (needs HF_TOKEN secret), then starts gunicorn
CMD ["./entrypoint.sh"]
