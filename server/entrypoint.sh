#!/bin/bash
set -e

# Download models from private HF model repo if not already present
if [ ! -f "/app/models/weatherwise/normal/best_model.keras" ]; then
    echo "[ENTRYPOINT] Downloading models from HuggingFace..."
    python download_models.py
    echo "[ENTRYPOINT] Models ready."
else
    echo "[ENTRYPOINT] Models already present, skipping download."
fi

# Start gunicorn
exec gunicorn main:app \
    --bind 0.0.0.0:7860 \
    --workers 1 \
    --threads 2 \
    --timeout 300 \
    --keep-alive 5 \
    --access-logfile - \
    --error-logfile -
