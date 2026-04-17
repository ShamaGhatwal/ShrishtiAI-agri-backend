"""Downloads model files from Hugging Face at Docker build time."""
from __future__ import annotations

import os

from huggingface_hub import snapshot_download

from config.paths import get_model_repo_id, get_local_model_root

MODEL_REPO_ID = get_model_repo_id()
LOCAL_MODEL_DIR = get_local_model_root()
MODEL_REPO_TYPE = os.getenv("MODEL_REPO_TYPE", "model").strip() or "model"

if not MODEL_REPO_ID:
    raise RuntimeError(
        "MODEL_REPO_ID or MODEL_ROOT_PATH must point to the Hugging Face model repository"
    )

hf_token = os.environ.get("HF_TOKEN")

print(f"Downloading models from {MODEL_REPO_ID} ...")
snapshot_download(
    repo_id=MODEL_REPO_ID,
    repo_type=MODEL_REPO_TYPE,
    local_dir=str(LOCAL_MODEL_DIR),
    token=hf_token,
    ignore_patterns=["*.git*", ".gitattributes"],
)
print(f"Models downloaded successfully to {LOCAL_MODEL_DIR}.")
