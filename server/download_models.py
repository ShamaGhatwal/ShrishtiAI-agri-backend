"""Downloads model files from private HF model repo at Docker build time."""
import os
from huggingface_hub import snapshot_download

token = os.environ.get("HF_TOKEN")
if not token:
    raise RuntimeError("HF_TOKEN secret is not set — cannot download private model repo")

print("Downloading models from projectgaia/ShrishtiAI-models ...")
snapshot_download(
    repo_id="projectgaia/ShrishtiAI-models",
    repo_type="model",
    local_dir="/app/models",
    token=token,
    ignore_patterns=["*.git*", ".gitattributes"],
)
print("Models downloaded successfully.")
