"""Shared path helpers for the backend."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from dotenv import load_dotenv

SERVER_DIR = Path(__file__).resolve().parent.parent


def _find_repo_root() -> Path:
    """Find the checkout root by looking for the shared models directory."""
    fallback_with_models: Path | None = None
    for candidate in (SERVER_DIR, *SERVER_DIR.parents):
        if not (candidate / "models").exists():
            continue

        if fallback_with_models is None:
            fallback_with_models = candidate

        # Prefer the real workspace root when available.
        if (candidate / "backend").exists() and (candidate / "web").exists():
            return candidate
        if (candidate / ".git").exists():
            return candidate

    return fallback_with_models or SERVER_DIR


REPO_ROOT = _find_repo_root()

# Load the server-local .env file regardless of the current working directory.
load_dotenv(SERVER_DIR / ".env")


def resolve_path(value: str, base_dir: Path = SERVER_DIR) -> Optional[Path]:
    """Resolve a relative or absolute filesystem path."""
    if not value:
        return None

    candidate = Path(value.strip())
    if candidate.is_absolute():
        return candidate.resolve()

    return (base_dir / candidate).resolve()


def get_local_model_root() -> Path:
    """Return the local filesystem root used for loaded model artifacts."""
    raw_value = os.getenv("MODEL_ROOT_PATH", "").strip()

    if not raw_value:
        return REPO_ROOT / "models"

    if raw_value.startswith(("http://", "https://")):
        return REPO_ROOT / "models"

    resolved = resolve_path(raw_value, base_dir=REPO_ROOT)
    return resolved if resolved is not None else (REPO_ROOT / "models")


def get_model_repo_id() -> str:
    """Return the Hugging Face repository id used to bootstrap models."""
    repo_id = os.getenv("MODEL_REPO_ID", "").strip()
    if repo_id:
        return repo_id

    raw_value = os.getenv("MODEL_ROOT_PATH", "").strip()
    if raw_value.startswith(("http://", "https://")):
        parsed = urlparse(raw_value)
        return parsed.path.strip("/")

    return ""
