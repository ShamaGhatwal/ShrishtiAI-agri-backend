"""API key compatibility routes for the web dashboard settings page."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from uuid import uuid4

from flask import Blueprint, jsonify, request

from utils import create_error_response, create_success_response

api_keys_bp = Blueprint("api_keys", __name__)

API_COSTS = {
    "hazardguard": 10,
    "weatherwise": 5,
    "geovision": 8,
    "data_layers": 2,
    "chatbot": 1,
    "timelapse": 4,
}

# In-memory per-user API key store for compatibility mode.
USER_KEYS: Dict[str, List[Dict[str, Any]]] = {}


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _user_key() -> str:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth.split(" ", 1)[1].strip()
        return token or "anonymous"
    return "anonymous"


def _ensure_store(user: str) -> List[Dict[str, Any]]:
    if user not in USER_KEYS:
        USER_KEYS[user] = []
    return USER_KEYS[user]


def _public_key_view(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": item["id"],
        "name": item["name"],
        "key_prefix": item["key_prefix"],
        "permissions": item["permissions"],
        "is_active": item["is_active"],
        "last_used_at": item.get("last_used_at"),
        "usage_count": int(item.get("usage_count", 0)),
        "credits_consumed": int(item.get("credits_consumed", 0)),
        "created_at": item["created_at"],
        "expires_at": item.get("expires_at"),
    }


@api_keys_bp.route("/api-keys", methods=["GET"])
def list_api_keys():
    keys = _ensure_store(_user_key())
    active_first = sorted(keys, key=lambda k: (not bool(k.get("is_active", False)), k.get("created_at", "")), reverse=False)
    return jsonify(create_success_response({"data": {"keys": [_public_key_view(k) for k in active_first]}}))


@api_keys_bp.route("/api-keys", methods=["POST"])
def create_api_key():
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    name = str(payload.get("name", "")).strip()
    if not name:
        return jsonify(create_error_response("name is required")), 400

    permissions = payload.get("permissions")
    if not isinstance(permissions, list) or not permissions:
        permissions = ["read"]

    raw_key = f"sk_{uuid4().hex}{uuid4().hex}"
    key_id = f"key_{uuid4().hex[:12]}"
    key_prefix = raw_key[:12]

    expires_at = payload.get("expires_at")
    if expires_at is None and payload.get("expires_in_days"):
        try:
            expires_days = int(payload["expires_in_days"])
            if expires_days > 0:
                expires_at = (datetime.now(timezone.utc) + timedelta(days=expires_days)).isoformat()
        except (TypeError, ValueError):
            expires_at = None

    item = {
        "id": key_id,
        "name": name,
        "raw_key": raw_key,
        "key_prefix": key_prefix,
        "permissions": permissions,
        "is_active": True,
        "last_used_at": None,
        "usage_count": 0,
        "credits_consumed": 0,
        "created_at": _iso_now(),
        "expires_at": expires_at,
    }

    _ensure_store(_user_key()).append(item)

    return jsonify(create_success_response({
        "data": {
            "api_key": raw_key,
            "key_id": key_id,
            "name": name,
            "key_prefix": key_prefix,
            "permissions": permissions,
            "created_at": item["created_at"],
            "expires_at": expires_at,
        }
    }))


@api_keys_bp.route("/api-keys/<string:key_id>/revoke", methods=["POST"])
def revoke_api_key(key_id: str):
    keys = _ensure_store(_user_key())
    for item in keys:
        if item["id"] == key_id:
            item["is_active"] = False
            return jsonify(create_success_response({"data": {"id": key_id, "revoked": True}}))

    return jsonify(create_error_response("API key not found")), 404


@api_keys_bp.route("/api-keys/<string:key_id>", methods=["DELETE"])
def delete_api_key(key_id: str):
    user = _user_key()
    keys = _ensure_store(user)
    kept = [item for item in keys if item["id"] != key_id]
    if len(kept) == len(keys):
        return jsonify(create_error_response("API key not found")), 404

    USER_KEYS[user] = kept
    return jsonify(create_success_response({"data": {"id": key_id, "deleted": True}}))


@api_keys_bp.route("/api-keys/usage", methods=["GET"])
def api_key_usage():
    key_id = request.args.get("key_id")
    keys = _ensure_store(_user_key())

    filtered = keys
    if key_id:
        filtered = [item for item in keys if item["id"] == key_id]

    total_calls = sum(int(item.get("usage_count", 0)) for item in filtered)
    total_credits = sum(int(item.get("credits_consumed", 0)) for item in filtered)

    return jsonify(create_success_response({
        "data": {
            "total_calls": total_calls,
            "total_credits": total_credits,
            "endpoint_breakdown": {},
            "recent_logs": [],
        }
    }))


@api_keys_bp.route("/api-keys/costs", methods=["GET"])
def api_key_costs():
    return jsonify(create_success_response({"data": {"costs": API_COSTS}}))
