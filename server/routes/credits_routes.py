"""Credits compatibility routes used by the web dashboard."""
from __future__ import annotations

from typing import Any, Dict

from flask import Blueprint, jsonify, request

from utils import create_error_response, create_success_response

credits_bp = Blueprint("credits", __name__)

DEFAULT_CREDITS = 30

CREDIT_BUNDLES = [
    {"id": "plus_1000", "credits": 1000, "price_inr": 999, "label": "1000 Credits"},
    {"id": "plus_10000", "credits": 10000, "price_inr": 7999, "label": "10000 Credits"},
    {"id": "plus_100000", "credits": 100000, "price_inr": 59999, "label": "100000 Credits"},
]

USER_CREDITS: Dict[str, int] = {}


def _user_key() -> str:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth.split(" ", 1)[1].strip() or "anonymous"
    return "anonymous"


def _get_balance(key: str) -> int:
    if key not in USER_CREDITS:
        USER_CREDITS[key] = DEFAULT_CREDITS
    return USER_CREDITS[key]


def _set_balance(key: str, value: int) -> int:
    USER_CREDITS[key] = max(0, int(value))
    return USER_CREDITS[key]


@credits_bp.route("/credits/balance", methods=["GET"])
def get_balance():
    key = _user_key()
    balance = _get_balance(key)
    return jsonify(create_success_response({"data": {"credits": balance}}))


@credits_bp.route("/credits/bundles", methods=["GET"])
def get_bundles():
    return jsonify(create_success_response({"data": {"bundles": CREDIT_BUNDLES}}))


@credits_bp.route("/credits/purchase", methods=["POST"])
def purchase_bundle():
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    bundle_id = str(payload.get("bundle_id", "")).strip()

    selected = next((bundle for bundle in CREDIT_BUNDLES if bundle["id"] == bundle_id), None)
    if selected is None:
        return jsonify(create_error_response("Invalid bundle_id")), 400

    key = _user_key()
    new_balance = _set_balance(key, _get_balance(key) + int(selected["credits"]))
    return jsonify(create_success_response({"data": {"remaining_credits": new_balance}}))


@credits_bp.route("/credits/deduct", methods=["POST"])
def deduct_credits():
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    amount = int(payload.get("amount", 0) or 0)
    if amount <= 0:
        return jsonify(create_error_response("amount must be a positive integer")), 400

    key = _user_key()
    current = _get_balance(key)
    if current < amount:
        return (
            jsonify(
                create_error_response(
                    "Insufficient credits",
                    {
                        "remaining_credits": current,
                        "required_credits": amount,
                    },
                )
            ),
            402,
        )

    new_balance = _set_balance(key, current - amount)
    return jsonify(create_success_response({"data": {"remaining_credits": new_balance}}))


@credits_bp.route("/credits/reset", methods=["POST"])
def reset_credits():
    key = _user_key()
    new_balance = _set_balance(key, DEFAULT_CREDITS)
    return jsonify(create_success_response({"data": {"credits": new_balance}}))
