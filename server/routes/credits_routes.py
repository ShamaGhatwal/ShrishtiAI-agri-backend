"""Credits routes backed by Supabase user profiles."""
from __future__ import annotations

from typing import Any, Dict

from flask import Blueprint, jsonify, request

from routes.auth_routes import require_auth
from utils import create_error_response, create_success_response

credits_bp = Blueprint("credits", __name__)

# Set by init_credits_routes() from main.py
auth_controller = None

DEFAULT_CREDITS = 30

CREDIT_BUNDLES = [
    {"id": "plus_1000", "credits": 1000, "price_inr": 999, "label": "1000 Credits"},
    {"id": "plus_10000", "credits": 10000, "price_inr": 7999, "label": "10000 Credits"},
    {"id": "plus_100000", "credits": 100000, "price_inr": 59999, "label": "100000 Credits"},
]


def init_credits_routes(controller_instance):
    """Inject the AuthController instance so credits can use Supabase."""
    global auth_controller
    auth_controller = controller_instance


def _get_auth_service():
    if auth_controller is None:
        return None
    return getattr(auth_controller, "auth_service", None)


@credits_bp.route("/credits/balance", methods=["GET"])
@require_auth
def get_balance(user_id: str, access_token: str):
    auth_service = _get_auth_service()
    if auth_service is None:
        return jsonify(create_error_response("Auth service not initialized")), 503

    balance = auth_service.get_credit_balance(user_id)
    return jsonify(create_success_response({"data": {"credits": balance}}))


@credits_bp.route("/credits/bundles", methods=["GET"])
def get_bundles():
    return jsonify(create_success_response({"data": {"bundles": CREDIT_BUNDLES}}))


@credits_bp.route("/credits/purchase", methods=["POST"])
@require_auth
def purchase_bundle(user_id: str, access_token: str):
    auth_service = _get_auth_service()
    if auth_service is None:
        return jsonify(create_error_response("Auth service not initialized")), 503

    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    bundle_id = str(payload.get("bundle_id", "")).strip()

    selected = next((bundle for bundle in CREDIT_BUNDLES if bundle["id"] == bundle_id), None)
    if selected is None:
        return jsonify(create_error_response("Invalid bundle_id")), 400

    next_balance = auth_service.add_credits(user_id, int(selected["credits"]))
    if next_balance is None:
        return jsonify(create_error_response("Failed to update credit balance")), 500

    return jsonify(create_success_response({"data": {"remaining_credits": next_balance}}))


@credits_bp.route("/credits/deduct", methods=["POST"])
@require_auth
def deduct_credits(user_id: str, access_token: str):
    auth_service = _get_auth_service()
    if auth_service is None:
        return jsonify(create_error_response("Auth service not initialized")), 503

    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    amount = int(payload.get("amount", 0) or 0)
    if amount <= 0:
        return jsonify(create_error_response("amount must be a positive integer")), 400

    result = auth_service.deduct_credits(user_id, amount)
    if not result.get("success"):
        return (
            jsonify(
                create_error_response(
                    result.get("error_message", "Insufficient credits"),
                    {
                        "remaining_credits": result.get("remaining_credits", 0),
                        "required_credits": amount,
                    },
                )
            ),
            402,
        )

    return jsonify(create_success_response({"data": {"remaining_credits": result["remaining_credits"]}}))


@credits_bp.route("/credits/reset", methods=["POST"])
@require_auth
def reset_credits(user_id: str, access_token: str):
    auth_service = _get_auth_service()
    if auth_service is None:
        return jsonify(create_error_response("Auth service not initialized")), 503

    next_balance = auth_service.reset_credits(user_id, DEFAULT_CREDITS)
    if next_balance is None:
        return jsonify(create_error_response("Failed to reset credit balance")), 500

    return jsonify(create_success_response({"data": {"credits": next_balance}}))
