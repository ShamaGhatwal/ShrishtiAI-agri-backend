"""
Auth Routes
RESTful endpoints for authentication, profile, and activity logging.
All database communication goes through the backend — the frontend
never talks to Supabase directly.
"""
from flask import Blueprint, request, jsonify
import logging
from functools import wraps

auth_bp = Blueprint("auth", __name__)
logger = logging.getLogger(__name__)

# Will be set by init_auth_routes()
auth_controller = None


def init_auth_routes(controller_instance):
    """Inject the AuthController instance."""
    global auth_controller
    auth_controller = controller_instance
    logger.info("Auth routes initialized with controller")


# ── Middleware helper ───────────────────────────────────────────────────

def require_auth(f):
    """
    Decorator that verifies the Authorization header and injects
    `user_id` into kwargs so the route handler can use it.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if auth_controller is None:
            return jsonify({"status": "error", "error": "Auth service not initialized"}), 503

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"status": "error", "error": "Missing or invalid Authorization header"}), 401

        token = auth_header.split(" ", 1)[1]
        result = auth_controller.get_me(token)

        if result.get("status") != "success":
            return jsonify({"status": "error", "error": "Invalid or expired token"}), 401

        # Attach user_id + token to kwargs
        kwargs["user_id"] = result["user"]["id"]
        kwargs["access_token"] = token
        return f(*args, **kwargs)
    return decorated


# ════════════════════════════════════════════════════════════════════════
#  PUBLIC AUTH ENDPOINTS
# ════════════════════════════════════════════════════════════════════════

@auth_bp.route("/auth/login", methods=["POST"])
def login():
    """POST /api/auth/login  –  sign in with email + password"""
    try:
        if auth_controller is None:
            return jsonify({"status": "error", "error": "Auth service not initialized"}), 503

        data = request.get_json() or {}
        data["device_info"] = request.headers.get("User-Agent", "")[:120]
        result = auth_controller.login(data)
        code = 200 if result.get("status") == "success" else 400
        return jsonify(result), code

    except Exception as e:
        logger.error(f"Login route error: {e}")
        return jsonify({"status": "error", "error": "Internal server error"}), 500


@auth_bp.route("/auth/signup", methods=["POST"])
def signup():
    """POST /api/auth/signup  –  register a new account"""
    try:
        if auth_controller is None:
            return jsonify({"status": "error", "error": "Auth service not initialized"}), 503

        data = request.get_json() or {}
        data["device_info"] = request.headers.get("User-Agent", "")[:120]
        result = auth_controller.signup(data)
        code = 201 if result.get("status") == "success" else 400
        return jsonify(result), code

    except Exception as e:
        logger.error(f"Signup route error: {e}")
        return jsonify({"status": "error", "error": "Internal server error"}), 500


@auth_bp.route("/auth/refresh", methods=["POST"])
def refresh():
    """POST /api/auth/refresh  –  refresh an expired session"""
    try:
        if auth_controller is None:
            return jsonify({"status": "error", "error": "Auth service not initialized"}), 503

        data = request.get_json() or {}
        result = auth_controller.refresh(data)
        code = 200 if result.get("status") == "success" else 401
        return jsonify(result), code

    except Exception as e:
        logger.error(f"Refresh route error: {e}")
        return jsonify({"status": "error", "error": "Internal server error"}), 500


@auth_bp.route("/auth/resend-verification", methods=["POST"])
def resend_verification():
    """POST /api/auth/resend-verification  –  resend email verification link"""
    try:
        if auth_controller is None:
            return jsonify({"status": "error", "error": "Auth service not initialized"}), 503

        data = request.get_json() or {}
        result = auth_controller.resend_verification(data)
        code = 200 if result.get("status") == "success" else 400
        return jsonify(result), code

    except Exception as e:
        logger.error(f"Resend verification route error: {e}")
        return jsonify({"status": "error", "error": "Internal server error"}), 500


# ════════════════════════════════════════════════════════════════════════
#  PROTECTED ENDPOINTS  (require_auth injects user_id + access_token)
# ════════════════════════════════════════════════════════════════════════

@auth_bp.route("/auth/me", methods=["GET"])
@require_auth
def get_me(user_id: str, access_token: str):
    """GET /api/auth/me  –  get current user info"""
    result = auth_controller.get_me(access_token)
    return jsonify(result), 200


@auth_bp.route("/auth/logout", methods=["POST"])
@require_auth
def logout(user_id: str, access_token: str):
    """POST /api/auth/logout"""
    result = auth_controller.logout(access_token, user_id)
    return jsonify(result), 200


# ── Profile ─────────────────────────────────────────────────────────────

@auth_bp.route("/auth/profile", methods=["GET"])
@require_auth
def get_profile(user_id: str, access_token: str):
    """GET /api/auth/profile"""
    result = auth_controller.get_profile(user_id)
    code = 200 if result.get("status") == "success" else 404
    return jsonify(result), code


@auth_bp.route("/auth/profile", methods=["PUT"])
@require_auth
def update_profile(user_id: str, access_token: str):
    """PUT /api/auth/profile"""
    data = request.get_json() or {}
    result = auth_controller.update_profile(user_id, data)
    code = 200 if result.get("status") == "success" else 400
    return jsonify(result), code


# ── Activity Logs ───────────────────────────────────────────────────────

@auth_bp.route("/auth/activity", methods=["POST"])
@require_auth
def log_activity(user_id: str, access_token: str):
    """POST /api/auth/activity"""
    data = request.get_json() or {}
    data["device_info"] = request.headers.get("User-Agent", "")[:120]
    result = auth_controller.log_activity(user_id, data)
    code = 200 if result.get("status") == "success" else 400
    return jsonify(result), code


@auth_bp.route("/auth/activity", methods=["GET"])
@require_auth
def get_activity_logs(user_id: str, access_token: str):
    """GET /api/auth/activity?limit=30"""
    limit = request.args.get("limit", 50, type=int)
    result = auth_controller.get_activity_logs(user_id, limit)
    return jsonify(result), 200


# ── Health ──────────────────────────────────────────────────────────────

@auth_bp.route("/auth/health", methods=["GET"])
def auth_health():
    """GET /api/auth/health"""
    if auth_controller is None:
        return jsonify({"status": "error", "error": "Auth service not initialized"}), 503
    return jsonify({"status": "success", "service": "auth", "healthy": True}), 200
