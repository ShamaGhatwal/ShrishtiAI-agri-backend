"""
Auth Controller
Handles authentication, profile, and activity-log operations.
Sits between routes and the AuthService.
"""
import logging
from typing import Dict, Any
from services.auth_service import AuthService
from models.auth_model import (
    LoginRequest, SignUpRequest, ProfileUpdate, ActivityLogEntry
)
from utils import create_error_response, create_success_response

logger = logging.getLogger(__name__)


class AuthController:
    """Controller for all auth-related endpoints"""

    def __init__(self, auth_service: AuthService):
        self.auth_service = auth_service

    # ── Auth ────────────────────────────────────────────────────────────

    def login(self, data: Dict[str, Any]) -> Dict[str, Any]:
        req = LoginRequest(
            email=data.get("email", ""),
            password=data.get("password", ""),
        )
        errors = req.validate()
        if errors:
            return create_error_response("Validation failed", {"errors": errors})

        ok, result = self.auth_service.sign_in(req.email, req.password)
        if not ok:
            return create_error_response(result.get("error", "Login failed"))

        # Log the login activity
        self.auth_service.log_activity(
            user_id=result["user"]["id"],
            activity_type="login",
            description=f"User logged in: {req.email}",
            device_info=data.get("device_info"),
        )

        return create_success_response(result, "Login successful")

    def signup(self, data: Dict[str, Any]) -> Dict[str, Any]:
        req = SignUpRequest(
            email=data.get("email", ""),
            password=data.get("password", ""),
            full_name=data.get("full_name", ""),
            organization=data.get("organization"),
            purpose=data.get("purpose"),
        )
        errors = req.validate()
        if errors:
            return create_error_response("Validation failed", {"errors": errors})

        ok, result = self.auth_service.sign_up(
            email=req.email,
            password=req.password,
            full_name=req.full_name,
            organization=req.organization,
            purpose=req.purpose,
        )
        if not ok:
            return create_error_response(result.get("error", "Signup failed"))

        # Log the signup activity
        self.auth_service.log_activity(
            user_id=result["user"]["id"],
            activity_type="signup",
            description=f"New user registered: {req.email}",
            device_info=data.get("device_info"),
        )

        return create_success_response(result, "Account created successfully")

    def logout(self, access_token: str, user_id: str) -> Dict[str, Any]:
        self.auth_service.sign_out(access_token)
        self.auth_service.log_activity(
            user_id=user_id,
            activity_type="logout",
            description="User logged out",
        )
        return create_success_response(None, "Logged out")

    def get_me(self, access_token: str) -> Dict[str, Any]:
        """Verify token and return current user info."""
        ok, user = self.auth_service.verify_token(access_token)
        if not ok or not user:
            return create_error_response("Invalid or expired token")
        return create_success_response({"user": user})

    def refresh(self, data: Dict[str, Any]) -> Dict[str, Any]:
        refresh_token = data.get("refresh_token", "")
        if not refresh_token:
            return create_error_response("refresh_token is required")
        ok, result = self.auth_service.refresh_session(refresh_token)
        if not ok:
            return create_error_response(result.get("error", "Refresh failed"))
        return create_success_response(result, "Session refreshed")

    def resend_verification(self, data: Dict[str, Any]) -> Dict[str, Any]:
        email = data.get("email", "").strip()
        if not email:
            return create_error_response("Email is required")
        ok, err = self.auth_service.resend_verification_email(email)
        if not ok:
            return create_error_response(err or "Failed to resend verification email")
        return create_success_response(None, "Verification email sent")

    # ── Profile ─────────────────────────────────────────────────────────

    def get_profile(self, user_id: str) -> Dict[str, Any]:
        profile = self.auth_service.get_profile(user_id)
        if not profile:
            return create_error_response("Profile not found")
        return create_success_response({"profile": profile})

    def update_profile(self, user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        update = ProfileUpdate(
            full_name=data.get("full_name"),
            organization=data.get("organization"),
            purpose=data.get("purpose"),
        )
        fields = update.to_dict()
        if not fields:
            return create_error_response("No fields to update")

        ok, err = self.auth_service.update_profile(user_id, fields)
        if not ok:
            return create_error_response(err or "Update failed")

        self.auth_service.log_activity(
            user_id=user_id,
            activity_type="profile_update",
            description="Profile info updated",
        )
        return create_success_response(None, "Profile updated")

    # ── Activity Logs ───────────────────────────────────────────────────

    def log_activity(self, user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        entry = ActivityLogEntry(
            activity_type=data.get("activity_type", ""),
            description=data.get("description"),
            metadata=data.get("metadata"),
            device_info=data.get("device_info"),
        )
        errors = entry.validate()
        if errors:
            return create_error_response("Validation failed", {"errors": errors})

        self.auth_service.log_activity(
            user_id=user_id,
            activity_type=entry.activity_type,
            description=entry.description,
            metadata=entry.metadata,
            device_info=entry.device_info,
        )
        return create_success_response(None, "Activity logged")

    def get_activity_logs(self, user_id: str, limit: int = 50) -> Dict[str, Any]:
        logs = self.auth_service.get_activity_logs(user_id, limit)
        return create_success_response({"logs": logs, "count": len(logs)})
