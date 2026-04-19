"""
Urban Planning Routes
Minimal API surface used by frontend Urban Planning tools.
"""

import logging
from typing import Any, Dict, Tuple

from flask import Blueprint, current_app, jsonify, request

logger = logging.getLogger(__name__)

urban_planning_bp = Blueprint("urban_planning", __name__)


def _service():
    return current_app.extensions.get("services", {}).get("urban_planning")


def _json_error(message: str, code: int = 400):
    return jsonify({"success": False, "error": message}), code


def _validate_coordinates(payload: Dict[str, Any], min_points: int) -> Tuple[list | None, tuple | None]:
    coordinates = payload.get("coordinates")
    if not isinstance(coordinates, list) or len(coordinates) < min_points:
        return None, _json_error(
            f"At least {min_points} coordinate pairs are required",
            400,
        )
    return coordinates, None


def _run_service(result: Dict[str, Any]):
    status = result.get("status")
    if status == "success":
        return jsonify({"success": True, "data": result, "credit_info": {"charged_credits": 0}}), 200
    if status == "no_data":
        return _json_error(result.get("message", "No data available"), 404)
    return _json_error(result.get("message", "Request failed"), 400)


@urban_planning_bp.route("/plot-area", methods=["POST"])
def calculate_plot_area():
    svc = _service()
    if not svc:
        return _json_error("Urban Planning service unavailable", 503)

    payload = request.get_json(silent=True) or {}
    coordinates, err = _validate_coordinates(payload, 3)
    if err:
        return err

    try:
        result = svc.calculate_plot_area(coordinates)
        return _run_service(result)
    except Exception as e:
        logger.error("urban plot-area failed: %s", e)
        return _json_error("Internal error", 500)


@urban_planning_bp.route("/road-length", methods=["POST"])
def calculate_road_length():
    svc = _service()
    if not svc:
        return _json_error("Urban Planning service unavailable", 503)

    payload = request.get_json(silent=True) or {}
    coordinates, err = _validate_coordinates(payload, 2)
    if err:
        return err

    try:
        result = svc.calculate_road_length(coordinates)
        return _run_service(result)
    except Exception as e:
        logger.error("urban road-length failed: %s", e)
        return _json_error("Internal error", 500)


@urban_planning_bp.route("/built-up", methods=["POST"])
def detect_built_up():
    svc = _service()
    if not svc:
        return _json_error("Urban Planning service unavailable", 503)

    payload = request.get_json(silent=True) or {}
    coordinates, err = _validate_coordinates(payload, 3)
    if err:
        return err

    date = payload.get("date")

    try:
        result = svc.detect_built_up(coordinates, date)
        return _run_service(result)
    except Exception as e:
        logger.error("urban built-up failed: %s", e)
        return _json_error("Internal error", 500)


@urban_planning_bp.route("/suitability", methods=["POST"])
def analyze_suitability():
    svc = _service()
    if not svc:
        return _json_error("Urban Planning service unavailable", 503)

    payload = request.get_json(silent=True) or {}
    coordinates, err = _validate_coordinates(payload, 3)
    if err:
        return err

    date = payload.get("date")

    try:
        result = svc.analyze_suitability(coordinates, date)
        return _run_service(result)
    except Exception as e:
        logger.error("urban suitability failed: %s", e)
        return _json_error("Internal error", 500)


@urban_planning_bp.route("/capabilities", methods=["GET"])
def get_capabilities():
    return jsonify({
        "success": True,
        "data": {
            "service": "urban_planning",
            "features": [
                {
                    "id": "built_up",
                    "name": "NDBI Analysis",
                    "description": "Global heatmap showing built-up areas",
                    "credit_cost": 5,
                    "type": "global",
                },
                {
                    "id": "plot_area",
                    "name": "Plot Measurement",
                    "description": "Calculate area of a drawn polygon",
                    "credit_cost": 2,
                    "type": "polygon",
                },
                {
                    "id": "road_length",
                    "name": "Road/Line Length",
                    "description": "Measure length of roads or paths",
                    "credit_cost": 2,
                    "type": "polyline",
                },
                {
                    "id": "suitability",
                    "name": "Suitability Analysis",
                    "description": "Score areas for building suitability",
                    "credit_cost": 8,
                    "type": "polygon",
                },
            ],
        },
    }), 200


@urban_planning_bp.route("/ping", methods=["GET"])
def ping():
    svc = _service()
    healthy = bool(svc)
    return jsonify({"success": healthy, "service": "urban_planning", "status": "healthy" if healthy else "unavailable"}), (200 if healthy else 503)
