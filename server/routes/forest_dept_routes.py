"""
Forest Department Routes
Minimal API surface used by frontend Forest Department tools.
"""

import logging
from typing import Any, Dict, Tuple

from flask import Blueprint, current_app, jsonify, request

logger = logging.getLogger(__name__)

forest_dept_bp = Blueprint("forest_dept", __name__)


def _service():
    return current_app.extensions.get("services", {}).get("forest_dept")


def _json_error(message: str, code: int = 400):
    return jsonify({"success": False, "error": message}), code


def _validate_coordinates(payload: Dict[str, Any], key: str = "coordinates", min_points: int = 3) -> Tuple[list | None, tuple | None]:
    coordinates = payload.get(key)
    if not isinstance(coordinates, list) or len(coordinates) < min_points:
        return None, _json_error(
            f"At least {min_points} coordinate pairs are required for '{key}'",
            400,
        )
    return coordinates, None


def _run_service(result: Dict[str, Any]):
    status = result.get("status")
    if status == "success":
        return jsonify({"success": True, "data": result, "remaining_credits": None}), 200
    if status == "no_data":
        return _json_error(result.get("message", "No data available"), 404)
    return _json_error(result.get("message", "Request failed"), 400)


@forest_dept_bp.route("/ndvi", methods=["GET"])
def get_ndvi_tiles():
    svc = _service()
    if not svc:
        return _json_error("Forest Department service unavailable", 503)

    try:
        result = svc.get_ndvi_tiles()
        if result.get("status") == "success":
            return jsonify({
                "success": True,
                "tile_url": result.get("tile_url"),
                "metadata": result.get("metadata"),
                "legend": result.get("legend"),
            }), 200
        return _json_error(result.get("message", "Failed to get NDVI tiles"), 500)
    except Exception as e:
        logger.error("forest ndvi failed: %s", e)
        return _json_error("Internal error", 500)


@forest_dept_bp.route("/soil-moisture-tiles", methods=["GET"])
def get_soil_moisture_tiles():
    svc = _service()
    if not svc:
        return _json_error("Forest Department service unavailable", 503)

    try:
        result = svc.get_soil_moisture_tiles()
        if result.get("status") == "success":
            return jsonify({
                "success": True,
                "tile_url": result.get("tile_url"),
                "metadata": result.get("metadata"),
                "legend": result.get("legend"),
            }), 200
        return _json_error(result.get("message", "Failed to get soil moisture tiles"), 500)
    except Exception as e:
        logger.error("forest soil-moisture-tiles failed: %s", e)
        return _json_error("Internal error", 500)


@forest_dept_bp.route("/active-fires-tiles", methods=["GET"])
def get_active_fires_tiles():
    svc = _service()
    if not svc:
        return _json_error("Forest Department service unavailable", 503)

    try:
        result = svc.get_active_fires_tiles()
        if result.get("status") == "success":
            return jsonify({
                "success": True,
                "tile_url": result.get("tile_url"),
                "metadata": result.get("metadata"),
                "legend": result.get("legend"),
            }), 200
        return _json_error(result.get("message", "Failed to get active fires tiles"), 500)
    except Exception as e:
        logger.error("forest active-fires-tiles failed: %s", e)
        return _json_error("Internal error", 500)


@forest_dept_bp.route("/crop-classification", methods=["POST"])
def classify_crops():
    svc = _service()
    if not svc:
        return _json_error("Forest Department service unavailable", 503)

    payload = request.get_json(silent=True) or {}
    coordinates, err = _validate_coordinates(payload)
    if err:
        return err

    season = payload.get("season", "kharif")

    try:
        result = svc.classify_crops(coordinates, season)
        return _run_service(result)
    except Exception as e:
        logger.error("forest crop-classification failed: %s", e)
        return _json_error("Internal error", 500)


@forest_dept_bp.route("/soil-moisture", methods=["POST"])
def analyze_soil_moisture():
    svc = _service()
    if not svc:
        return _json_error("Forest Department service unavailable", 503)

    payload = request.get_json(silent=True) or {}
    coordinates, err = _validate_coordinates(payload)
    if err:
        return err

    try:
        result = svc.analyze_soil_moisture(coordinates)
        return _run_service(result)
    except Exception as e:
        logger.error("forest soil-moisture failed: %s", e)
        return _json_error("Internal error", 500)


@forest_dept_bp.route("/fire-risk", methods=["POST"])
def analyze_fire_risk():
    svc = _service()
    if not svc:
        return _json_error("Forest Department service unavailable", 503)

    payload = request.get_json(silent=True) or {}
    coordinates, err = _validate_coordinates(payload)
    if err:
        return err

    try:
        result = svc.analyze_fire_risk(coordinates)
        return _run_service(result)
    except Exception as e:
        logger.error("forest fire-risk failed: %s", e)
        return _json_error("Internal error", 500)


@forest_dept_bp.route("/plantation-suitability", methods=["POST"])
def analyze_plantation_suitability():
    svc = _service()
    if not svc:
        return _json_error("Forest Department service unavailable", 503)

    payload = request.get_json(silent=True) or {}
    coordinates, err = _validate_coordinates(payload)
    if err:
        return err

    try:
        result = svc.analyze_plantation_suitability(coordinates)
        return _run_service(result)
    except Exception as e:
        logger.error("forest plantation-suitability failed: %s", e)
        return _json_error("Internal error", 500)


@forest_dept_bp.route("/compensatory-plantation", methods=["POST"])
def plan_compensatory_plantation():
    svc = _service()
    if not svc:
        return _json_error("Forest Department service unavailable", 503)

    payload = request.get_json(silent=True) or {}
    removal_coordinates, err = _validate_coordinates(payload, key="removal_coordinates")
    if err:
        return err

    search_coordinates = payload.get("search_coordinates")
    if search_coordinates is not None and (not isinstance(search_coordinates, list) or len(search_coordinates) < 3):
        return _json_error("search_coordinates must have at least 3 coordinate pairs", 400)

    try:
        result = svc.plan_compensatory_plantation(removal_coordinates, search_coordinates)
        return _run_service(result)
    except Exception as e:
        logger.error("forest compensatory-plantation failed: %s", e)
        return _json_error("Internal error", 500)


@forest_dept_bp.route("/tree-growth", methods=["POST"])
def estimate_tree_growth():
    svc = _service()
    if not svc:
        return _json_error("Forest Department service unavailable", 503)

    payload = request.get_json(silent=True) or {}
    coordinates, err = _validate_coordinates(payload)
    if err:
        return err

    try:
        result = svc.estimate_tree_growth(coordinates)
        return _run_service(result)
    except Exception as e:
        logger.error("forest tree-growth failed: %s", e)
        return _json_error("Internal error", 500)


@forest_dept_bp.route("/species-recommendation", methods=["POST"])
def recommend_species():
    svc = _service()
    if not svc:
        return _json_error("Forest Department service unavailable", 503)

    payload = request.get_json(silent=True) or {}
    coordinates, err = _validate_coordinates(payload)
    if err:
        return err

    try:
        result = svc.recommend_species(coordinates)
        return _run_service(result)
    except Exception as e:
        logger.error("forest species-recommendation failed: %s", e)
        return _json_error("Internal error", 500)


@forest_dept_bp.route("/health", methods=["GET"])
def health():
    svc = _service()
    healthy = bool(svc)
    return jsonify({"success": healthy, "service": "forest_dept", "status": "healthy" if healthy else "unavailable"}), (200 if healthy else 503)


@forest_dept_bp.route("/capabilities", methods=["GET"])
def capabilities():
    return jsonify({
        "success": True,
        "features": [
            {"id": "ndvi", "name": "NDVI Analysis", "description": "Global vegetation health heatmap", "credit_cost": 0, "type": "global"},
            {"id": "fire_risk", "name": "Active Fires (FIRMS)", "description": "Real-time fire hotspots & thermal anomalies", "credit_cost": 0, "type": "global"},
            {"id": "soil_moisture", "name": "Soil Moisture", "description": "Global moisture heatmap (NDMI)", "credit_cost": 0, "type": "global"},
            {"id": "crop_classification", "name": "Crop Classification", "description": "Classify crops (rice, wheat, etc.)", "credit_cost": 8, "type": "polygon"},
            {"id": "plantation_suitability", "name": "Plantation Suitability", "description": "Best areas for tree plantation", "credit_cost": 8, "type": "polygon"},
            {"id": "compensatory_plantation", "name": "Compensatory Plantation", "description": "Plan replacement planting areas", "credit_cost": 10, "type": "polygon"},
            {"id": "tree_growth", "name": "Vegetation Trend", "description": "10-year NDVI analysis with charts", "credit_cost": 5, "type": "polygon"},
            {"id": "species_recommendation", "name": "Species Recommendation", "description": "Suggest suitable plant species", "credit_cost": 5, "type": "polygon"},
        ],
    }), 200
