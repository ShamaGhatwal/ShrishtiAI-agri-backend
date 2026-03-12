"""
GeoVision Fusion Prediction Routes
RESTful API endpoints for the GeoVision multi-model fusion pipeline
"""

import logging
from flask import Blueprint, request, jsonify, current_app
from typing import Dict, Any
import traceback

logger = logging.getLogger(__name__)

# Blueprint
geovision_bp = Blueprint('geovision', __name__)


def get_controller():
    """Get the GeoVision controller from app context."""
    return current_app.extensions.get('controllers', {}).get('geovision')


def handle_request_error(error: Exception, endpoint: str) -> tuple:
    logger.error(f"Error in {endpoint}: {error}")
    logger.error(traceback.format_exc())
    return jsonify({
        'success': False,
        'error': f"Internal server error in {endpoint}",
        'message': 'Request processing failed',
        'details': str(error) if current_app.debug else 'Enable debug mode for details'
    }), 500


# ═══════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@geovision_bp.route('/predict', methods=['POST'])
def predict_fusion():
    """
    Run GeoVision fusion prediction for a location.

    POST /api/geovision/predict
    Body: { "latitude": float, "longitude": float }

    Automatically uses the most recent available weather data.
    Returns comprehensive disaster + weather regime prediction.
    """
    try:
        controller = get_controller()
        if controller is None:
            return jsonify({
                'success': False,
                'error': 'GeoVision service not initialized'
            }), 503

        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json'
            }), 400

        request_data = request.get_json()
        if not request_data:
            return jsonify({
                'success': False,
                'error': 'Empty request body'
            }), 400

        logger.info(f"[GEOVISION_ROUTE] Predict request: lat={request_data.get('latitude')}, "
                     f"lon={request_data.get('longitude')}")

        result = controller.predict_fusion(request_data)

        status_code = 200 if result.get('success') else 400
        return jsonify(result), status_code

    except Exception as e:
        return handle_request_error(e, 'geovision/predict')


@geovision_bp.route('/health', methods=['GET'])
def service_health():
    """
    GET /api/geovision/health
    Returns model load status and service statistics.
    """
    try:
        controller = get_controller()
        if controller is None:
            return jsonify({
                'success': False,
                'error': 'GeoVision service not initialized'
            }), 503

        result = controller.get_service_status()
        return jsonify(result), 200

    except Exception as e:
        return handle_request_error(e, 'geovision/health')


@geovision_bp.route('/models', methods=['GET'])
def list_models():
    """
    GET /api/geovision/models
    Returns details about loaded models.
    """
    try:
        controller = get_controller()
        if controller is None:
            return jsonify({
                'success': False,
                'error': 'GeoVision service not initialized'
            }), 503

        status = controller.get_service_status()
        model_details = status.get('data', {}).get('model_details', {})
        return jsonify({
            'success': True,
            'models': model_details
        }), 200

    except Exception as e:
        return handle_request_error(e, 'geovision/models')
