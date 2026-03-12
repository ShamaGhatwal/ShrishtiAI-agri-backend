"""
HazardGuard Disaster Prediction Routes
RESTful API endpoints for disaster risk prediction at specific locations
"""

import logging
from flask import Blueprint, request, jsonify, current_app
from typing import Dict, Any
import traceback

from controllers.hazardguard_prediction_controller import HazardGuardPredictionController

logger = logging.getLogger(__name__)

# Create blueprint for HazardGuard prediction routes
hazardguard_bp = Blueprint('hazardguard', __name__)

def get_controller():
    """Get the HazardGuard controller from app context"""
    return current_app.extensions.get('controllers', {}).get('hazardguard')

def handle_request_error(error: Exception, endpoint: str) -> tuple[Dict[str, Any], int]:
    """Handle request errors with consistent logging and response format"""
    error_msg = f"Error in {endpoint}: {str(error)}"
    logger.error(error_msg)
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    return {
        'success': False,
        'error': f"Internal server error in {endpoint}",
        'message': 'Request processing failed',
        'details': str(error) if current_app.debug else 'Enable debug mode for details'
    }, 500

@hazardguard_bp.route('/predict', methods=['POST'])
def predict_disaster_risk():
    """
    Predict disaster risk for a specific location
    
    Primary endpoint for map-based location selection and disaster prediction.
    
    Expected JSON payload:
    {
        \"latitude\": float,     # Required: -90 to 90
        \"longitude\": float,    # Required: -180 to 180  
        \"reference_date\": \"YYYY-MM-DD\"  # Optional: date for weather data collection
    }
    
    Returns:
    {
        \"success\": true/false,
        \"message\": \"string\",
        \"data\": {
            \"location\": {
                \"latitude\": float,
                \"longitude\": float,
                \"coordinates_message\": \"string\"
            },
            \"prediction\": {
                \"prediction\": \"DISASTER\" or \"NORMAL\",
                \"probability\": {
                    \"disaster\": float,  # 0.0 to 1.0
                    \"normal\": float     # 0.0 to 1.0
                },
                \"confidence\": float,   # Difference between probabilities
                \"metadata\": {
                    \"features_used\": int,
                    \"features_selected\": int,
                    \"model_type\": \"string\",
                    \"forecast_horizon_days\": int,
                    \"prediction_timestamp\": \"ISO timestamp\"
                }
            },
            \"data_collection_summary\": {
                \"weather_data\": true/false,
                \"feature_engineering\": true/false,
                \"raster_data\": true/false
            },
            \"processing_details\": {
                \"total_processing_time_seconds\": float,
                \"weather_date_range\": \"string\",
                \"forecast_horizon_days\": int,
                \"data_sources\": [\"array\"]
            }
        },
        \"processing_info\": {
            \"total_processing_time_seconds\": float,
            \"prediction_class\": \"DISASTER\" or \"NORMAL\",
            \"disaster_probability\": float,
            \"confidence\": float
        },
        \"timestamp\": \"ISO timestamp\"
    }
    """
    try:
        # Validate request content type
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json',
                'message': 'Invalid request format'
            }), 400
        
        # Get request data
        request_data = request.get_json()
        if not request_data:
            return jsonify({
                'success': False,
                'error': 'Empty request body',
                'message': 'JSON payload required',
                'required_fields': {
                    'latitude': 'float (-90 to 90)',
                    'longitude': 'float (-180 to 180)',
                    'reference_date': 'string (YYYY-MM-DD, optional)'
                }
            }), 400
        
        # Get controller from app context
        logger.info(f"[HAZARDGUARD] Received prediction request: {request_data}")
        controller = get_controller()
        if not controller:
            logger.error("[HAZARDGUARD] Controller not found in app context!")
            return jsonify({
                'success': False,
                'error': 'HazardGuard service not available',
                'message': 'Service not properly initialized'
            }), 503
        
        logger.info(f"[HAZARDGUARD] Controller found, making prediction...")
        # Process using controller
        result = controller.predict_disaster_risk(request_data)
        logger.info(f"[HAZARDGUARD] Prediction result success={result.get('success')}")
        
        # Return response with appropriate status code
        status_code = 200 if result['success'] else 400
        return jsonify(result), status_code
        
    except Exception as e:
        error_response, status_code = handle_request_error(e, 'predict_disaster_risk')
        return jsonify(error_response), status_code

@hazardguard_bp.route('/predict/batch', methods=['POST'])
def predict_batch_locations():
    """
    Predict disaster risk for multiple locations in batch
    
    Expected JSON payload:
    {
        \"locations\": [
            {
                \"latitude\": float,
                \"longitude\": float,
                \"reference_date\": \"YYYY-MM-DD\"  # Optional
            },
            {
                \"latitude\": float,
                \"longitude\": float
            },
            ...  # Maximum 50 locations
        ]
    }
    
    Returns:
    {
        \"success\": true/false,
        \"message\": \"string\",
        \"data\": {
            \"results\": [
                {
                    \"location_index\": int,
                    \"success\": true/false,
                    \"location\": {\"latitude\": float, \"longitude\": float},
                    \"prediction\": {\"prediction\": \"string\", \"probability\": {}, ...},
                    \"processing_time_seconds\": float
                },
                ...
            ],
            \"summary\": {
                \"total_locations\": int,
                \"successful_predictions\": int,
                \"failed_predictions\": int,
                \"success_rate_percent\": float
            }
        },
        \"processing_info\": {
            \"batch_size\": int,
            \"processing_mode\": \"sequential\"
        }
    }
    """
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json',
                'message': 'Invalid request format'
            }), 400
        
        request_data = request.get_json()
        if not request_data:
            return jsonify({
                'success': False,
                'error': 'Empty request body',
                'message': 'JSON payload required'
            }), 400
        
        # Get controller from app context
        controller = get_controller()
        if not controller:
            return jsonify({
                'success': False,
                'error': 'HazardGuard service not available',
                'message': 'Service not properly initialized'
            }), 503
            
        # Process using controller
        result = controller.predict_batch_locations(request_data)
        
        status_code = 200 if result['success'] else 400
        return jsonify(result), status_code
        
    except Exception as e:
        error_response, status_code = handle_request_error(e, 'predict_batch_locations')
        return jsonify(error_response), status_code

@hazardguard_bp.route('/capabilities', methods=['GET'])
def get_prediction_capabilities():
    """
    Get information about HazardGuard prediction capabilities and requirements
    
    Returns:
    {
        \"success\": true,
        \"message\": \"string\",
        \"data\": {
            \"prediction_type\": \"string\",
            \"supported_disaster_types\": [\"array\"],
            \"forecasting_horizon\": \"string\",
            \"geographic_coverage\": \"string\",
            \"data_sources\": {
                \"weather_data\": \"string\",
                \"engineered_features\": \"string\",
                \"raster_data\": \"string\",
                \"total_features\": \"string\"
            },
            \"model_details\": {
                \"algorithm\": \"string\",
                \"feature_selection\": \"string\",
                \"preprocessing\": \"string\",
                \"validation\": \"string\"
            },
            \"input_requirements\": {
                \"required_fields\": [\"array\"],
                \"optional_fields\": [\"array\"],
                \"coordinate_ranges\": {\"object\"}
            },
            \"output_format\": {\"object\"},
            \"batch_processing\": {\"object\"},
            \"service_status\": {\"object\"}
        }
    }
    """
    try:
        # Get controller from app context
        controller = get_controller()
        if not controller:
            return jsonify({
                'success': False,
                'error': 'HazardGuard service not available',
                'message': 'Service not properly initialized'
            }), 503
            
        # Get capabilities using controller
        result = controller.get_prediction_capabilities()
        
        status_code = 200 if result['success'] else 500
        return jsonify(result), status_code
        
    except Exception as e:
        error_response, status_code = handle_request_error(e, 'get_prediction_capabilities')
        return jsonify(error_response), status_code

@hazardguard_bp.route('/validate/coordinates', methods=['POST'])
def validate_coordinates():
    """
    Validate coordinates without making prediction (for testing/validation)
    
    Expected JSON payload:
    {
        \"latitude\": float,
        \"longitude\": float,
        \"reference_date\": \"YYYY-MM-DD\"  # Optional
    }
    
    Returns:
    {
        \"success\": true/false,
        \"message\": \"string\",
        \"data\": {
            \"coordinates\": {
                \"latitude\": float,
                \"longitude\": float,
                \"reference_date\": \"string\" or null
            },
            \"validation_message\": \"string\"
        }
    }
    """
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json',
                'message': 'Invalid request format'
            }), 400
        
        request_data = request.get_json()
        if not request_data:
            return jsonify({
                'success': False,
                'error': 'Empty request body',
                'message': 'JSON payload required'
            }), 400
        
        # Get controller from app context
        controller = get_controller()
        if not controller:
            return jsonify({
                'success': False,
                'error': 'HazardGuard service not available',
                'message': 'Service not properly initialized'
            }), 503
            
        # Validate using controller
        result = controller.validate_coordinates_only(request_data)
        
        status_code = 200 if result['success'] else 400
        return jsonify(result), status_code
        
    except Exception as e:
        error_response, status_code = handle_request_error(e, 'validate_coordinates')
        return jsonify(error_response), status_code

@hazardguard_bp.route('/health', methods=['GET'])
def get_service_health():
    """
    Get HazardGuard service health and performance statistics
    
    Returns:
    {
        \"success\": true/false,
        \"message\": \"string\",
        \"data\": {
            \"service_status\": \"ready\" or \"not_initialized\" or \"error\",
            \"uptime_seconds\": float,
            \"uptime_hours\": float,
            \"model_loaded\": true/false,
            \"model_info\": {
                \"is_loaded\": true/false,
                \"model_metadata\": {\"object\"},
                \"feature_counts\": {\"object\"},
                \"forecasting\": {\"object\"},
                \"prediction_statistics\": {\"object\"}
            },
            \"statistics\": {
                \"total_requests\": int,
                \"successful_predictions\": int,
                \"failed_predictions\": int,
                \"success_rate_percent\": float,
                \"data_collection_failures\": int,
                \"weather_fetch_failures\": int,
                \"feature_engineering_failures\": int,
                \"raster_fetch_failures\": int,
                \"average_processing_time_seconds\": float
            },
            \"service_dependencies\": {\"object\"},
            \"last_updated\": \"ISO timestamp\"
        }
    }
    """
    try:
        # Get controller from app context
        controller = get_controller()
        if not controller:
            return jsonify({
                'success': False,
                'error': 'HazardGuard service not available',
                'message': 'Service not properly initialized'
            }), 503
            
        # Get health info using controller  
        result = controller.get_service_health()
        
        status_code = 200 if result['success'] else 500
        return jsonify(result), status_code
        
    except Exception as e:
        error_response, status_code = handle_request_error(e, 'get_service_health')
        return jsonify(error_response), status_code

@hazardguard_bp.route('/initialize', methods=['POST'])
def initialize_service():
    """
    Initialize HazardGuard service (load model components)
    
    Returns:
    {
        \"success\": true/false,
        \"message\": \"string\",
        \"data\": {
            \"service_status\": \"ready\" or error info,
            \"initialization_message\": \"string\"
        }
    }
    """
    try:
        # Get controller from app context
        controller = get_controller()
        if not controller:
            return jsonify({
                'success': False,
                'error': 'HazardGuard service not available',
                'message': 'Service not properly initialized'
            }), 503
            
        # Initialize using controller
        result = controller.initialize_controller()
        
        status_code = 200 if result['success'] else 500
        return jsonify(result), status_code
        
    except Exception as e:
        error_response, status_code = handle_request_error(e, 'initialize_service')
        return jsonify(error_response), status_code

@hazardguard_bp.route('/statistics/reset', methods=['POST'])
def reset_statistics():
    """
    Reset HazardGuard service and model statistics
    
    Returns:
    {
        \"success\": true/false,
        \"message\": \"string\",
        \"data\": {
            \"status\": \"success\" or \"error\",
            \"message\": \"string\",
            \"timestamp\": \"ISO timestamp\"
        }
    }
    """
    try:
        # Get controller from app context
        controller = get_controller()
        if not controller:
            return jsonify({
                'success': False,
                'error': 'HazardGuard service not available',
                'message': 'Service not properly initialized'
            }), 503
            
        # Reset statistics using controller
        result = controller.reset_service_statistics()
        
        status_code = 200 if result['success'] else 500
        return jsonify(result), status_code
        
    except Exception as e:
        error_response, status_code = handle_request_error(e, 'reset_statistics')
        return jsonify(error_response), status_code

@hazardguard_bp.route('/ping', methods=['GET'])
def ping():
    """
    Simple ping endpoint to check if HazardGuard service is responsive
    
    Returns:
    {
        \"success\": true,
        \"message\": \"Service is responsive\",
        \"data\": {
            \"service\": \"hazardguard_disaster_prediction\",
            \"status\": \"active\",
            \"timestamp\": \"ISO timestamp\"
        }
    }
    """
    try:
        from datetime import datetime
        
        return jsonify({
            'success': True,
            'message': 'HazardGuard service is responsive',
            'data': {
                'service': 'hazardguard_disaster_prediction',
                'status': 'active',
                'prediction_types': ['DISASTER', 'NORMAL'],
                'supported_disasters': ['Flood', 'Storm', 'Landslide', 'Drought'],
                'timestamp': datetime.now().isoformat()
            }
        }), 200
        
    except Exception as e:
        error_response, status_code = handle_request_error(e, 'ping')
        return jsonify(error_response), status_code

# Error handlers for the blueprint
@hazardguard_bp.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'message': 'The requested HazardGuard endpoint does not exist',
        'available_endpoints': [
            '/predict - POST: Predict disaster risk for location',
            '/predict/batch - POST: Predict for multiple locations',
            '/capabilities - GET: Get prediction capabilities',
            '/validate/coordinates - POST: Validate coordinates',
            '/health - GET: Get service health',
            '/initialize - POST: Initialize service',
            '/statistics/reset - POST: Reset statistics',
            '/ping - GET: Service ping test'
        ]
    }), 404

@hazardguard_bp.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({
        'success': False,
        'error': 'Method not allowed',
        'message': 'The HTTP method is not allowed for this HazardGuard endpoint',
        'allowed_methods': ['GET', 'POST']
    }), 405

@hazardguard_bp.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error in HazardGuard: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': 'An unexpected error occurred in HazardGuard service'
    }), 500

# Blueprint registration information
def get_blueprint_info() -> Dict[str, Any]:
    """Get information about this blueprint"""
    return {
        'name': 'hazardguard',
        'description': 'HazardGuard disaster prediction API endpoints',
        'version': '1.0.0',
        'prediction_type': 'Binary Classification (DISASTER vs NORMAL)',
        'supported_disasters': ['Flood', 'Storm', 'Landslide', 'Drought'],
        'endpoints': {
            '/predict': {
                'methods': ['POST'],
                'description': 'Predict disaster risk for specific location'
            },
            '/predict/batch': {
                'methods': ['POST'],
                'description': 'Predict disaster risk for multiple locations'
            },
            '/capabilities': {
                'methods': ['GET'],
                'description': 'Get prediction capabilities and requirements'
            },
            '/validate/coordinates': {
                'methods': ['POST'],
                'description': 'Validate coordinates without prediction'
            },
            '/health': {
                'methods': ['GET'],
                'description': 'Get service health and performance statistics'
            },
            '/initialize': {
                'methods': ['POST'],
                'description': 'Initialize service and load model'
            },
            '/statistics/reset': {
                'methods': ['POST'],
                'description': 'Reset service and model statistics'
            },
            '/ping': {
                'methods': ['GET'],
                'description': 'Simple service ping test'
            }
        },
        'features': {
            'map_based_prediction': True,
            'batch_processing': True,
            'coordinate_validation': True,
            'health_monitoring': True,
            'statistical_tracking': True,
            'forecasting_horizon': '1 day ahead',
            'max_batch_size': 50
        },
        'data_requirements': {
            'weather_data': '60-day sequences, 17 variables',
            'engineered_features': '60-day sequences, 19 variables', 
            'raster_data': '9 geographic variables',
            'total_features': '~300 after statistical expansion'
        }
    }