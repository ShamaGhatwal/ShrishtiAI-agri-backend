"""
Post-Disaster Feature Engineering Routes for HazardGuard System
RESTful API endpoints for post-disaster feature engineering operations
"""

import logging
from flask import Blueprint, request, jsonify, current_app
from typing import Dict, Any
import traceback

from controllers.post_disaster_feature_engineering_controller import PostDisasterFeatureEngineeringController

logger = logging.getLogger(__name__)

# Create blueprint for post-disaster feature engineering routes
post_disaster_feature_engineering_bp = Blueprint('post_disaster_feature_engineering', __name__)

# Initialize controller
controller = PostDisasterFeatureEngineeringController()

def handle_request_error(error: Exception, endpoint: str) -> Dict[str, Any]:
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

@post_disaster_feature_engineering_bp.route('/process', methods=['POST'])
def process_single_coordinate_features():
    """
    Process post-disaster feature engineering for a single coordinate
    
    Expected JSON payload:
    {
        \"weather_data\": {
            \"POST_temperature_C\": [list of 60 daily values],
            \"POST_humidity_%\": [list of 60 daily values],
            ... (17 weather variables total)
        },
        \"coordinate\": [latitude, longitude] (optional),
        \"global_stats\": {} (optional)
    }
    
    Returns:
    {
        \"success\": true/false,
        \"message\": \"string\",
        \"data\": {
            \"coordinate\": [lat, lon],
            \"features\": {
                \"POST_temp_normalized\": [60 values],
                \"POST_temp_range\": [60 values],
                ... (19 features total)
            },
            \"metadata\": {}
        },
        \"processing_info\": {
            \"processing_time_seconds\": float,
            \"features_count\": int,
            \"days_processed\": int
        }
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
                'message': 'JSON payload required'
            }), 400
        
        # Process using controller
        result = controller.process_single_coordinate_features(request_data)
        
        # Return response with appropriate status code
        status_code = 200 if result['success'] else 400
        return jsonify(result), status_code
        
    except Exception as e:
        error_response, status_code = handle_request_error(e, 'process_single_coordinate_features')
        return jsonify(error_response), status_code

@post_disaster_feature_engineering_bp.route('/batch', methods=['POST'])
def process_batch_features():
    """
    Process post-disaster feature engineering for multiple coordinates
    
    Expected JSON payload:
    {
        \"weather_datasets\": [
            {weather_data_1},
            {weather_data_2},
            ...
        ],
        \"coordinates\": [[lat1, lon1], [lat2, lon2], ...] (optional)
    }
    
    Returns:
    {
        \"success\": true/false,
        \"message\": \"string\",
        \"data\": {
            \"results\": [
                {
                    \"coordinate_index\": int,
                    \"coordinate\": [lat, lon],
                    \"success\": true/false,
                    \"features\": {},
                    \"metadata\": {}
                },
                ...
            ],
            \"global_statistics\": {},
            \"summary\": {
                \"total_coordinates\": int,
                \"successful_coordinates\": int,
                \"failed_coordinates\": int,
                \"success_rate_percent\": float
            }
        },
        \"processing_info\": {}
    }
    """
    try:
        # Validate request
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
        
        # Process using controller
        result = controller.process_batch_features(request_data)
        
        # Return response
        status_code = 200 if result['success'] else 400
        return jsonify(result), status_code
        
    except Exception as e:
        error_response, status_code = handle_request_error(e, 'process_batch_features')
        return jsonify(error_response), status_code

@post_disaster_feature_engineering_bp.route('/export/csv', methods=['POST'])
def export_to_csv():
    """
    Export feature engineering results to CSV format
    
    Expected JSON payload:
    {
        \"results\": [list of feature engineering results],
        \"include_metadata\": true/false (optional, default: true)
    }
    
    Returns:
    {
        \"success\": true/false,
        \"message\": \"string\",
        \"data\": {
            \"csv_data\": \"string (CSV content)\",
            \"row_count\": int,
            \"column_count\": int,
            \"columns\": [list of column names]
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
        
        # Process using controller
        result = controller.export_to_csv(request_data)
        
        status_code = 200 if result['success'] else 400
        return jsonify(result), status_code
        
    except Exception as e:
        error_response, status_code = handle_request_error(e, 'export_to_csv')
        return jsonify(error_response), status_code

@post_disaster_feature_engineering_bp.route('/validate/coordinates', methods=['POST'])
def validate_coordinates():
    """
    Validate coordinate input format
    
    Expected JSON payload:
    {
        \"coordinates\": [[lat1, lon1], [lat2, lon2], ...]
    }
    
    Returns:
    {
        \"success\": true/false,
        \"message\": \"string\",
        \"data\": {
            \"coordinates\": [validated coordinates],
            \"count\": int,
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
        
        # Validate using controller
        result = controller.validate_coordinates(request_data)
        
        status_code = 200 if result['success'] else 400
        return jsonify(result), status_code
        
    except Exception as e:
        error_response, status_code = handle_request_error(e, 'validate_coordinates')
        return jsonify(error_response), status_code

@post_disaster_feature_engineering_bp.route('/validate/weather', methods=['POST'])
def validate_weather_data():
    """
    Validate weather data input format
    
    Expected JSON payload:
    {
        \"weather_data\": {
            \"POST_temperature_C\": [60 values],
            \"POST_humidity_%\": [60 values],
            ... (17 variables total)
        }
    }
    
    Returns:
    {
        \"success\": true/false,
        \"message\": \"string\",
        \"data\": {
            \"validation_message\": \"string\",
            \"variables_count\": int,
            \"days_per_variable\": int,
            \"detected_variables\": [list]
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
        
        # Validate using controller
        result = controller.validate_weather_input(request_data)
        
        status_code = 200 if result['success'] else 400
        return jsonify(result), status_code
        
    except Exception as e:
        error_response, status_code = handle_request_error(e, 'validate_weather_data')
        return jsonify(error_response), status_code

@post_disaster_feature_engineering_bp.route('/features/info', methods=['GET'])
def get_feature_info():
    """
    Get information about input variables and output features
    
    Returns:
    {
        \"success\": true,
        \"message\": \"string\",
        \"data\": {
            \"input_variables\": {
                \"count\": int,
                \"variables\": [list of variable names],
                \"description\": \"string\"
            },
            \"output_features\": {
                \"count\": int,
                \"features\": [list of feature names],
                \"descriptions\": {feature_name: {description, unit, calculation}},
                \"description\": \"string\"
            },
            \"processing_info\": {
                \"days_per_coordinate\": int,
                \"feature_engineering_type\": \"string\"
            }
        }
    }
    """
    try:
        # Get feature info using controller
        result = controller.get_feature_info()
        
        status_code = 200 if result['success'] else 500
        return jsonify(result), status_code
        
    except Exception as e:
        error_response, status_code = handle_request_error(e, 'get_feature_info')
        return jsonify(error_response), status_code

@post_disaster_feature_engineering_bp.route('/health', methods=['GET'])
def get_service_health():
    """
    Get service health and performance statistics
    
    Returns:
    {
        \"success\": true/false,
        \"message\": \"string\",
        \"data\": {
            \"service_status\": \"healthy/error\",
            \"service_uptime_seconds\": float,
            \"service_uptime_hours\": float,
            \"total_requests\": int,
            \"successful_requests\": int,
            \"failed_requests\": int,
            \"success_rate_percent\": float,
            \"total_coordinates_processed\": int,
            \"average_processing_time_seconds\": float,
            \"model_statistics\": {},
            \"feature_counts\": {},
            \"last_updated\": \"ISO timestamp\"
        }
    }
    """
    try:
        # Get health info using controller
        result = controller.get_service_health()
        
        status_code = 200 if result['success'] else 500
        return jsonify(result), status_code
        
    except Exception as e:
        error_response, status_code = handle_request_error(e, 'get_service_health')
        return jsonify(error_response), status_code

@post_disaster_feature_engineering_bp.route('/statistics/reset', methods=['POST'])
def reset_statistics():
    """
    Reset service and model statistics
    
    Returns:
    {
        \"success\": true/false,
        \"message\": \"string\",
        \"data\": {
            \"status\": \"success/error\",
            \"message\": \"string\",
            \"timestamp\": \"ISO timestamp\"
        }
    }
    """
    try:
        # Reset statistics using controller
        result = controller.reset_statistics()
        
        status_code = 200 if result['success'] else 500
        return jsonify(result), status_code
        
    except Exception as e:
        error_response, status_code = handle_request_error(e, 'reset_statistics')
        return jsonify(error_response), status_code

@post_disaster_feature_engineering_bp.route('/ping', methods=['GET'])
def ping():
    """
    Simple ping endpoint to check if service is responsive
    
    Returns:
    {
        \"success\": true,
        \"message\": \"Service is responsive\",
        \"data\": {
            \"service\": \"post_disaster_feature_engineering\",
            \"status\": \"active\",
            \"timestamp\": \"ISO timestamp\"
        }
    }
    """
    try:
        from datetime import datetime
        
        return jsonify({
            'success': True,
            'message': 'Service is responsive',
            'data': {
                'service': 'post_disaster_feature_engineering',
                'status': 'active',
                'timestamp': datetime.now().isoformat()
            }
        }), 200
        
    except Exception as e:
        error_response, status_code = handle_request_error(e, 'ping')
        return jsonify(error_response), status_code

# Error handlers for the blueprint
@post_disaster_feature_engineering_bp.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist',
        'available_endpoints': [
            '/process - POST: Process single coordinate features',
            '/batch - POST: Process multiple coordinates',
            '/export/csv - POST: Export results to CSV',
            '/validate/coordinates - POST: Validate coordinates',
            '/validate/weather - POST: Validate weather data',
            '/features/info - GET: Get feature information',
            '/health - GET: Get service health',
            '/statistics/reset - POST: Reset statistics',
            '/ping - GET: Service ping test'
        ]
    }), 404

@post_disaster_feature_engineering_bp.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({
        'success': False,
        'error': 'Method not allowed',
        'message': 'The HTTP method is not allowed for this endpoint',
        'allowed_methods': ['GET', 'POST']
    }), 405

@post_disaster_feature_engineering_bp.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': 'An unexpected error occurred while processing the request'
    }), 500

# Blueprint registration information
def get_blueprint_info() -> Dict[str, Any]:
    """Get information about this blueprint"""
    return {
        'name': 'post_disaster_feature_engineering',
        'description': 'Post-disaster feature engineering API endpoints',
        'version': '1.0.0',
        'endpoints': {
            '/process': {
                'methods': ['POST'],
                'description': 'Process single coordinate feature engineering'
            },
            '/batch': {
                'methods': ['POST'], 
                'description': 'Process multiple coordinates batch feature engineering'
            },
            '/export/csv': {
                'methods': ['POST'],
                'description': 'Export results to CSV format'
            },
            '/validate/coordinates': {
                'methods': ['POST'],
                'description': 'Validate coordinate input format'
            },
            '/validate/weather': {
                'methods': ['POST'],
                'description': 'Validate weather data input format'
            },
            '/features/info': {
                'methods': ['GET'],
                'description': 'Get input variables and output features information'
            },
            '/health': {
                'methods': ['GET'],
                'description': 'Get service health and performance statistics'
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
            'input_variables': 17,
            'output_features': 19,
            'days_per_coordinate': 60,
            'supports_batch_processing': True,
            'supports_csv_export': True,
            'supports_validation': True,
            'supports_health_monitoring': True
        }
    }