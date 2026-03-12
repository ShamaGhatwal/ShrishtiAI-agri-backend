"""
WeatherWise Prediction Routes
RESTful API endpoints for LSTM weather forecasting with disaster context
"""

import logging
from flask import Blueprint, request, jsonify, current_app
from typing import Dict, Any
import traceback

from controllers.weatherwise_prediction_controller import WeatherWisePredictionController

logger = logging.getLogger(__name__)

# Create blueprint for WeatherWise prediction routes
weatherwise_bp = Blueprint('weatherwise', __name__)

def get_controller():
    """Get the WeatherWise controller from app context"""
    return current_app.extensions.get('controllers', {}).get('weatherwise')

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

@weatherwise_bp.route('/forecast', methods=['POST'])
def generate_weather_forecast():
    """
    Generate weather forecast for a specific location using LSTM models
    
    Primary endpoint for weather forecasting with disaster context.
    
    Expected JSON payload:
    {
        \"latitude\": float,         # Required: -90 to 90
        \"longitude\": float,        # Required: -180 to 180
        \"reference_date\": \"YYYY-MM-DD\",  # Optional: date for historical data collection
        \"disaster_type\": \"Normal|Flood|Drought|Storm|Landslide\",  # Optional: disaster context
        \"forecast_days\": int       # Optional: number of days to forecast (1-365, default 60)
    }
    
    Returns:
    {
        \"success\": bool,
        \"message\": str,
        \"data\": {
            \"forecast\": {
                \"temperature_C\": [float],      # Daily temperature forecast
                \"precipitation_mm\": [float],   # Daily precipitation forecast
                \"humidity_%\": [float],         # Daily humidity forecast
                \"wind_speed_mps\": [float],     # Daily wind speed forecast
                \"surface_pressure_hPa\": [float], # Daily pressure forecast
                \"solar_radiation_wm2\": [float]   # Daily solar radiation forecast
            },
            \"forecast_dates\": [str],           # Forecast dates (YYYY-MM-DD)
            \"forecast_variables\": [str],       # Variables included in forecast
            \"model_context\": str,             # Disaster context model used
            \"location\": {
                \"latitude\": float,
                \"longitude\": float
            },
            \"forecast_summary\": {
                \"horizon_days\": int,
                \"variables_count\": int,
                \"model_used\": str
            }
        },
        \"processing_info\": {
            \"processing_time_seconds\": float,
            \"forecast_model\": str,
            \"forecast_horizon_days\": int,
            \"data_sources\": [str]
        }
    }
    
    Error Response:
    {
        \"success\": false,
        \"error\": str,
        \"message\": str
    }
    
    HTTP Status Codes:
    - 200: Forecast generated successfully
    - 400: Invalid request parameters
    - 500: Internal server error
    """
    try:
        logger.info(f"[WEATHERWISE] ========== NEW FORECAST REQUEST ==========")
        logger.info(f"[WEATHERWISE] Forecast request from {request.remote_addr}")
        logger.info(f"[WEATHERWISE] Request method: {request.method}")
        
        # Get controller
        controller = get_controller()
        if not controller:
            logger.error("[ERROR] WeatherWise controller not initialized")
            return {
                'success': False,
                'error': 'WeatherWise service not available',
                'message': 'Controller not initialized'
            }, 503
        
        # Get request data
        request_data = request.get_json()
        logger.info(f"[WEATHERWISE] Raw request data: {request_data}")
        
        if not request_data:
            logger.error("[WEATHERWISE] No JSON data provided in request")
            return {
                'success': False,
                'error': 'No JSON data provided',
                'message': 'Request must contain JSON data'
            }, 400
        
        # Log request parameters (excluding sensitive data)
        safe_params = {
            'latitude': request_data.get('latitude'),
            'longitude': request_data.get('longitude'),
            'disaster_type': request_data.get('disaster_type', 'Not provided'),
            'reference_date': request_data.get('reference_date', 'Not provided')
        }
        logger.info(f"[WEATHERWISE] Request parameters: {safe_params}")
        
        # Process forecast request
        logger.info(f"[WEATHERWISE] Calling controller.forecast_weather()...")
        result = controller.forecast_weather(request_data)
        
        # Log result status
        logger.info(f"[WEATHERWISE] Controller response received")
        if result.get('success'):
            logger.info(f"[WEATHERWISE] Forecast request success=True")
            if result.get('data'):
                logger.info(f"[WEATHERWISE] Model: {result.get('data', {}).get('model_context', 'Unknown')}")
        else:
            logger.warning(f"[WEATHERWISE] Forecast request failed")
            logger.warning(f"[WEATHERWISE] Error: {result.get('error', 'Unknown')}")
        
        logger.info(f"[WEATHERWISE] ========== REQUEST COMPLETE ==========")
        
        # Return response with appropriate status code
        status_code = 200 if result.get('success') else 400
        return jsonify(result), status_code
        
    except Exception as e:
        response_data, status_code = handle_request_error(e, '/api/weatherwise/forecast')
        return jsonify(response_data), status_code

@weatherwise_bp.route('/models', methods=['GET'])
def get_available_models():
    """
    Get available disaster context models for weather forecasting
    
    Returns information about available LSTM models and their capabilities.
    
    Returns:
    {
        \"success\": bool,
        \"message\": str,
        \"data\": {
            \"available_disaster_contexts\": [str],  # Available model contexts
            \"model_info\": {
                \"available_models\": [str],
                \"forecast_variables\": [str],
                \"input_features\": int,
                \"default_horizon_days\": int
            },
            \"default_context\": str,
            \"supported_forecast_variables\": [str]
        }
    }
    
    HTTP Status Codes:
    - 200: Models information retrieved successfully
    - 500: Internal server error
    """
    try:
        logger.info(f"[WEATHERWISE] Models request from {request.remote_addr}")
        
        # Get controller
        controller = get_controller()
        if not controller:
            logger.error("[ERROR] WeatherWise controller not initialized")
            return {
                'success': False,
                'error': 'WeatherWise service not available',
                'message': 'Controller not initialized'
            }, 503
        
        # Get available models
        result = controller.get_available_models()
        
        logger.info(f"[WEATHERWISE] Models request success={result.get('success')}")
        
        # Return response
        status_code = 200 if result.get('success') else 500
        return jsonify(result), status_code
        
    except Exception as e:
        response_data, status_code = handle_request_error(e, '/api/weatherwise/models')
        return jsonify(response_data), status_code

@weatherwise_bp.route('/health', methods=['GET'])
def get_service_health():
    """
    Get WeatherWise service health and status information
    
    Returns detailed information about service status, model availability, 
    and performance statistics.
    
    Returns:
    {
        \"success\": bool,
        \"message\": str,
        \"data\": {
            \"controller_info\": {
                \"controller_name\": str,
                \"controller_stats\": {
                    \"controller_start_time\": str,
                    \"total_requests\": int,
                    \"successful_requests\": int,
                    \"failed_requests\": int
                }
            },
            \"service_health\": {
                \"service_name\": str,
                \"status\": str,
                \"models_loaded\": bool,
                \"available_disaster_contexts\": [str],
                \"statistics\": {...},
                \"supported_forecast_variables\": [str],
                \"default_forecast_horizon_days\": int
            }
        }
    }
    
    HTTP Status Codes:
    - 200: Health information retrieved successfully
    - 503: Service unavailable or unhealthy
    """
    try:
        logger.debug(f"[WEATHERWISE] Health check from {request.remote_addr}")
        
        # Get controller
        controller = get_controller()
        if not controller:
            logger.error("[ERROR] WeatherWise controller not initialized")
            return {
                'success': False,
                'error': 'WeatherWise service not available',
                'message': 'Controller not initialized',
                'status': 'unhealthy'
            }, 503
        
        # Get service health
        result = controller.get_service_status()
        
        # Determine status code based on service health
        is_healthy = (result.get('success') and 
                     result.get('data', {}).get('service_health', {}).get('status') == 'healthy')
        status_code = 200 if is_healthy else 503
        
        logger.debug(f"[WEATHERWISE] Health check success={result.get('success')}, healthy={is_healthy}")
        
        return jsonify(result), status_code
        
    except Exception as e:
        response_data, status_code = handle_request_error(e, '/api/weatherwise/health')
        return jsonify(response_data), status_code

@weatherwise_bp.route('/info', methods=['GET'])
def get_service_info():
    """
    Get general WeatherWise service information
    
    Returns basic information about the WeatherWise service capabilities.
    
    Returns:
    {
        \"success\": true,
        \"message\": str,
        \"data\": {
            \"service_name\": \"WeatherWise\",
            \"description\": str,
            \"version\": str,
            \"capabilities\": [str],
            \"supported_disaster_contexts\": [str],
            \"forecast_variables\": [str],
            \"default_forecast_horizon_days\": int,
            \"input_requirements\": [str]
        }
    }
    """
    try:
        logger.debug(f"[WEATHERWISE] Info request from {request.remote_addr}")
        
        return jsonify({
            'success': True,
            'message': 'WeatherWise service information',
            'data': {
                'service_name': 'WeatherWise',
                'description': 'LSTM-based weather forecasting with disaster context modeling',
                'version': '1.0.0',
                'capabilities': [
                    '60-day weather forecasting',
                    'Disaster-context modeling',
                    'Multi-variable predictions',
                    'Historical data integration'
                ],
                'supported_disaster_contexts': ['Normal', 'Flood', 'Drought', 'Storm', 'Landslide'],
                'forecast_variables': [
                    'temperature_C',
                    'precipitation_mm', 
                    'humidity_%',
                    'wind_speed_mps',
                    'surface_pressure_hPa',
                    'solar_radiation_wm2'
                ],
                'default_forecast_horizon_days': 60,
                'input_requirements': [
                    'latitude (-90 to 90)',
                    'longitude (-180 to 180)',
                    'reference_date (optional)',
                    'disaster_type (optional)',
                    'forecast_days (optional, 1-365)'
                ]
            }
        }), 200
        
    except Exception as e:
        response_data, status_code = handle_request_error(e, '/api/weatherwise/info')
        return jsonify(response_data), status_code

# Register error handlers for the blueprint
@weatherwise_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'message': 'The requested WeatherWise endpoint does not exist'
    }), 404

@weatherwise_bp.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'success': False,
        'error': 'Method not allowed',
        'message': 'The HTTP method is not allowed for this WeatherWise endpoint'
    }), 405

@weatherwise_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'message': 'An internal error occurred in the WeatherWise service'
    }), 500