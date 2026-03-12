"""
Weather Data API Routes
RESTful endpoints for NASA POWER weather data operations
"""
from flask import Blueprint, request, jsonify, g
import logging
from controllers.weather_controller import WeatherController
from services.weather_service import NASAPowerService

# Initialize blueprint
weather_bp = Blueprint('weather', __name__)

# Configure logging
logger = logging.getLogger(__name__)

# Service and controller will be initialized by main app
weather_service = None
weather_controller = None

def init_weather_routes(controller_instance: WeatherController):
    """Initialize weather routes with controller instance"""
    global weather_controller
    weather_controller = controller_instance
    logger.info("Weather routes initialized with controller")

@weather_bp.route('/weather/data', methods=['GET', 'POST'])
def get_weather_data():
    """
    Get weather data for specific coordinates and disaster date
    
    GET parameters:
        - lat: Latitude (required)
        - lon: Longitude (required)  
        - date: Disaster date in YYYY-MM-DD format (required)
        - days_before: Number of days before disaster to fetch (optional, default: 60)
    
    POST body:
        {
            "latitude": float,
            "longitude": float,
            "disaster_date": "YYYY-MM-DD",
            "days_before": int (optional)
        }
    """
    try:
        if weather_controller is None:
            return jsonify({
                'status': 'error',
                'message': 'Weather service not initialized',
                'data': None
            }), 503
        
        if request.method == 'GET':
            # Handle GET request with query parameters
            data = {
                'latitude': request.args.get('lat'),
                'longitude': request.args.get('lon'),
                'disaster_date': request.args.get('date'),
                'days_before': request.args.get('days_before', 60)
            }
        else:
            # Handle POST request with JSON body
            data = request.get_json()
            
            if not data:
                return jsonify({
                    'status': 'error',
                    'message': 'No JSON data provided',
                    'data': None
                }), 400
        
        # Get weather data
        result = weather_controller.get_weather_data(data)
        
        # Return response with appropriate status code
        status_code = 200 if result.get('status') == 'success' else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Weather data API error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Weather API error: {str(e)}',
            'data': None
        }), 500

@weather_bp.route('/weather/time-series', methods=['GET', 'POST'])
def get_weather_time_series():
    """
    Get weather data as time series
    
    Parameters: Same as get_weather_data
    
    Returns time series DataFrame data with dates and weather values
    """
    try:
        if weather_controller is None:
            return jsonify({
                'status': 'error',
                'message': 'Weather service not initialized',
                'data': None
            }), 503
        if request.method == 'GET':
            data = {
                'latitude': request.args.get('lat'),
                'longitude': request.args.get('lon'),
                'disaster_date': request.args.get('date'),
                'days_before': request.args.get('days_before', 60)
            }
        else:
            data = request.get_json()
            
            if not data:
                return jsonify({
                    'status': 'error',
                    'message': 'No JSON data provided',
                    'data': None
                }), 400
        
        # Get time series data
        result = weather_controller.get_weather_time_series(data)
        
        status_code = 200 if result.get('status') == 'success' else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Time series API error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Time series API error: {str(e)}',
            'data': None
        }), 500

@weather_bp.route('/weather/batch', methods=['POST'])
def batch_weather_data():
    """
    Get weather data for multiple locations
    
    POST body:
        {
            "locations": [
                {
                    "latitude": float,
                    "longitude": float,
                    "disaster_date": "YYYY-MM-DD",
                    "days_before": int (optional)
                },
                ...
            ]
        }
    """
    try:
        if weather_controller is None:
            return jsonify({
                'status': 'error',
                'message': 'Weather service not initialized',
                'data': None
            }), 503
            
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided',
                'data': None
            }), 400
        
        # Get batch weather data
        result = weather_controller.batch_get_weather_data(data)
        
        status_code = 200 if result.get('status') == 'success' else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Batch weather API error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Batch weather API error: {str(e)}',
            'data': None
        }), 500

@weather_bp.route('/weather/summary', methods=['GET', 'POST'])
def get_weather_summary():
    """
    Get weather data summary statistics
    
    Parameters: Same as get_weather_data
    
    Returns summary statistics for all weather fields
    """
    try:
        if weather_controller is None:
            return jsonify({
                'status': 'error',
                'message': 'Weather service not initialized',
                'data': None
            }), 503
        if request.method == 'GET':
            data = {
                'latitude': request.args.get('lat'),
                'longitude': request.args.get('lon'),
                'disaster_date': request.args.get('date'),
                'days_before': request.args.get('days_before', 60)
            }
        else:
            data = request.get_json()
            
            if not data:
                return jsonify({
                    'status': 'error',
                    'message': 'No JSON data provided',
                    'data': None
                }), 400
        
        # Get weather summary
        result = weather_controller.get_weather_summary(data)
        
        status_code = 200 if result.get('status') == 'success' else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Weather summary API error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Weather summary API error: {str(e)}',
            'data': None
        }), 500

@weather_bp.route('/weather/fields', methods=['GET'])
def get_available_fields():
    """
    Get available weather fields and their descriptions
    
    Returns information about all available weather data fields
    """
    try:
        if weather_controller is None:
            return jsonify({
                'status': 'error',
                'message': 'Weather service not initialized',
                'data': None
            }), 503
        result = weather_controller.get_available_fields()
        
        status_code = 200 if result.get('status') == 'success' else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Available fields API error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Available fields API error: {str(e)}',
            'data': None
        }), 500

@weather_bp.route('/weather/status', methods=['GET'])
def get_service_status():
    """
    Get weather service status and health information
    
    Returns service health, initialization status, and configuration
    """
    try:
        if weather_controller is None:
            return jsonify({
                'status': 'error',
                'message': 'Weather service not initialized',
                'data': None
            }), 503
        result = weather_controller.get_service_status()
        
        status_code = 200 if result.get('status') == 'success' else 424  # Failed Dependency
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Service status API error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Service status API error: {str(e)}',
            'data': None
        }), 500

@weather_bp.route('/weather/test', methods=['GET'])
def test_weather_service():
    """
    Test the weather service with a known location
    
    Returns test results for NASA POWER API connectivity
    """
    try:
        if weather_controller is None:
            return jsonify({
                'status': 'error',
                'message': 'Weather service not initialized',
                'data': None
            }), 503
        # Use Mumbai coordinates as test location
        test_data = {
            'latitude': 19.076,
            'longitude': 72.8777,
            'disaster_date': '2024-01-15',
            'days_before': 7
        }
        
        logger.info("Testing weather service with Mumbai coordinates")
        result = weather_controller.get_weather_data(test_data)
        
        # Add test metadata
        if result.get('status') == 'success':
            result['data']['test_info'] = {
                'test_location': 'Mumbai, India',
                'test_coordinates': f"{test_data['latitude']}, {test_data['longitude']}",
                'test_date': test_data['disaster_date'],
                'test_period': f"{test_data['days_before']} days"
            }
        
        status_code = 200 if result.get('status') == 'success' else 424
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Weather test API error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Weather test API error: {str(e)}',
            'data': None
        }), 500

@weather_bp.errorhandler(400)
def bad_request(error):
    """Handle bad request errors"""
    return jsonify({
        'status': 'error',
        'message': 'Bad request: Invalid parameters',
        'data': None
    }), 400

@weather_bp.errorhandler(404)
def not_found(error):
    """Handle not found errors"""
    return jsonify({
        'status': 'error',
        'message': 'Weather endpoint not found',
        'data': None
    }), 404

@weather_bp.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    return jsonify({
        'status': 'error',
        'message': 'Internal server error',
        'data': None
    }), 500

# Blueprint registration function
def register_weather_routes(app):
    """Register weather routes with Flask app"""
    app.register_blueprint(weather_bp, url_prefix='/api')
    logger.info("Weather routes registered successfully")
    
    return weather_bp

# Route documentation
WEATHER_ROUTES_DOC = {
    'endpoints': {
        '/api/weather/data': {
            'methods': ['GET', 'POST'],
            'description': 'Get weather data for coordinates and date',
            'parameters': ['lat', 'lon', 'date', 'days_before (optional)']
        },
        '/api/weather/time-series': {
            'methods': ['GET', 'POST'],
            'description': 'Get weather data as time series',
            'parameters': ['lat', 'lon', 'date', 'days_before (optional)']
        },
        '/api/weather/batch': {
            'methods': ['POST'],
            'description': 'Get weather data for multiple locations',
            'parameters': ['locations array with lat/lon/date/days_before']
        },
        '/api/weather/summary': {
            'methods': ['GET', 'POST'],
            'description': 'Get weather data summary statistics',
            'parameters': ['lat', 'lon', 'date', 'days_before (optional)']
        },
        '/api/weather/fields': {
            'methods': ['GET'],
            'description': 'Get available weather fields and descriptions',
            'parameters': []
        },
        '/api/weather/status': {
            'methods': ['GET'],
            'description': 'Get weather service status and health',
            'parameters': []
        },
        '/api/weather/test': {
            'methods': ['GET'],
            'description': 'Test weather service connectivity',
            'parameters': []
        }
    },
    'data_source': 'NASA POWER API',
    'fields': 17,
    'temporal_resolution': 'daily',
    'max_batch_size': 100
}

if __name__ == '__main__':
    print("Weather Routes Documentation:")
    print(f"Available endpoints: {len(WEATHER_ROUTES_DOC['endpoints'])}")
    for endpoint, info in WEATHER_ROUTES_DOC['endpoints'].items():
        print(f"  {endpoint}: {info['description']}")