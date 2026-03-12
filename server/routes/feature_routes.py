"""
Feature Engineering API Routes
RESTful endpoints for weather feature engineering operations
"""
from flask import Blueprint, request, jsonify
import logging
from controllers.feature_engineering_controller import FeatureEngineeringController
from services.feature_engineering_service import FeatureEngineeringService

# Initialize blueprint
features_bp = Blueprint('features', __name__)

# Configure logging
logger = logging.getLogger(__name__)

# Service and controller will be initialized by main app
feature_service = None
feature_controller = None

def init_feature_routes(controller_instance: FeatureEngineeringController):
    """Initialize feature engineering routes with controller instance"""
    global feature_controller
    feature_controller = controller_instance
    logger.info("Feature engineering routes initialized with controller")

@features_bp.route('/features/process', methods=['POST'])
def process_features():
    """
    Compute engineered features from weather data
    
    POST body:
        {
            "weather_data": {
                "temperature_C": [list of 60 daily values],
                "humidity_perc": [list of 60 daily values],
                ...
            },
            "event_duration": float (optional, default: 1.0),
            "include_metadata": bool (optional, default: true)
        }
    
    Response: Engineered features with 19 computed feature arrays
    """
    try:
        if feature_controller is None:
            return jsonify({
                'status': 'error',
                'message': 'Feature engineering service not initialized',
                'data': None
            }), 503
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided',
                'data': None
            }), 400
        
        # Process features
        result = feature_controller.process_features(data)
        
        # Return response with appropriate status code
        status_code = 200 if result.get('status') == 'success' else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Features process API error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Features API error: {str(e)}',
            'data': None
        }), 500

@features_bp.route('/features/batch', methods=['POST'])
def process_batch_features():
    """
    Process multiple weather datasets for feature engineering
    
    POST body:
        {
            "batch_data": [
                {
                    "id": "optional_identifier",
                    "weather_data": {...},
                    "event_duration": float (optional)
                },
                ...
            ],
            "include_metadata": bool (optional, default: true)
        }
    
    Maximum batch size: 100 items
    """
    try:
        if feature_controller is None:
            return jsonify({
                'status': 'error',
                'message': 'Feature engineering service not initialized',
                'data': None
            }), 503
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided',
                'data': None
            }), 400
        
        # Process batch
        result = feature_controller.process_batch_features(data)
        
        status_code = 200 if result.get('status') == 'success' else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Batch features API error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Batch features API error: {str(e)}',
            'data': None
        }), 500

@features_bp.route('/features/dataframe', methods=['POST'])
def create_feature_dataframe():
    """
    Create time series DataFrame with weather data and engineered features
    
    POST body:
        {
            "weather_data": {...},
            "disaster_date": "YYYY-MM-DD",
            "days_before": int,
            "event_duration": float (optional, default: 1.0)
        }
    
    Returns DataFrame with dates, weather data, and engineered features
    """
    try:
        if feature_controller is None:
            return jsonify({
                'status': 'error',
                'message': 'Feature engineering service not initialized',
                'data': None
            }), 503
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided',
                'data': None
            }), 400
        
        # Create DataFrame
        result = feature_controller.create_feature_dataframe(data)
        
        status_code = 200 if result.get('status') == 'success' else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Feature DataFrame API error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Feature DataFrame API error: {str(e)}',
            'data': None
        }), 500

@features_bp.route('/features/validate', methods=['POST'])
def validate_weather_data():
    """
    Validate weather data for feature engineering readiness
    
    POST body:
        {
            "weather_data": {
                "temperature_C": [...],
                "humidity_perc": [...],
                ...
            }
        }
    
    Returns validation results and readiness status
    """
    try:
        if feature_controller is None:
            return jsonify({
                'status': 'error',
                'message': 'Feature engineering service not initialized',
                'data': None
            }), 503
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided',
                'data': None
            }), 400
        
        # Validate data
        result = feature_controller.validate_weather_data(data)
        
        status_code = 200 if result.get('status') == 'success' else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Weather validation API error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Weather validation API error: {str(e)}',
            'data': None
        }), 500

@features_bp.route('/features/export', methods=['POST'])
def process_and_export():
    """
    Process features and export in specified format
    
    POST body:
        {
            "weather_data": {...},
            "disaster_date": "YYYY-MM-DD",
            "days_before": int,
            "event_duration": float (optional, default: 1.0),
            "export_format": "dict|dataframe|json" (optional, default: "dict")
        }
    
    Returns processed features in requested format
    """
    try:
        if feature_controller is None:
            return jsonify({
                'status': 'error',
                'message': 'Feature engineering service not initialized',
                'data': None
            }), 503
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided',
                'data': None
            }), 400
        
        # Process and export
        result = feature_controller.process_and_export(data)
        
        status_code = 200 if result.get('status') == 'success' else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Process export API error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Process export API error: {str(e)}',
            'data': None
        }), 500

@features_bp.route('/features/info', methods=['GET'])
def get_feature_info():
    """
    Get information about available engineered features
    
    Returns detailed information about all 19 engineered features
    """
    try:
        if feature_controller is None:
            return jsonify({
                'status': 'error',
                'message': 'Feature engineering service not initialized',
                'data': None
            }), 503
        
        result = feature_controller.get_feature_info()
        
        status_code = 200 if result.get('status') == 'success' else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Feature info API error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Feature info API error: {str(e)}',
            'data': None
        }), 500

@features_bp.route('/features/status', methods=['GET'])
def get_service_status():
    """
    Get feature engineering service status and health
    
    Returns service health, initialization status, and configuration
    """
    try:
        if feature_controller is None:
            return jsonify({
                'status': 'error',
                'message': 'Feature engineering service not initialized',
                'data': None
            }), 503
        
        result = feature_controller.get_service_status()
        
        status_code = 200 if result.get('status') == 'success' else 424  # Failed Dependency
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Feature status API error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Feature status API error: {str(e)}',
            'data': None
        }), 500

@features_bp.route('/features/test', methods=['GET', 'POST'])
def test_feature_service():
    """
    Test the feature engineering service with sample data
    
    Returns test results for feature engineering functionality
    """
    try:
        if feature_controller is None:
            return jsonify({
                'status': 'error',
                'message': 'Feature engineering service not initialized',
                'data': None
            }), 503
        
        # Create sample weather data (7 days)
        sample_weather_data = {
            'temperature_C': [25.5, 26.0, 24.8, 27.2, 25.9, 25.1, 26.3],
            'humidity_perc': [65.2, 67.8, 70.1, 62.5, 68.9, 66.0, 66.4],
            'wind_speed_mps': [3.2, 2.8, 4.1, 5.6, 3.9, 2.4, 3.7],
            'precipitation_mm': [0.0, 2.3, 5.1, 0.0, 1.2, 0.8, 3.4],
            'surface_pressure_hPa': [1013.2, 1012.8, 1011.5, 1014.1, 1013.7, 1012.9, 1013.4],
            'solar_radiation_wm2': [220.5, 180.3, 150.8, 240.2, 200.1, 190.7, 210.9],
            'temperature_max_C': [30.2, 31.1, 29.5, 32.8, 30.9, 29.8, 31.5],
            'temperature_min_C': [20.8, 20.9, 20.1, 21.6, 21.0, 20.4, 21.2],
            'specific_humidity_g_kg': [12.5, 13.1, 13.8, 11.9, 13.2, 12.8, 12.9],
            'dew_point_C': [18.2, 19.1, 19.8, 17.5, 18.9, 18.4, 18.7],
            'wind_speed_10m_mps': [4.1, 3.6, 5.2, 7.1, 4.9, 3.0, 4.6],
            'cloud_amount_perc': [30.0, 60.0, 80.0, 20.0, 50.0, 40.0, 70.0],
            'sea_level_pressure_hPa': [1013.5, 1013.1, 1011.8, 1014.4, 1014.0, 1013.2, 1013.7],
            'surface_soil_wetness_perc': [45.0, 52.0, 68.0, 42.0, 48.0, 50.0, 58.0],
            'wind_direction_10m_degrees': [180.0, 165.0, 220.0, 195.0, 170.0, 210.0, 185.0],
            'evapotranspiration_wm2': [85.2, 72.1, 58.9, 95.8, 82.4, 78.6, 88.3],
            'root_zone_soil_moisture_perc': [55.0, 61.0, 74.0, 52.0, 58.0, 60.0, 68.0]
        }
        
        test_data = {
            'weather_data': sample_weather_data,
            'event_duration': 2.0,
            'include_metadata': True
        }
        
        logger.info("Testing feature engineering service with sample weather data")
        result = feature_controller.process_features(test_data)
        
        # Add test metadata
        if result.get('status') == 'success':
            result['data']['test_info'] = {
                'test_description': 'Sample 7-day weather data processing',
                'sample_size': 7,
                'weather_fields': len(sample_weather_data),
                'features_computed': len(result['data']['engineered_features']) if 'engineered_features' in result['data'] else 0
            }
        
        status_code = 200 if result.get('status') == 'success' else 424
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Feature test API error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Feature test API error: {str(e)}',
            'data': None
        }), 500

@features_bp.errorhandler(400)
def bad_request(error):
    """Handle bad request errors"""
    return jsonify({
        'status': 'error',
        'message': 'Bad request: Invalid parameters',
        'data': None
    }), 400

@features_bp.errorhandler(404)
def not_found(error):
    """Handle not found errors"""
    return jsonify({
        'status': 'error',
        'message': 'Feature engineering endpoint not found',
        'data': None
    }), 404

@features_bp.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    return jsonify({
        'status': 'error',
        'message': 'Internal server error',
        'data': None
    }), 500

# Blueprint registration function
def register_feature_routes(app):
    """Register feature engineering routes with Flask app"""
    app.register_blueprint(features_bp, url_prefix='/api')
    logger.info("Feature engineering routes registered successfully")
    
    return features_bp

# Route documentation
FEATURE_ROUTES_DOC = {
    'endpoints': {
        '/api/features/process': {
            'methods': ['POST'],
            'description': 'Compute engineered features from weather data',
            'parameters': ['weather_data', 'event_duration (optional)', 'include_metadata (optional)']
        },
        '/api/features/batch': {
            'methods': ['POST'],
            'description': 'Process multiple weather datasets (max 100)',
            'parameters': ['batch_data array with weather_data/event_duration']
        },
        '/api/features/dataframe': {
            'methods': ['POST'],
            'description': 'Create time series DataFrame with features',
            'parameters': ['weather_data', 'disaster_date', 'days_before', 'event_duration (optional)']
        },
        '/api/features/validate': {
            'methods': ['POST'],
            'description': 'Validate weather data readiness',
            'parameters': ['weather_data']
        },
        '/api/features/export': {
            'methods': ['POST'],
            'description': 'Process and export in specified format',
            'parameters': ['weather_data', 'disaster_date', 'days_before', 'export_format (optional)']
        },
        '/api/features/info': {
            'methods': ['GET'],
            'description': 'Get information about engineered features',
            'parameters': []
        },
        '/api/features/status': {
            'methods': ['GET'],
            'description': 'Get service status and health',
            'parameters': []
        },
        '/api/features/test': {
            'methods': ['GET', 'POST'],
            'description': 'Test feature engineering service',
            'parameters': []
        }
    },
    'features_computed': 19,
    'weather_fields_required': 17,
    'nan_handling': 'Proper NaN propagation - NaN input produces NaN output for that day only',
    'max_batch_size': 100
}

if __name__ == '__main__':
    print("Feature Engineering Routes Documentation:")
    print(f"Available endpoints: {len(FEATURE_ROUTES_DOC['endpoints'])}")
    for endpoint, info in FEATURE_ROUTES_DOC['endpoints'].items():
        print(f"  {endpoint}: {info['description']}")