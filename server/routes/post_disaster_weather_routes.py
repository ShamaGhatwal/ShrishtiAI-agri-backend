"""
Post-Disaster Weather Data Routes for HazardGuard System
RESTful API endpoints for post-disaster weather data operations
"""

import logging
from flask import Blueprint, request, jsonify
from functools import wraps
import json

from controllers.post_disaster_weather_controller import PostDisasterWeatherController

logger = logging.getLogger(__name__)

# Global controller instance
post_disaster_weather_controller = None

def create_post_disaster_weather_routes(config: dict = None) -> Blueprint:
    """
    Create and configure post-disaster weather routes blueprint
    
    Args:
        config: Configuration dictionary for weather service settings
    
    Returns:
        Flask Blueprint with configured routes
    """
    global post_disaster_weather_controller
    
    # Create blueprint
    post_disaster_weather_bp = Blueprint('post_disaster_weather', __name__, url_prefix='/api/post-disaster-weather')
    
    # Initialize controller with configuration
    try:
        controller_config = config or {}
        post_disaster_weather_controller = PostDisasterWeatherController(
            days_after_disaster=controller_config.get('days_after_disaster', 60),
            max_workers=controller_config.get('max_workers', 1),
            retry_limit=controller_config.get('retry_limit', 5),
            retry_delay=controller_config.get('retry_delay', 15),
            rate_limit_pause=controller_config.get('rate_limit_pause', 900),
            request_delay=controller_config.get('request_delay', 0.5)
        )
        logger.info("Post-disaster weather controller initialized for routes")
    except Exception as e:
        logger.error(f"Failed to initialize post-disaster weather controller: {e}")
        raise
    
    def handle_json_errors(f):
        """Decorator to handle JSON parsing errors"""
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                if request.method in ['POST', 'PUT']:
                    if not request.is_json:
                        return jsonify({
                            'success': False,
                            'error': 'Request must be JSON format',
                            'status_code': 400
                        }), 400
                    
                    # Validate JSON can be parsed
                    request.get_json(force=True)
                
                return f(*args, **kwargs)
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                return jsonify({
                    'success': False,
                    'error': f'Invalid JSON format: {str(e)}',
                    'status_code': 400
                }), 400
            except Exception as e:
                logger.error(f"Request handling error: {e}")
                return jsonify({
                    'success': False,
                    'error': f'Request processing error: {str(e)}',
                    'status_code': 500
                }), 500
        
        return wrapper
    
    # ===== CORE PROCESSING ENDPOINTS =====
    
    @post_disaster_weather_bp.route('/process', methods=['POST'])
    @handle_json_errors
    def process_post_disaster_weather():
        """
        Process post-disaster weather extraction for coordinates
        
        Expected JSON:
        {
            \"coordinates\": [
                {\"latitude\": 12.9716, \"longitude\": 77.5946}
            ],
            \"disaster_dates\": [\"2023-01-15\"],
            \"variables\": [\"POST_temperature_C\", \"POST_precipitation_mm\"] # optional
        }
        """
        try:
            request_data = request.get_json()
            result = post_disaster_weather_controller.process_post_disaster_weather(request_data)
            return jsonify(result), result.get('status_code', 200)
            
        except Exception as e:
            logger.error(f"Process endpoint error: {e}")
            return jsonify({
                'success': False,
                'error': f'Processing failed: {str(e)}',
                'status_code': 500
            }), 500
    
    @post_disaster_weather_bp.route('/batch', methods=['POST'])
    @handle_json_errors  
    def process_batch_weather():
        """
        Process batch post-disaster weather extraction
        
        Expected JSON:
        {
            \"coordinates\": [
                {\"latitude\": 12.9716, \"longitude\": 77.5946},
                {\"latitude\": 17.3850, \"longitude\": 78.4867}
            ],
            \"disaster_dates\": [\"2023-01-15\", \"2023-02-20\"],
            \"variables\": [\"POST_temperature_C\"] # optional
        }
        """
        try:
            request_data = request.get_json()
            result = post_disaster_weather_controller.process_batch_weather(request_data)
            return jsonify(result), result.get('status_code', 200)
            
        except Exception as e:
            logger.error(f"Batch endpoint error: {e}")
            return jsonify({
                'success': False,
                'error': f'Batch processing failed: {str(e)}',
                'status_code': 500
            }), 500
    
    # ===== VALIDATION ENDPOINTS =====
    
    @post_disaster_weather_bp.route('/validate/coordinates', methods=['POST'])
    @handle_json_errors
    def validate_coordinates():
        """
        Validate coordinate format and ranges
        
        Expected JSON:
        {
            \"coordinates\": [
                {\"latitude\": 12.9716, \"longitude\": 77.5946}
            ]
        }
        """
        try:
            request_data = request.get_json()
            result = post_disaster_weather_controller.validate_coordinates(request_data)
            return jsonify(result), result.get('status_code', 200)
            
        except Exception as e:
            logger.error(f"Coordinate validation endpoint error: {e}")
            return jsonify({
                'success': False,
                'error': f'Coordinate validation failed: {str(e)}',
                'status_code': 500
            }), 500
    
    @post_disaster_weather_bp.route('/validate/dates', methods=['POST'])
    @handle_json_errors
    def validate_disaster_dates():
        """
        Validate disaster date format and ranges
        
        Expected JSON:
        {
            \"disaster_dates\": [\"2023-01-15\", \"2023-02-20\"]
        }
        """
        try:
            request_data = request.get_json()
            result = post_disaster_weather_controller.validate_disaster_dates(request_data)
            return jsonify(result), result.get('status_code', 200)
            
        except Exception as e:
            logger.error(f"Date validation endpoint error: {e}")
            return jsonify({
                'success': False,
                'error': f'Date validation failed: {str(e)}',
                'status_code': 500
            }), 500
    
    # ===== INFORMATION ENDPOINTS =====
    
    @post_disaster_weather_bp.route('/variables', methods=['GET'])
    def get_available_variables():
        """Get available post-disaster weather variables"""
        try:
            result = post_disaster_weather_controller.get_available_variables()
            return jsonify(result), result.get('status_code', 200)
            
        except Exception as e:
            logger.error(f"Variables endpoint error: {e}")
            return jsonify({
                'success': False,
                'error': f'Failed to get variables: {str(e)}',
                'status_code': 500
            }), 500
    
    @post_disaster_weather_bp.route('/info', methods=['GET'])
    def get_service_info():
        """Get comprehensive service information"""
        try:
            result = post_disaster_weather_controller.get_service_info()
            return jsonify(result), result.get('status_code', 200)
            
        except Exception as e:
            logger.error(f"Service info endpoint error: {e}")
            return jsonify({
                'success': False,
                'error': f'Failed to get service info: {str(e)}',
                'status_code': 500
            }), 500
    
    # ===== EXPORT ENDPOINTS =====
    
    @post_disaster_weather_bp.route('/export/dataframe', methods=['POST'])
    @handle_json_errors
    def export_to_dataframe():
        """
        Export weather data to DataFrame format
        
        Expected JSON:  
        {
            \"weather_data\": [
                {
                    \"latitude\": 12.9716,
                    \"longitude\": 77.5946,
                    \"disaster_date\": \"2023-01-15\",
                    \"POST_temperature_C\": [25.1, 26.2, ...],
                    \"success\": true
                }
            ]
        }
        """
        try:
            request_data = request.get_json()
            result = post_disaster_weather_controller.export_to_dataframe(request_data)
            return jsonify(result), result.get('status_code', 200)
            
        except Exception as e:
            logger.error(f"DataFrame export endpoint error: {e}")
            return jsonify({
                'success': False, 
                'error': f'DataFrame export failed: {str(e)}',
                'status_code': 500
            }), 500
    
    @post_disaster_weather_bp.route('/export/file', methods=['POST'])
    @handle_json_errors
    def export_to_file():
        """
        Export weather data to file
        
        Expected JSON:
        {
            \"weather_data\": [...],
            \"filepath\": \"/path/to/output.json\",
            \"file_format\": \"json\" # json, csv, xlsx
        }
        """
        try:
            request_data = request.get_json()
            result = post_disaster_weather_controller.export_to_file(request_data)
            return jsonify(result), result.get('status_code', 200)
            
        except Exception as e:
            logger.error(f"File export endpoint error: {e}")
            return jsonify({
                'success': False,
                'error': f'File export failed: {str(e)}',
                'status_code': 500
            }), 500
    
    # ===== STATUS AND MONITORING ENDPOINTS =====
    
    @post_disaster_weather_bp.route('/status', methods=['GET'])
    def get_processing_statistics():
        """Get service processing statistics and performance metrics"""
        try:
            result = post_disaster_weather_controller.get_processing_statistics()
            return jsonify(result), result.get('status_code', 200)
            
        except Exception as e:
            logger.error(f"Status endpoint error: {e}")
            return jsonify({
                'success': False,
                'error': f'Failed to get status: {str(e)}',
                'status_code': 500
            }), 500
    
    @post_disaster_weather_bp.route('/health', methods=['GET'])
    def health_check():
        """Service health check endpoint"""
        try:
            result = post_disaster_weather_controller.get_service_health()
            return jsonify(result), result.get('status_code', 200)
            
        except Exception as e:
            logger.error(f"Health check endpoint error: {e}")
            return jsonify({
                'success': False,
                'error': f'Health check failed: {str(e)}',
                'status_code': 500
            }), 500
    
    @post_disaster_weather_bp.route('/test', methods=['GET'])  
    def test_api_connection():
        """Test NASA POWER API connectivity"""
        try:
            result = post_disaster_weather_controller.test_api_connection()
            return jsonify(result), result.get('status_code', 200)
            
        except Exception as e:
            logger.error(f"API test endpoint error: {e}")
            return jsonify({
                'success': False,
                'error': f'API test failed: {str(e)}',
                'status_code': 500
            }), 500
    
    # ===== ERROR HANDLERS =====
    
    @post_disaster_weather_bp.errorhandler(404)
    def not_found(error):
        """Handle 404 errors"""
        return jsonify({
            'success': False,
            'error': 'Endpoint not found',
            'status_code': 404,
            'available_endpoints': [
                'POST /api/post-disaster-weather/process',
                'POST /api/post-disaster-weather/batch', 
                'POST /api/post-disaster-weather/validate/coordinates',
                'POST /api/post-disaster-weather/validate/dates',
                'GET  /api/post-disaster-weather/variables',
                'GET  /api/post-disaster-weather/info',
                'POST /api/post-disaster-weather/export/dataframe',
                'POST /api/post-disaster-weather/export/file',
                'GET  /api/post-disaster-weather/status',
                'GET  /api/post-disaster-weather/health',
                'GET  /api/post-disaster-weather/test'
            ]
        }), 404
    
    @post_disaster_weather_bp.errorhandler(405)
    def method_not_allowed(error):
        """Handle 405 errors"""
        return jsonify({
            'success': False,
            'error': f'Method {request.method} not allowed for this endpoint',
            'status_code': 405,
            'allowed_methods': error.description if hasattr(error, 'description') else 'Unknown'
        }), 405
    
    @post_disaster_weather_bp.errorhandler(500)
    def internal_server_error(error):
        """Handle 500 errors"""
        logger.error(f"Internal server error: {error}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'status_code': 500
        }), 500
    
    logger.info("Post-disaster weather routes configured successfully")
    logger.info("Available endpoints:")
    logger.info("  POST /api/post-disaster-weather/process - Extract weather data")
    logger.info("  POST /api/post-disaster-weather/batch - Batch processing")
    logger.info("  POST /api/post-disaster-weather/validate/coordinates - Validate coordinates")
    logger.info("  POST /api/post-disaster-weather/validate/dates - Validate disaster dates")
    logger.info("  GET  /api/post-disaster-weather/variables - Get available variables") 
    logger.info("  GET  /api/post-disaster-weather/info - Get service information")
    logger.info("  POST /api/post-disaster-weather/export/dataframe - Export to DataFrame")
    logger.info("  POST /api/post-disaster-weather/export/file - Export to file")
    logger.info("  GET  /api/post-disaster-weather/status - Get processing statistics")
    logger.info("  GET  /api/post-disaster-weather/health - Health check")
    logger.info("  GET  /api/post-disaster-weather/test - Test API connectivity")
    
    return post_disaster_weather_bp


# Initialize routes function for main app
def init_post_disaster_weather_routes(controller_instance: PostDisasterWeatherController = None):
    """
    Initialize post-disaster weather routes with existing controller instance
    
    Args:
        controller_instance: Existing controller instance to use
    """
    global post_disaster_weather_controller
    
    if controller_instance:
        post_disaster_weather_controller = controller_instance
        logger.info("Post-disaster weather routes initialized with existing controller")
    else:
        logger.warning("No controller instance provided for post-disaster weather routes")


# Standalone blueprint for testing
post_disaster_weather_bp = Blueprint('post_disaster_weather', __name__, url_prefix='/api/post-disaster-weather')

# Simple health check for standalone testing
@post_disaster_weather_bp.route('/health', methods=['GET'])
def standalone_health_check():
    """Standalone health check when controller not initialized"""
    return jsonify({
        'service': 'post_disaster_weather',
        'status': 'ready',
        'message': 'Post-disaster weather service is ready (controller not initialized)'
    }), 200