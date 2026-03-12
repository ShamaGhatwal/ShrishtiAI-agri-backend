"""
Raster Data Routes for HazardGuard System
RESTful API endpoints for raster data operations
"""

from flask import Blueprint, request, jsonify
import logging

from controllers.raster_data_controller import RasterDataController

logger = logging.getLogger(__name__)

def create_raster_routes(raster_config=None):
    """Create and configure raster data routes"""
    
    # Create Blueprint
    raster_bp = Blueprint('raster', __name__, url_prefix='/api/raster')
    
    # Initialize controller
    controller = RasterDataController(raster_config)
    
    @raster_bp.route('/process', methods=['POST'])
    def process_raster_extraction():
        """
        Extract raster data for given coordinates
        
        Request body:
        {
            "coordinates": [
                {"longitude": 121.0, "latitude": 14.0},
                {"longitude": 122.0, "latitude": 15.0}
            ],
            "features": ["soil_type", "elevation_m"]  # optional
        }
        
        Returns:
        {
            "success": true,
            "message": "Successfully extracted raster data for 2 coordinates",
            "data": [
                {
                    "longitude": 121.0,
                    "latitude": 14.0,
                    "soil_type": 6,
                    "elevation_m": 123.45,
                    ...
                }
            ],
            "metadata": {
                "coordinates_processed": 2,
                "features_extracted": 9,
                "processing_time_seconds": 1.23
            }
        }
        """
        try:
            if not request.is_json:
                return jsonify({
                    'success': False,
                    'error': 'Request must be JSON',
                    'data': None
                }), 400
            
            result = controller.process_raster_extraction(request.get_json())
            return jsonify(result), result.get('status_code', 200)
            
        except Exception as e:
            logger.error(f"Error in raster extraction endpoint: {e}")
            return jsonify({
                'success': False,
                'error': f'Internal server error: {str(e)}',
                'data': None
            }), 500
    
    @raster_bp.route('/batch', methods=['POST'])
    def process_batch_extraction():
        """
        Extract raster data for large coordinate sets in batches
        
        Request body:
        {
            "coordinates": [...],  # large list of coordinates
            "batch_size": 100,     # optional, default 100
            "features": ["soil_type", "elevation_m"]  # optional
        }
        
        Returns:
        {
            "success": true,
            "message": "Successfully processed batch extraction for 1000 coordinates",
            "data": [...],  # all processed results
            "metadata": {
                "total_coordinates": 1000,
                "batch_size": 100,
                "batches_processed": 10,
                "coordinates_processed": 1000,
                "processing_time_seconds": 45.67
            }
        }
        """
        try:
            if not request.is_json:
                return jsonify({
                    'success': False,
                    'error': 'Request must be JSON',
                    'data': None
                }), 400
            
            result = controller.process_batch_extraction(request.get_json())
            return jsonify(result), result.get('status_code', 200)
            
        except Exception as e:
            logger.error(f"Error in batch raster extraction endpoint: {e}")
            return jsonify({
                'success': False,
                'error': f'Internal server error: {str(e)}',
                'data': None
            }), 500
    
    @raster_bp.route('/dataframe', methods=['POST'])
    def create_dataframe():
        """
        Create pandas DataFrame from raster extraction
        
        Request body:
        {
            "coordinates": [...],
            "features": ["soil_type", "elevation_m"]  # optional
        }
        
        Returns:
        {
            "success": true,
            "message": "Successfully created DataFrame with 100 rows",
            "data": [...],  # DataFrame records
            "metadata": {
                "dataframe_shape": [100, 11],
                "dataframe_columns": ["longitude", "latitude", "soil_type", ...],
                "dataframe_info": "DataFrame info string"
            }
        }
        """
        try:
            if not request.is_json:
                return jsonify({
                    'success': False,
                    'error': 'Request must be JSON',
                    'data': None
                }), 400
            
            result = controller.create_dataframe(request.get_json())
            return jsonify(result), result.get('status_code', 200)
            
        except Exception as e:
            logger.error(f"Error in DataFrame creation endpoint: {e}")
            return jsonify({
                'success': False,
                'error': f'Internal server error: {str(e)}',
                'data': None
            }), 500
    
    @raster_bp.route('/export', methods=['POST'])
    def export_data():
        """
        Export raster data in various formats
        
        Request body:
        {
            "coordinates": [...],
            "format": "json|csv|excel",
            "features": ["soil_type", "elevation_m"]  # optional
        }
        
        Returns:
        {
            "success": true,
            "message": "Successfully exported data in csv format",
            "data": "longitude,latitude,soil_type,elevation_m\\n121.0,14.0,6,123.45",
            "metadata": {
                "export_format": "csv",
                "content_type": "text/csv"
            }
        }
        """
        try:
            if not request.is_json:
                return jsonify({
                    'success': False,
                    'error': 'Request must be JSON',
                    'data': None
                }), 400
            
            result = controller.export_data(request.get_json())
            return jsonify(result), result.get('status_code', 200)
            
        except Exception as e:
            logger.error(f"Error in data export endpoint: {e}")
            return jsonify({
                'success': False,
                'error': f'Internal server error: {str(e)}',
                'data': None
            }), 500
    
    @raster_bp.route('/validate', methods=['POST'])
    def validate_coordinates():
        """
        Validate coordinate format and ranges
        
        Request body:
        {
            "coordinates": [
                {"longitude": 121.0, "latitude": 14.0},
                {"longitude": 181.0, "latitude": 95.0}  # invalid
            ]
        }
        
        Returns:
        {
            "success": true,
            "message": "Coordinate validation completed",
            "data": {
                "valid": false,
                "message": "Coordinates contain invalid longitude or latitude values",
                "coordinate_count": 2
            }
        }
        """
        try:
            if not request.is_json:
                return jsonify({
                    'success': False,
                    'error': 'Request must be JSON',
                    'data': None
                }), 400
            
            result = controller.validate_coordinates(request.get_json())
            return jsonify(result), result.get('status_code', 200)
            
        except Exception as e:
            logger.error(f"Error in coordinate validation endpoint: {e}")
            return jsonify({
                'success': False,
                'error': f'Internal server error: {str(e)}',
                'data': None
            }), 500
    
    @raster_bp.route('/features', methods=['GET'])
    def get_available_features():
        """
        Get information about available raster features
        
        Returns:
        {
            "success": true,
            "message": "Successfully retrieved available features",
            "data": {
                "soil_type": {
                    "description": "Soil classification (HWSD2)",
                    "range": "0-33 (encoded classes)",
                    "unit": "categorical",
                    "available": true
                },
                ...
            },
            "metadata": {
                "availability": {...},
                "configuration": {...}
            }
        }
        """
        try:
            result = controller.get_available_features()
            return jsonify(result), result.get('status_code', 200)
            
        except Exception as e:
            logger.error(f"Error in get available features endpoint: {e}")
            return jsonify({
                'success': False,
                'error': f'Internal server error: {str(e)}',
                'data': None
            }), 500
    
    @raster_bp.route('/info', methods=['GET'])
    def get_feature_info():
        """
        Get detailed information about raster features
        
        Returns:
        {
            "success": true,
            "message": "Feature information retrieved successfully",
            "data": {
                "soil_type": {
                    "description": "Soil classification (HWSD2)",
                    "range": "0-33 (encoded classes)",
                    "unit": "categorical",
                    "available": true,
                    "path_configured": true
                },
                ...
            },
            "metadata": {
                "total_features": 9,
                "nodata_values": {"numeric": -9999.0, "categorical": 0},
                "coordinate_system": "EPSG:4326 (WGS84)"
            }
        }
        """
        try:
            result = controller.get_feature_info()
            return jsonify(result), result.get('status_code', 200)
            
        except Exception as e:
            logger.error(f"Error in get feature info endpoint: {e}")
            return jsonify({
                'success': False,
                'error': f'Internal server error: {str(e)}',
                'data': None
            }), 500
    
    @raster_bp.route('/status', methods=['GET'])
    def get_service_status():
        """
        Get raster service status and health information
        
        Returns:
        {
            "success": true,
            "message": "Service status retrieved successfully",
            "data": {
                "service_health": "healthy|degraded|no_data",
                "request_count": 42,
                "processing_statistics": {
                    "total_extractions": 20,
                    "successful_extractions": 18,
                    "failed_extractions": 2,
                    "success_rate": 90.0
                },
                "configuration_validation": {
                    "soil": {"exists": true, "readable": true},
                    ...
                }
            }
        }
        """
        try:
            result = controller.get_service_status()
            return jsonify(result), result.get('status_code', 200)
            
        except Exception as e:
            logger.error(f"Error in service status endpoint: {e}")
            return jsonify({
                'success': False,
                'error': f'Internal server error: {str(e)}',
                'data': None
            }), 500
    
    @raster_bp.route('/test', methods=['GET'])
    def test_extraction():
        """
        Test raster extraction with sample coordinates
        
        Returns:
        {
            "success": true,
            "message": "Raster extraction test successful",
            "data": {
                "longitude": 121.0,
                "latitude": 14.0,
                "soil_type": 6,
                "elevation_m": 123.45,
                ...
            },
            "metadata": {
                "processing_time": 0.5,
                "test_coordinates": [{"longitude": 121.0, "latitude": 14.0}]
            }
        }
        """
        try:
            result = controller.test_extraction()
            return jsonify(result), result.get('status_code', 200)
            
        except Exception as e:
            logger.error(f"Error in test extraction endpoint: {e}")
            return jsonify({
                'success': False,
                'error': f'Internal server error: {str(e)}',
                'data': None
            }), 500
    
    @raster_bp.route('/health', methods=['GET'])
    def health_check():
        """
        Simple health check endpoint
        
        Returns:
        {
            "status": "healthy",
            "service": "raster_data",
            "message": "Raster data service is operational"
        }
        """
        try:
            return jsonify({
                'status': 'healthy',
                'service': 'raster_data',
                'message': 'Raster data service is operational'
            }), 200
            
        except Exception as e:
            logger.error(f"Error in health check endpoint: {e}")
            return jsonify({
                'status': 'unhealthy',
                'service': 'raster_data',
                'message': f'Service error: {str(e)}'
            }), 500
    
    # Error handlers for the blueprint
    @raster_bp.errorhandler(404)
    def not_found_error(error):
        return jsonify({
            'success': False,
            'error': 'Endpoint not found',
            'data': None,
            'available_endpoints': [
                '/api/raster/process',
                '/api/raster/batch',
                '/api/raster/dataframe',
                '/api/raster/export',
                '/api/raster/validate',
                '/api/raster/features',
                '/api/raster/info',
                '/api/raster/status',
                '/api/raster/test',
                '/api/raster/health'
            ]
        }), 404
    
    @raster_bp.errorhandler(405)
    def method_not_allowed_error(error):
        return jsonify({
            'success': False,
            'error': 'Method not allowed for this endpoint',
            'data': None
        }), 405
    
    @raster_bp.errorhandler(500)
    def internal_server_error(error):
        logger.error(f"Internal server error in raster routes: {error}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'data': None
        }), 500
    
    return raster_bp