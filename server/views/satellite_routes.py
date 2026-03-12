"""
Satellite Routes
API endpoints for satellite data functionality
"""
from flask import Blueprint, request, jsonify
from controllers.satellite_controller import SatelliteController
from typing import Dict, Any
import logging

# Create blueprint
satellite_bp = Blueprint('satellite', __name__, url_prefix='/api/satellite')
logger = logging.getLogger(__name__)

# Controller will be injected via factory
satellite_controller: SatelliteController = None

def init_satellite_routes(controller: SatelliteController):
    """Initialize satellite routes with controller"""
    global satellite_controller
    satellite_controller = controller

@satellite_bp.route('/point', methods=['GET'])
def get_point_data():
    """
    Get satellite data for a specific point
    
    Query parameters:
    - latitude: Required latitude coordinate
    - longitude: Required longitude coordinate
    - start_date: Optional start date (YYYY-MM-DD), default: 30 days ago
    - end_date: Optional end date (YYYY-MM-DD), default: today
    - collection: Optional satellite collection, default: COPERNICUS/S2_SR
    - cloud_filter: Optional cloud coverage threshold (0-100), default: 20
    """
    try:
        if not satellite_controller:
            return jsonify({
                'error': 'Satellite service not initialized',
                'status': 'error'
            }), 500
        
        # Parse query parameters
        data = {
            'latitude': request.args.get('latitude', type=float),
            'longitude': request.args.get('longitude', type=float),
            'start_date': request.args.get('start_date'),
            'end_date': request.args.get('end_date'),
            'collection': request.args.get('collection', 'COPERNICUS/S2_SR'),
            'cloud_filter': request.args.get('cloud_filter', type=int, default=20)
        }
        
        # Process request
        result = satellite_controller.get_point_data(data)
        
        if result['status'] == 'success':
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Point data error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@satellite_bp.route('/point', methods=['POST'])
def post_point_data():
    """
    Get satellite data for a specific point (POST version for complex parameters)
    
    Expected JSON:
    {
        "latitude": 12.34,
        "longitude": 56.78,
        "start_date": "2024-01-01",  // Optional
        "end_date": "2024-01-31",    // Optional
        "collection": "COPERNICUS/S2_SR",  // Optional
        "cloud_filter": 20  // Optional
    }
    """
    try:
        if not satellite_controller:
            return jsonify({
                'error': 'Satellite service not initialized',
                'status': 'error'
            }), 500
        
        # Validate request
        if not request.is_json:
            return jsonify({
                'error': 'Content-Type must be application/json',
                'status': 'error'
            }), 400
        
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'Request body is required',
                'status': 'error'
            }), 400
        
        # Process request
        result = satellite_controller.get_point_data(data)
        
        if result['status'] == 'success':
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Point data POST error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@satellite_bp.route('/region', methods=['POST'])
def get_region_data():
    """
    Get satellite data for a region
    
    Expected JSON:
    {
        "bounds": [
            [-122.5, 37.7],  // [longitude, latitude] pairs
            [-122.4, 37.8],
            [-122.3, 37.7]
        ],
        "start_date": "2024-01-01",  // Optional
        "end_date": "2024-01-31",    // Optional
        "scale": 10  // Optional, pixel scale in meters
    }
    """
    try:
        if not satellite_controller:
            return jsonify({
                'error': 'Satellite service not initialized',
                'status': 'error'
            }), 500
        
        # Validate request
        if not request.is_json:
            return jsonify({
                'error': 'Content-Type must be application/json',
                'status': 'error'
            }), 400
        
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'Request body is required',
                'status': 'error'
            }), 400
        
        # Process request
        result = satellite_controller.get_region_data(data)
        
        if result['status'] == 'success':
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Region data error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@satellite_bp.route('/availability', methods=['GET'])
def check_availability():
    """
    Check satellite data availability for a location
    
    Query parameters:
    - latitude: Required latitude coordinate
    - longitude: Required longitude coordinate
    - days_back: Optional number of days to check back (1-365), default: 30
    """
    try:
        if not satellite_controller:
            return jsonify({
                'error': 'Satellite service not initialized',
                'status': 'error'
            }), 500
        
        # Parse query parameters
        data = {
            'latitude': request.args.get('latitude', type=float),
            'longitude': request.args.get('longitude', type=float),
            'days_back': request.args.get('days_back', type=int, default=30)
        }
        
        # Process request
        result = satellite_controller.check_availability(data)
        
        if result['status'] == 'success':
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Availability check error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@satellite_bp.route('/availability', methods=['POST'])
def post_availability():
    """
    Check satellite data availability (POST version)
    
    Expected JSON:
    {
        "latitude": 12.34,
        "longitude": 56.78,
        "days_back": 30  // Optional
    }
    """
    try:
        if not satellite_controller:
            return jsonify({
                'error': 'Satellite service not initialized',
                'status': 'error'
            }), 500
        
        # Validate request
        if not request.is_json:
            return jsonify({
                'error': 'Content-Type must be application/json',
                'status': 'error'
            }), 400
        
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'Request body is required',
                'status': 'error'
            }), 400
        
        # Process request
        result = satellite_controller.check_availability(data)
        
        if result['status'] == 'success':
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Availability POST error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@satellite_bp.route('/status', methods=['GET'])
def get_service_status():
    """Get satellite service status"""
    try:
        if not satellite_controller:
            return jsonify({
                'error': 'Satellite service not initialized',
                'status': 'error'
            }), 500
        
        result = satellite_controller.get_service_status()
        
        if result['status'] == 'success':
            return jsonify(result), 200
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Service status error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@satellite_bp.route('/collections', methods=['GET'])
def get_available_collections():
    """Get list of available satellite collections"""
    try:
        collections = [
            {
                'id': 'COPERNICUS/S2_SR',
                'name': 'Sentinel-2 MSI: MultiSpectral Instrument, Level-2A',
                'description': 'Atmospheric corrected surface reflectance',
                'resolution': '10-60m',
                'bands': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'],
                'revisit_time': '5 days'
            },
            {
                'id': 'COPERNICUS/S2',
                'name': 'Sentinel-2 MSI: MultiSpectral Instrument, Level-1C',
                'description': 'Top-of-atmosphere reflectance',
                'resolution': '10-60m',
                'bands': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12'],
                'revisit_time': '5 days'
            },
            {
                'id': 'LANDSAT/LC08/C02/T1_L2',
                'name': 'Landsat 8 Collection 2 Tier 1 Level-2',
                'description': 'Atmospherically corrected surface reflectance',
                'resolution': '30m',
                'bands': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
                'revisit_time': '16 days'
            }
        ]
        
        return jsonify({
            'status': 'success',
            'collections': collections
        }), 200
        
    except Exception as e:
        logger.error(f"Collections error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500