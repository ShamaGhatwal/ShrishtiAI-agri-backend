"""
Chat Routes
API endpoints for chat functionality
"""
from flask import Blueprint, request, jsonify
from controllers.chat_controller import ChatController
from typing import Dict, Any
import logging

# Create blueprint
chat_bp = Blueprint('chat', __name__, url_prefix='/api/chat')
logger = logging.getLogger(__name__)

# Controller will be injected via factory
chat_controller: ChatController = None

def init_chat_routes(controller: ChatController):
    """Initialize chat routes with controller"""
    global chat_controller
    chat_controller = controller

@chat_bp.route('/message', methods=['POST'])
def send_message():
    """
    Send a chat message and get AI response
    
    Expected JSON:
    {
        "message": "Your question here",
        "context": {  // Optional
            "location": {
                "latitude": 12.34,
                "longitude": 56.78
            }
        }
    }
    """
    try:
        if not chat_controller:
            return jsonify({
                'error': 'Chat service not initialized',
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
        
        # Process message
        result = chat_controller.handle_chat_message(data)
        
        if result['status'] == 'success':
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Chat message error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@chat_bp.route('/analyze', methods=['POST'])
def analyze_location():
    """
    Analyze a specific location for disaster indicators
    
    Expected JSON:
    {
        "latitude": 12.34,
        "longitude": 56.78,
        "days_back": 30,  // Optional, default 30
        "cloud_filter": 20,  // Optional, default 20
        "query": "Custom analysis query"  // Optional
    }
    """
    try:
        if not chat_controller:
            return jsonify({
                'error': 'Chat service not initialized',
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
        
        # Process analysis
        result = chat_controller.analyze_location(data)
        
        if result['status'] == 'success':
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Location analysis error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@chat_bp.route('/disaster/<disaster_type>', methods=['GET'])
def get_disaster_info(disaster_type: str):
    """
    Get information about a specific disaster type
    
    Query parameters:
    - latitude: Optional latitude for location-specific info
    - longitude: Optional longitude for location-specific info
    """
    try:
        if not chat_controller:
            return jsonify({
                'error': 'Chat service not initialized',
                'status': 'error'
            }), 500
        
        # Get optional location data
        location_data = None
        lat = request.args.get('latitude', type=float)
        lon = request.args.get('longitude', type=float)
        
        if lat is not None and lon is not None:
            location_data = {
                'latitude': lat,
                'longitude': lon
            }
        
        # Get disaster information
        result = chat_controller.get_disaster_info(disaster_type, location_data)
        
        if result['status'] == 'success':
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Disaster info error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'status': 'error'
        }), 500

@chat_bp.route('/health', methods=['GET'])
def chat_health():
    """Check chat service health"""
    try:
        if not chat_controller:
            return jsonify({
                'status': 'error',
                'message': 'Chat controller not initialized'
            }), 500
        
        # Check if AI service is initialized
        ai_initialized = chat_controller.ai_service.initialized
        gee_initialized = chat_controller.gee_service.initialized
        
        return jsonify({
            'status': 'success',
            'chat_service': 'healthy',
            'ai_service': 'healthy' if ai_initialized else 'unhealthy',
            'gee_service': 'healthy' if gee_initialized else 'unhealthy',
            'overall_health': 'healthy' if (ai_initialized and gee_initialized) else 'partial'
        }), 200
        
    except Exception as e:
        logger.error(f"Chat health check error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Health check failed'
        }), 500