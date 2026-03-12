"""
Chat Controller
Handles chat-related business logic and service coordination
"""
import logging
from typing import Dict, Any, Optional
from flask import request
from services.ai_service import AIService
from services.gee_service import GEEService

class ChatController:
    """Controller for chat operations"""
    
    def __init__(self, ai_service: AIService, gee_service: GEEService):
        self.ai_service = ai_service
        self.gee_service = gee_service
        self.logger = logging.getLogger(__name__)
    
    def handle_chat_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming chat message and generate response
        
        Args:
            data: Request data containing message and optional context
            
        Returns:
            Response dictionary
        """
        try:
            # Extract message from request data
            message = data.get('message', '').strip()
            if not message:
                return {
                    'error': 'Message is required',
                    'status': 'error'
                }
            
            # Extract optional context
            context = data.get('context', {})
            
            # Check if satellite data is requested
            location = context.get('location')
            if location and isinstance(location, dict):
                lat = location.get('latitude')
                lon = location.get('longitude')
                
                if lat is not None and lon is not None:
                    # Add satellite data to context
                    context = self._enrich_context_with_satellite_data(context, lat, lon)
            
            # Generate AI response
            response = self.ai_service.generate_response(message, context)
            
            if response['status'] == 'success':
                return {
                    'response': response['message'],
                    'status': 'success',
                    'metadata': {
                        'model': response.get('model'),
                        'attempt': response.get('attempt'),
                        'context_enriched': bool(context)
                    }
                }
            else:
                return {
                    'error': response.get('message', 'Failed to generate response'),
                    'status': 'error'
                }
        
        except Exception as e:
            self.logger.error(f"Chat message processing error: {str(e)}")
            return {
                'error': f'Internal server error: {str(e)}',
                'status': 'error'
            }
    
    def _enrich_context_with_satellite_data(self, context: Dict[str, Any], lat: float, lon: float) -> Dict[str, Any]:
        """
        Enrich context with satellite data if GEE is available
        
        Args:
            context: Current context
            lat: Latitude
            lon: Longitude
            
        Returns:
            Enriched context
        """
        try:
            if self.gee_service.initialized:
                # Get recent satellite data (last 30 days)
                from datetime import datetime, timedelta
                
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                
                satellite_data = self.gee_service.get_satellite_data(
                    latitude=lat,
                    longitude=lon,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                
                context['satellite_data'] = satellite_data
                
        except Exception as e:
            self.logger.warning(f"Failed to enrich context with satellite data: {str(e)}")
        
        return context
    
    def analyze_location(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a specific location for disaster indicators
        
        Args:
            data: Request data containing location and analysis parameters
            
        Returns:
            Analysis results
        """
        try:
            # Extract location data
            latitude = data.get('latitude')
            longitude = data.get('longitude')
            
            if latitude is None or longitude is None:
                return {
                    'error': 'Latitude and longitude are required',
                    'status': 'error'
                }
            
            # Get satellite data
            from datetime import datetime, timedelta
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=data.get('days_back', 30))
            
            satellite_data = self.gee_service.get_satellite_data(
                latitude=latitude,
                longitude=longitude,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                cloud_filter=data.get('cloud_filter', 20)
            )
            
            # Get AI analysis
            user_query = data.get('query', 'Analyze this location for potential disaster indicators')
            analysis = self.ai_service.analyze_satellite_data(satellite_data, user_query)
            
            return {
                'status': 'success',
                'location': {
                    'latitude': latitude,
                    'longitude': longitude
                },
                'satellite_data': satellite_data,
                'analysis': analysis,
                'parameters': {
                    'days_back': data.get('days_back', 30),
                    'cloud_filter': data.get('cloud_filter', 20)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Location analysis error: {str(e)}")
            return {
                'error': f'Analysis failed: {str(e)}',
                'status': 'error'
            }
    
    def get_disaster_info(self, disaster_type: str, location_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get information about a specific disaster type
        
        Args:
            disaster_type: Type of disaster
            location_data: Optional location context
            
        Returns:
            Disaster information
        """
        try:
            # Validate disaster type
            valid_disasters = ['flood', 'drought', 'storm', 'landslide', 'wildfire', 'earthquake']
            if disaster_type.lower() not in valid_disasters:
                return {
                    'error': f'Invalid disaster type. Valid types: {", ".join(valid_disasters)}',
                    'status': 'error'
                }
            
            # Get AI insights
            insights = self.ai_service.get_disaster_insights(disaster_type, location_data)
            
            return {
                'status': 'success',
                'disaster_type': disaster_type,
                'insights': insights,
                'location_specific': bool(location_data)
            }
            
        except Exception as e:
            self.logger.error(f"Disaster info error: {str(e)}")
            return {
                'error': f'Failed to get disaster information: {str(e)}',
                'status': 'error'
            }