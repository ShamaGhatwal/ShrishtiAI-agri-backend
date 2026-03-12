"""
WeatherWise Prediction Controller
Handles HTTP requests for LSTM weather forecasting
"""

import logging
from typing import Dict, Any
from datetime import datetime

from services.weatherwise_prediction_service import WeatherWisePredictionService

logger = logging.getLogger(__name__)

class WeatherWisePredictionController:
    """Controller for handling WeatherWise LSTM weather forecasting requests"""
    
    def __init__(self, weatherwise_service: WeatherWisePredictionService = None):
        """Initialize WeatherWise controller"""
        self.service = weatherwise_service or WeatherWisePredictionService()
        self.controller_stats = {
            'controller_start_time': datetime.now().isoformat(),
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0
        }
        
        logger.info("WeatherWise prediction controller initialized")
    
    def initialize_controller(self) -> Dict[str, Any]:
        """
        Initialize the controller by setting up the WeatherWise service
        
        Returns:
            Initialize response dictionary
        """
        try:
            logger.info("Initializing WeatherWise prediction controller...")
            
            # Initialize the WeatherWise service
            service_success, service_message = self.service.initialize_service()
            
            if service_success:
                available_models = self.service.get_available_models()
                logger.info(f"[SUCCESS] WeatherWise controller initialized with {len(available_models)} models")
                
                return self._create_response(
                    success=True,
                    message="WeatherWise controller initialized successfully",
                    data={
                        'service_status': 'initialized',
                        'available_models': available_models,
                        'default_forecast_days': 60,
                        'supported_variables': self.service.prediction_model.forecast_variables
                    }
                )
            else:
                logger.error(f"[ERROR] WeatherWise service initialization failed: {service_message}")
                return self._create_response(
                    success=False,
                    message="WeatherWise controller initialization failed",
                    error=service_message
                )
                
        except Exception as e:
            logger.error(f"WeatherWise controller initialization error: {e}")
            return self._create_response(
                success=False,
                message="WeatherWise controller initialization error",
                error=f"Controller error: {str(e)}"
            )
    
    def forecast_weather(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate weather forecast for location
        
        Args:
            request_data: Request dictionary with latitude, longitude, etc.
        
        Returns:
            Forecast response dictionary
        """
        self.controller_stats['total_requests'] += 1
        
        try:
            logger.info(f"[WEATHERWISE_CONTROLLER] ===== FORECAST REQUEST START =====")
            logger.info(f"[WEATHERWISE_CONTROLLER] Processing forecast request...")
            logger.info(f"[WEATHERWISE_CONTROLLER] Total requests so far: {self.controller_stats['total_requests']}")
            
            # Extract request parameters
            latitude = request_data.get('latitude')
            longitude = request_data.get('longitude')
            reference_date = request_data.get('reference_date')
            disaster_type = request_data.get('disaster_type', 'Normal')
            forecast_days = request_data.get('forecast_days', 60)
            
            logger.info(f"[WEATHERWISE_CONTROLLER] Extracted parameters:")
            logger.info(f"[WEATHERWISE_CONTROLLER]   - latitude: {latitude}")
            logger.info(f"[WEATHERWISE_CONTROLLER]   - longitude: {longitude}")
            logger.info(f"[WEATHERWISE_CONTROLLER]   - reference_date: {reference_date}")
            logger.info(f"[WEATHERWISE_CONTROLLER]   - disaster_type: {disaster_type}")
            logger.info(f"[WEATHERWISE_CONTROLLER]   - forecast_days: {forecast_days}")
            
            # Reject if TF models are still warming up in the background thread
            if not self.service.service_stats.get('models_loaded'):
                return self._create_response(
                    success=False,
                    message="Service is warming up, please retry in a moment",
                    error="models_not_ready"
                )

            # Validate required parameters
            if latitude is None or longitude is None:
                logger.error("[WEATHERWISE_CONTROLLER] Missing required parameters")
                return self._create_response(
                    success=False,
                    message="Missing required parameters",
                    error="Both 'latitude' and 'longitude' are required"
                )
            
            logger.info(f"[WEATHERWISE_CONTROLLER] Parameters validated successfully")
            
            # Validate disaster type
            available_models = self.service.get_available_models()
            if disaster_type not in available_models and available_models:
                logger.warning(f"Requested disaster type '{disaster_type}' not available, using '{available_models[0]}'")
                disaster_type = available_models[0]
            
            # Validate forecast days
            try:
                forecast_days = int(forecast_days)
                if forecast_days < 1 or forecast_days > 365:
                    forecast_days = 60  # Default
            except (ValueError, TypeError):
                forecast_days = 60
            
            logger.info(f"[WEATHERWISE_CONTROLLER] Calling service.generate_weather_forecast()...")
            logger.info(f"Processing weather forecast for ({latitude}, {longitude})")
            logger.info(f"Parameters: disaster_type={disaster_type}, forecast_days={forecast_days}, reference_date={reference_date}")
            
            # Generate forecast using service
            forecast_result = self.service.generate_weather_forecast(
                latitude=latitude,
                longitude=longitude,
                reference_date=reference_date,
                disaster_type=disaster_type,
                forecast_days=forecast_days
            )
            logger.info(f"[WEATHERWISE_CONTROLLER] Service call completed")
            logger.info(f"[WEATHERWISE_CONTROLLER] Forecast result success: {forecast_result.get('success')}")
            
            
            if forecast_result['success']:
                self.controller_stats['successful_requests'] += 1
                
                response_data = {
                    'forecast': forecast_result['weather_forecast'],
                    'forecast_dates': forecast_result['forecast_dates'],
                    'forecast_variables': forecast_result['forecast_variables'],
                    'model_context': forecast_result['model_type'],
                    'location': forecast_result['location'],
                    'forecast_summary': {
                        'horizon_days': forecast_result['forecast_horizon_days'],
                        'variables_count': len(forecast_result['forecast_variables']),
                        'model_used': forecast_result['model_type']
                    },
                    'data_collection_summary': forecast_result.get('data_collection', {})
                }
                
                return self._create_response(
                    success=True,
                    message="Weather forecast generated successfully",
                    data=response_data,
                    processing_info={
                        'processing_time_seconds': forecast_result.get('processing_time_seconds', 0),
                        'forecast_model': forecast_result['model_type'],
                        'forecast_horizon_days': forecast_result['forecast_horizon_days'],
                        'data_sources': forecast_result.get('processing_info', {}).get('data_sources', [])
                    }
                )
            else:
                self.controller_stats['failed_requests'] += 1
                return self._create_response(
                    success=False,
                    message="Weather forecast generation failed",
                    error=forecast_result.get('error', 'Unknown forecast error'),
                    data={
                        'location': forecast_result.get('location'),
                        'processing_time_seconds': forecast_result.get('processing_time_seconds', 0)
                    }
                )
                
        except Exception as e:
            self.controller_stats['failed_requests'] += 1
            logger.error(f"WeatherWise controller forecast error: {e}")
            return self._create_response(
                success=False,
                message="Weather forecast error",
                error=f"Controller error: {str(e)}"
            )
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get WeatherWise service status and health information
        
        Returns:
            Service status response
        """
        try:
            # Get service health from the service layer
            service_health = self.service.get_service_health()
            
            # Add controller statistics
            response_data = {
                'controller_info': {
                    'controller_name': 'WeatherWise Prediction Controller',
                    'controller_stats': self.controller_stats
                },
                'service_health': service_health
            }
            
            return self._create_response(
                success=True,
                message="WeatherWise service status retrieved successfully",
                data=response_data
            )
            
        except Exception as e:
            logger.error(f"WeatherWise status error: {e}")
            return self._create_response(
                success=False,
                message="Failed to retrieve service status",
                error=f"Status error: {str(e)}"
            )
    
    def get_available_models(self) -> Dict[str, Any]:
        """
        Get available disaster context models
        
        Returns:
            Available models response
        """
        try:
            available_models = self.service.get_available_models()
            model_info = self.service.prediction_model.get_model_info()
            
            return self._create_response(
                success=True,
                message="Available models retrieved successfully",
                data={
                    'available_disaster_contexts': available_models,
                    'model_info': model_info,
                    'default_context': 'Normal',
                    'supported_forecast_variables': model_info.get('forecast_variables', [])
                }
            )
            
        except Exception as e:
            logger.error(f"WeatherWise models list error: {e}")
            return self._create_response(
                success=False,
                message="Failed to retrieve available models",
                error=f"Models error: {str(e)}"
            )
    
    def _create_response(self, success: bool, message: str, 
                       data: Dict[str, Any] = None, error: str = None, 
                       processing_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create standardized response dictionary
        
        Args:
            success: Success status
            message: Response message
            data: Response data (optional)
            error: Error message (optional)
            processing_info: Processing information (optional)
        
        Returns:
            Standardized response dictionary
        """
        response = {
            'success': success,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'service': 'WeatherWise'
        }
        
        if data is not None:
            response['data'] = data
            
        if error is not None:
            response['error'] = error
            
        if processing_info is not None:
            response['processing_info'] = processing_info
            
        return response