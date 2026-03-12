"""
HazardGuard Disaster Prediction Controller
API request coordination and response formatting for disaster predictions
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from services.hazardguard_prediction_service import HazardGuardPredictionService

logger = logging.getLogger(__name__)

class HazardGuardPredictionController:
    """Controller for HazardGuard disaster prediction API operations"""
    
    def __init__(self, service: Optional[HazardGuardPredictionService] = None):
        """Initialize the HazardGuard prediction controller"""
        self.service = service or HazardGuardPredictionService()
        
        # Standard response templates
        self.success_template = {
            'success': True,
            'message': 'Operation completed successfully',
            'data': {},
            'timestamp': None,
            'processing_info': {}
        }
        
        self.error_template = {
            'success': False,
            'error': 'Unknown error',
            'message': 'Operation failed',
            'data': None,
            'timestamp': None
        }
        
        logger.info("HazardGuard prediction controller initialized")
    
    def initialize_controller(self) -> Dict[str, Any]:
        """
        Initialize the controller by setting up the service
        
        Returns:
            Initialization response
        """
        try:
            success, message = self.service.initialize_service()
            
            if success:
                return self._create_response(
                    success=True,
                    message="HazardGuard controller initialized successfully",
                    data={
                        'service_status': 'ready',
                        'initialization_message': message
                    }
                )
            else:
                return self._create_response(
                    success=False,
                    message="Controller initialization failed",
                    error=message
                )
                
        except Exception as e:
            logger.error(f"Controller initialization error: {e}")
            return self._create_response(
                success=False,
                message="Controller initialization error",
                error=f"Controller error: {str(e)}"
            )
    
    def _create_response(self, success: bool = True, message: str = '', 
                        data: Any = None, error: str = '', 
                        processing_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Create standardized API response
        
        Args:
            success: Whether the operation was successful
            message: Success or error message
            data: Response data
            error: Error message (for failed operations)
            processing_info: Additional processing information
        
        Returns:
            Standardized response dictionary
        """
        if success:
            response = self.success_template.copy()
            response['message'] = message or 'Operation completed successfully'
            response['data'] = data
            response['processing_info'] = processing_info or {}
        else:
            response = self.error_template.copy()
            response['error'] = error or 'Unknown error'
            response['message'] = message or 'Operation failed'
            response['data'] = data
        
        response['timestamp'] = datetime.now().isoformat()
        return response
    
    def validate_prediction_request(self, request_data: Dict[str, Any]) -> Tuple[bool, str, Optional[Tuple[float, float, Optional[str]]]]:
        """
        Validate prediction request data
        
        Args:
            request_data: Request dictionary containing location data
        
        Returns:
            Tuple of (is_valid, message, (latitude, longitude, reference_date))
        """
        try:
            # Check for required fields
            if 'latitude' not in request_data:
                return False, "Missing required field: 'latitude'", None
            
            if 'longitude' not in request_data:
                return False, "Missing required field: 'longitude'", None
            
            # Extract and validate coordinates
            try:
                latitude = float(request_data['latitude'])
                longitude = float(request_data['longitude'])
            except (ValueError, TypeError):
                return False, "Latitude and longitude must be numeric values", None
            
            # Validate coordinate ranges
            if not (-90 <= latitude <= 90):
                return False, f"Invalid latitude {latitude} (must be -90 to 90)", None
            
            if not (-180 <= longitude <= 180):
                return False, f"Invalid longitude {longitude} (must be -180 to 180)", None
            
            # Optional reference date validation
            reference_date = request_data.get('reference_date')
            if reference_date:
                try:
                    # Validate date format
                    datetime.strptime(reference_date, '%Y-%m-%d')
                except ValueError:
                    return False, "Invalid reference_date format. Use YYYY-MM-DD.", None
            
            return True, f"Request validation successful: ({latitude}, {longitude})", (latitude, longitude, reference_date)
            
        except Exception as e:
            logger.error(f"Request validation error: {e}")
            return False, f"Validation error: {str(e)}", None
    
    def predict_disaster_risk(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Primary endpoint: Predict disaster risk for a location
        
        Args:
            request_data: Request dictionary with latitude, longitude, and optional reference_date
        
        Returns:
            Disaster prediction response
        """
        try:
            # Validate request
            is_valid, validation_message, parsed_data = self.validate_prediction_request(request_data)
            
            if not is_valid:
                return self._create_response(
                    success=False,
                    message="Request validation failed",
                    error=validation_message,
                    data={
                        'required_fields': ['latitude', 'longitude'],
                        'optional_fields': ['reference_date (YYYY-MM-DD)'],
                        'coordinate_ranges': 'latitude: -90 to 90, longitude: -180 to 180'
                    }
                )
            
            latitude, longitude, reference_date = parsed_data
            
            logger.info(f"Processing disaster prediction for ({latitude}, {longitude})")
            
            # Make prediction using service
            prediction_result = self.service.predict_disaster_for_location(
                latitude=latitude,
                longitude=longitude,
                reference_date=reference_date
            )
            
            if prediction_result['success']:
                response_data = {
                    'location': prediction_result['location'],
                    'prediction': prediction_result['prediction'],
                    'data_collection_summary': {
                        'weather_data': prediction_result['data_collection']['weather']['success'],
                        'feature_engineering': prediction_result['data_collection']['features']['success'],
                        'raster_data': prediction_result['data_collection']['raster']['success']
                    },
                    'processing_details': prediction_result['processing_info']
                }
                
                # Add disaster types if available
                if prediction_result.get('disaster_types'):
                    response_data['disaster_types'] = prediction_result['disaster_types']
                
                return self._create_response(
                    success=True,
                    message="Disaster prediction completed successfully",
                    data=response_data,
                    processing_info={
                        'total_processing_time_seconds': prediction_result['processing_info']['total_processing_time_seconds'],
                        'prediction_class': prediction_result['prediction']['prediction'],
                        'disaster_probability': prediction_result['prediction']['probability']['disaster'],
                        'confidence': prediction_result['prediction']['confidence']
                    }
                )
            else:
                return self._create_response(
                    success=False,
                    message="Disaster prediction failed",
                    error=prediction_result.get('error', 'Unknown prediction error'),
                    data={
                        'location': prediction_result.get('location'),
                        'data_collection': prediction_result.get('data_collection'),
                        'processing_time_seconds': prediction_result.get('processing_time_seconds', 0)
                    }
                )
            
        except Exception as e:
            logger.error(f"Controller prediction error: {e}")
            return self._create_response(
                success=False,
                message="Disaster prediction error",
                error=f"Controller error: {str(e)}"
            )
    
    def predict_batch_locations(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Batch prediction endpoint: Predict disaster risk for multiple locations
        
        Args:
            request_data: Request dictionary with 'locations' array
        
        Returns:
            Batch prediction response
        """
        try:
            # Validate batch request
            locations = request_data.get('locations', [])
            
            if not locations or not isinstance(locations, list):
                return self._create_response(
                    success=False,
                    message="Batch prediction validation failed",
                    error="'locations' field must be a non-empty array",
                    data={
                        'required_format': {
                            'locations': [
                                {'latitude': float, 'longitude': float, 'reference_date': 'YYYY-MM-DD (optional)'},
                                {'latitude': float, 'longitude': float}
                            ]
                        }
                    }
                )
            
            if len(locations) > 50:  # Limit batch size
                return self._create_response(
                    success=False,
                    message="Batch size limit exceeded",
                    error="Maximum 50 locations per batch request"
                )
            
            logger.info(f"Processing batch prediction for {len(locations)} locations")
            
            results = []
            successful_predictions = 0
            failed_predictions = 0
            
            for i, location_data in enumerate(locations):
                try:
                    # Validate individual location
                    is_valid, validation_message, parsed_data = self.validate_prediction_request(location_data)
                    
                    if not is_valid:
                        results.append({
                            'location_index': i + 1,
                            'success': False,
                            'error': validation_message,
                            'location_data': location_data
                        })
                        failed_predictions += 1
                        continue
                    
                    latitude, longitude, reference_date = parsed_data
                    
                    # Make prediction
                    prediction_result = self.service.predict_disaster_for_location(
                        latitude=latitude,
                        longitude=longitude,
                        reference_date=reference_date
                    )
                    
                    if prediction_result['success']:
                        batch_entry = {
                            'location_index': i + 1,
                            'success': True,
                            'location': prediction_result['location'],
                            'prediction': prediction_result['prediction'],
                            'processing_time_seconds': prediction_result['processing_info']['total_processing_time_seconds']
                        }
                        # Include disaster type classification if available
                        if prediction_result.get('disaster_types'):
                            batch_entry['disaster_types'] = prediction_result['disaster_types']
                        results.append(batch_entry)
                        successful_predictions += 1
                    else:
                        results.append({
                            'location_index': i + 1,
                            'success': False,
                            'error': prediction_result.get('error', 'Prediction failed'),
                            'location': prediction_result.get('location'),
                            'processing_time_seconds': prediction_result.get('processing_time_seconds', 0)
                        })
                        failed_predictions += 1
                        
                except Exception as e:
                    results.append({
                        'location_index': i + 1,
                        'success': False,
                        'error': f"Location processing error: {str(e)}",
                        'location_data': location_data
                    })
                    failed_predictions += 1
            
            # Calculate success rate
            total_locations = len(locations)
            success_rate = (successful_predictions / total_locations * 100) if total_locations > 0 else 0
            
            return self._create_response(
                success=successful_predictions > 0,
                message=f"Batch prediction completed: {successful_predictions}/{total_locations} successful",
                data={
                    'results': results,
                    'summary': {
                        'total_locations': total_locations,
                        'successful_predictions': successful_predictions,
                        'failed_predictions': failed_predictions,
                        'success_rate_percent': success_rate
                    }
                },
                processing_info={
                    'batch_size': total_locations,
                    'processing_mode': 'sequential'
                }
            )
            
        except Exception as e:
            logger.error(f"Controller batch prediction error: {e}")
            return self._create_response(
                success=False,
                message="Batch prediction error",
                error=f"Controller error: {str(e)}"
            )
    
    def get_prediction_capabilities(self) -> Dict[str, Any]:
        """
        Get information about HazardGuard prediction capabilities
        
        Returns:
            Capabilities information response
        """
        try:
            # Get service status to include model info
            service_status = self.service.get_service_status()
            
            capabilities = {
                'prediction_type': 'Binary Classification (DISASTER vs NORMAL)',
                'supported_disaster_types': ['Flood', 'Storm', 'Landslide', 'Drought'],
                'forecasting_horizon': '1 day ahead',
                'geographic_coverage': 'Global (latitude: -90 to 90, longitude: -180 to 180)',
                'data_sources': {
                    'weather_data': 'NASA POWER API (17 variables, 60-day sequences)',
                    'engineered_features': 'Weather-derived features (19 variables)',
                    'raster_data': 'Geographic/Environmental data (9 variables)',
                    'total_features': '~300 features after statistical expansion'
                },
                'model_details': {
                    'algorithm': 'XGBoost Binary Classifier',
                    'feature_selection': 'SelectKBest with f_classif',
                    'preprocessing': 'StandardScaler normalization',
                    'validation': '5-fold GroupKFold cross-validation'
                },
                'input_requirements': {
                    'required_fields': ['latitude', 'longitude'],
                    'optional_fields': ['reference_date (YYYY-MM-DD)'],
                    'coordinate_ranges': {
                        'latitude': {'min': -90, 'max': 90},
                        'longitude': {'min': -180, 'max': 180}
                    }
                },
                'output_format': {
                    'prediction': 'DISASTER or NORMAL',
                    'probabilities': {
                        'disaster': 'float (0.0 to 1.0)',
                        'normal': 'float (0.0 to 1.0)'
                    },
                    'confidence': 'float (difference between class probabilities)',
                    'processing_metadata': 'timing, feature counts, etc.'
                },
                'batch_processing': {
                    'supported': True,
                    'max_locations_per_request': 50
                },
                'service_status': service_status
            }
            
            return self._create_response(
                success=True,
                message="HazardGuard capabilities retrieved successfully",
                data=capabilities
            )
            
        except Exception as e:
            logger.error(f"Controller capabilities error: {e}")
            return self._create_response(
                success=False,
                message="Capabilities retrieval error",
                error=f"Controller error: {str(e)}"
            )
    
    def get_service_health(self) -> Dict[str, Any]:
        """
        Get HazardGuard service health and performance statistics
        
        Returns:
            Service health response
        """
        try:
            service_status = self.service.get_service_status()
            
            if service_status.get('service_status') in ['ready', 'healthy']:
                return self._create_response(
                    success=True,
                    message="HazardGuard service is healthy",
                    data=service_status
                )
            else:
                return self._create_response(
                    success=False,
                    message="HazardGuard service health issues detected",
                    error=service_status.get('error', 'Service not ready'),
                    data=service_status
                )
            
        except Exception as e:
            logger.error(f"Controller health check error: {e}")
            return self._create_response(
                success=False,
                message="Health check error",
                error=f"Controller error: {str(e)}"
            )
    
    def reset_service_statistics(self) -> Dict[str, Any]:
        """
        Reset HazardGuard service statistics
        
        Returns:
            Statistics reset response
        """
        try:
            reset_result = self.service.reset_statistics()
            
            if reset_result['status'] == 'success':
                return self._create_response(
                    success=True,
                    message="HazardGuard statistics reset successfully",
                    data=reset_result
                )
            else:
                return self._create_response(
                    success=False,
                    message="Statistics reset failed",
                    error=reset_result['message']
                )
            
        except Exception as e:
            logger.error(f"Controller statistics reset error: {e}")
            return self._create_response(
                success=False,
                message="Statistics reset error",
                error=f"Controller error: {str(e)}"
            )
    
    def validate_coordinates_only(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate coordinates without making prediction (for testing/validation)
        
        Args:
            request_data: Request dictionary containing coordinates
        
        Returns:
            Coordinate validation response
        """
        try:
            is_valid, validation_message, parsed_data = self.validate_prediction_request(request_data)
            
            if is_valid:
                latitude, longitude, reference_date = parsed_data
                
                return self._create_response(
                    success=True,
                    message="Coordinate validation successful",
                    data={
                        'coordinates': {
                            'latitude': latitude,
                            'longitude': longitude,
                            'reference_date': reference_date
                        },
                        'validation_message': validation_message
                    }
                )
            else:
                return self._create_response(
                    success=False,
                    message="Coordinate validation failed",
                    error=validation_message,
                    data={
                        'required_format': {
                            'latitude': 'float (-90 to 90)',
                            'longitude': 'float (-180 to 180)',
                            'reference_date': 'string (YYYY-MM-DD, optional)'
                        }
                    }
                )
            
        except Exception as e:
            logger.error(f"Controller coordinate validation error: {e}")
            return self._create_response(
                success=False,
                message="Coordinate validation error",
                error=f"Controller error: {str(e)}"
            )