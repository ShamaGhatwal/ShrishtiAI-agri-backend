"""
Post-Disaster Weather Data Controller for HazardGuard System  
API request coordination and response formatting for post-disaster weather operations
"""

import logging
from typing import Dict, List, Optional, Any, Union
from flask import request, jsonify
import pandas as pd
from datetime import datetime

from services.post_disaster_weather_service import PostDisasterWeatherService

logger = logging.getLogger(__name__)

class PostDisasterWeatherController:
    """Controller layer for post-disaster weather data API operations"""
    
    def __init__(self, 
                 days_after_disaster: int = 60,
                 max_workers: int = 1,
                 retry_limit: int = 5,
                 retry_delay: int = 15,
                 rate_limit_pause: int = 900,
                 request_delay: float = 0.5):
        """Initialize post-disaster weather controller"""
        try:
            self.service = PostDisasterWeatherService(
                days_after_disaster=days_after_disaster,
                max_workers=max_workers,
                retry_limit=retry_limit,
                retry_delay=retry_delay,
                rate_limit_pause=rate_limit_pause,
                request_delay=request_delay
            )
            self.request_count = 0
            logger.info("PostDisasterWeatherController initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostDisasterWeatherController: {e}")
            raise
    
    def _success_response(self, data: Any, message: str = "Success", metadata: Optional[Dict] = None, status_code: int = 200) -> Dict[str, Any]:
        """Create standardized success response"""
        response = {
            'success': True,
            'data': data,
            'message': message,
            'status_code': status_code,
            'metadata': metadata or {}
        }
        # Add request tracking metadata
        response['metadata']['request_count'] = self.request_count
        response['metadata']['timestamp'] = datetime.now().isoformat()
        
        return response
    
    def _error_response(self, error: str, status_code: int = 400, details: Optional[Dict] = None) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            'success': False,
            'error': error,
            'data': None,
            'status_code': status_code,
            'metadata': {
                'request_count': self.request_count,
                'timestamp': datetime.now().isoformat(),
                'details': details or {}
            }
        }
    
    def process_post_disaster_weather(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle post-disaster weather extraction API request"""
        try:
            self.request_count += 1
            logger.info(f"Processing post-disaster weather request #{self.request_count}")
            
            # Validate request structure
            if not isinstance(request_data, dict):
                return self._error_response("Request must be a JSON object", 400)
            
            # Extract and validate required fields
            coordinates = request_data.get('coordinates', [])
            if not coordinates:
                return self._error_response("'coordinates' field is required and must be non-empty", 400)
            
            disaster_dates = request_data.get('disaster_dates', [])
            if not disaster_dates:
                return self._error_response("'disaster_dates' field is required and must be non-empty", 400)
            
            # Extract optional fields
            variables = request_data.get('variables')
            if variables and not isinstance(variables, list):
                return self._error_response("'variables' must be a list of variable names", 400)
            
            # Validate coordinates format
            is_valid, validation_message = self.service.validate_coordinates(coordinates)
            if not is_valid:
                return self._error_response(f"Invalid coordinates: {validation_message}", 400)
            
            # Process weather extraction
            result = self.service.process_post_disaster_weather(coordinates, disaster_dates, variables)
            
            if result['success']:
                return self._success_response(
                    data=result['data'],
                    message=result['message'],
                    metadata=result['metadata']
                )
            else:
                return self._error_response(result['error'], 500)
                
        except Exception as e:
            logger.error(f"Controller error processing weather request: {e}")
            return self._error_response(f"Processing error: {str(e)}", 500)
    
    def process_batch_weather(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle batch post-disaster weather processing"""
        try:
            self.request_count += 1
            logger.info(f"Processing batch post-disaster weather request #{self.request_count}")
            
            # Validate request structure
            if not isinstance(request_data, dict):
                return self._error_response("Request must be a JSON object", 400)
            
            # Validate batch request
            is_valid, validation_message = self.service.validate_batch_request(request_data)
            if not is_valid:
                return self._error_response(f"Invalid batch request: {validation_message}", 400)
            
            # Extract fields
            coordinates = request_data['coordinates']
            disaster_dates = request_data['disaster_dates']
            variables = request_data.get('variables')
            
            # Process batch
            result = self.service.process_post_disaster_weather(coordinates, disaster_dates, variables)
            
            if result['success']:
                return self._success_response(
                    data=result['data'],
                    message=f"Batch processing completed: {result['message']}",
                    metadata={
                        **result['metadata'],
                        'batch_size': len(coordinates),
                        'processing_type': 'batch'
                    }
                )
            else:
                return self._error_response(result['error'], 500)
                
        except Exception as e:
            logger.error(f"Controller error processing batch request: {e}")
            return self._error_response(f"Batch processing error: {str(e)}", 500)
    
    def validate_coordinates(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate coordinate format and ranges"""
        try:
            self.request_count += 1
            
            coordinates = request_data.get('coordinates', [])
            if not coordinates:
                return self._error_response("'coordinates' field is required", 400)
            
            is_valid, message = self.service.validate_coordinates(coordinates)
            
            return self._success_response(
                data={
                    'valid': is_valid,
                    'message': message,
                    'coordinates_count': len(coordinates)
                },
                message="Coordinate validation completed"
            )
            
        except Exception as e:
            logger.error(f"Coordinate validation error: {e}")
            return self._error_response(f"Validation error: {str(e)}", 500)
    
    def validate_disaster_dates(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate disaster date format and ranges"""
        try:
            self.request_count += 1
            
            disaster_dates = request_data.get('disaster_dates', [])
            if not disaster_dates:
                return self._error_response("'disaster_dates' field is required", 400)
            
            is_valid, message, parsed_dates = self.service.validate_disaster_dates(disaster_dates)
            
            validation_data = {
                'valid': is_valid,
                'message': message,
                'dates_count': len(disaster_dates)
            }
            
            if is_valid:
                validation_data['parsed_dates'] = [d.strftime('%Y-%m-%d') for d in parsed_dates]
                validation_data['date_range'] = {
                    'earliest': min(parsed_dates).strftime('%Y-%m-%d'),
                    'latest': max(parsed_dates).strftime('%Y-%m-%d')
                }
            
            return self._success_response(
                data=validation_data,
                message="Date validation completed"
            )
            
        except Exception as e:
            logger.error(f"Date validation error: {e}")
            return self._error_response(f"Date validation error: {str(e)}", 500)
    
    def get_available_variables(self) -> Dict[str, Any]:
        """Get available post-disaster weather variables"""
        try:
            self.request_count += 1
            result = self.service.get_available_variables()
            
            if result['success']:
                return self._success_response(
                    data=result['variables'],
                    message=result['message'],
                    metadata={
                        'total_variables': result['total_variables'],
                        'days_per_variable': result['days_per_variable']
                    }
                )
            else:
                return self._error_response(result['error'], 500)
                
        except Exception as e:
            logger.error(f"Error getting available variables: {e}")
            return self._error_response(f"Failed to get variables: {str(e)}", 500)
    
    def export_to_dataframe(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Export weather data to DataFrame format"""
        try:
            self.request_count += 1
            
            weather_data = request_data.get('weather_data', [])
            if not weather_data:
                return self._error_response("'weather_data' field is required", 400)
            
            result = self.service.export_to_dataframe(weather_data)
            
            if result['success']:
                df = result['dataframe']
                
                return self._success_response(
                    data={
                        'dataframe_info': {
                            'shape': result['shape'],
                            'columns': result['columns'],
                            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
                            'dtypes': df.dtypes.astype(str).to_dict()
                        }
                    },
                    message=result['message']
                )
            else:
                return self._error_response(result['error'], 500)
                
        except Exception as e:
            logger.error(f"DataFrame export error: {e}")
            return self._error_response(f"Export error: {str(e)}", 500)
    
    def export_to_file(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Export weather data to file"""
        try:
            self.request_count += 1
            
            # Validate required fields
            weather_data = request_data.get('weather_data', [])
            if not weather_data:
                return self._error_response("'weather_data' field is required", 400)
            
            filepath = request_data.get('filepath')
            if not filepath:
                return self._error_response("'filepath' field is required", 400)
            
            file_format = request_data.get('file_format', 'json')
            
            result = self.service.export_to_file(weather_data, filepath, file_format)
            
            if result['success']:
                return self._success_response(
                    data={
                        'filepath': result['filepath'],
                        'file_format': result['file_format'],
                        'file_size_mb': round(result['file_size_bytes'] / 1024 / 1024, 2),
                        'coordinates_exported': result['coordinates_exported']
                    },
                    message=result['message']
                )
            else:
                return self._error_response(result['error'], 500)
                
        except Exception as e:
            logger.error(f"File export error: {e}")
            return self._error_response(f"Export error: {str(e)}", 500)
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get service processing statistics"""
        try:
            self.request_count += 1
            result = self.service.get_processing_statistics()
            
            if result['success']:
                return self._success_response(
                    data=result['statistics'],
                    message="Successfully retrieved processing statistics"
                )
            else:
                return self._error_response(result['error'], 500)
                
        except Exception as e:
            logger.error(f"Statistics error: {e}")
            return self._error_response(f"Failed to get statistics: {str(e)}", 500)
    
    def get_service_health(self) -> Dict[str, Any]:
        """Get service health status"""
        try:
            self.request_count += 1
            result = self.service.get_service_status()
            
            return self._success_response(
                data=result,
                message=result.get('message', 'Service status retrieved')
            )
            
        except Exception as e:
            logger.error(f"Service health error: {e}")
            return self._error_response(f"Health check failed: {str(e)}", 500)
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get comprehensive service information"""
        try:
            self.request_count += 1
            
            # Get variables info
            variables_result = self.service.get_available_variables()
            
            # Get service status
            status_result = self.service.get_service_status()
            
            service_info = {
                'service_name': 'Post-Disaster Weather Data Service',
                'description': 'Fetches weather data for 60 days after disaster occurrence',
                'version': '1.0.0',
                'api_source': 'NASA POWER',
                'data_type': 'post_disaster_weather',
                'days_after_disaster': self.service.model.days_after_disaster,
                'total_variables': len(self.service.model.WEATHER_FIELDS),
                'variable_categories': {
                    'temperature': ['POST_temperature_C', 'POST_temperature_max_C', 'POST_temperature_min_C', 'POST_dew_point_C'],
                    'humidity': ['POST_humidity_%', 'POST_specific_humidity_g_kg'],
                    'wind': ['POST_wind_speed_mps', 'POST_wind_speed_10m_mps', 'POST_wind_direction_10m_degrees'],
                    'precipitation': ['POST_precipitation_mm'],
                    'pressure': ['POST_surface_pressure_hPa', 'POST_sea_level_pressure_hPa'],
                    'radiation': ['POST_solar_radiation_wm2', 'POST_evapotranspiration_wm2'],
                    'cloud': ['POST_cloud_amount_%'],
                    'soil': ['POST_surface_soil_wetness_%', 'POST_root_zone_soil_moisture_%']
                },
                'features': {
                    'time_series_data': True,
                    'statistical_summaries': True,
                    'missing_value_handling': True,
                    'batch_processing': True,
                    'multiple_export_formats': True
                },
                'status': status_result
            }
            
            if variables_result['success']:
                service_info['variables'] = variables_result['variables']
            
            return self._success_response(
                data=service_info,
                message="Service information retrieved successfully"
            )
            
        except Exception as e:
            logger.error(f"Service info error: {e}")
            return self._error_response(f"Failed to get service info: {str(e)}", 500)
    
    def test_api_connection(self) -> Dict[str, Any]:
        """Test NASA POWER API connectivity"""
        try:
            self.request_count += 1
            
            # Test with a simple coordinate
            test_coordinates = [{'latitude': 0.0, 'longitude': 0.0}]
            test_dates = [datetime(2023, 1, 1)]  # Use a safe past date
            
            logger.info("Testing NASA POWER API connectivity...")
            
            result = self.service.process_post_disaster_weather(
                coordinates=test_coordinates,
                disaster_dates=test_dates,
                variables=['POST_temperature_C']  # Test with just one variable
            )
            
            api_test_result = {
                'api_accessible': result['success'],
                'test_coordinate': test_coordinates[0],
                'test_date': test_dates[0].strftime('%Y-%m-%d'),
                'response_time_seconds': result.get('metadata', {}).get('processing_time_seconds', 0),
                'nasa_api_status': 'operational' if result['success'] else 'error'
            }
            
            if result['success']:
                api_test_result['data_quality'] = {
                    'variables_returned': len([k for k in result['data'][0].keys() if k.startswith('POST_')]),
                    'time_series_length': result['data'][0].get('days_fetched', 0) if result['data'] else 0
                }
            else:
                api_test_result['error_details'] = result['error']
            
            return self._success_response(
                data=api_test_result,
                message="API connectivity test completed"
            )
            
        except Exception as e:
            logger.error(f"API test error: {e}")
            return self._error_response(f"API test failed: {str(e)}", 500)