"""
Post-Disaster Feature Engineering Controller for HazardGuard System
API request coordination and response formatting
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from services.post_disaster_feature_engineering_service import PostDisasterFeatureEngineeringService

logger = logging.getLogger(__name__)

class PostDisasterFeatureEngineeringController:
    """Controller for post-disaster feature engineering API operations"""
    
    def __init__(self):
        """Initialize the post-disaster feature engineering controller"""
        self.service = PostDisasterFeatureEngineeringService()
        
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
        
        logger.info("PostDisasterFeatureEngineeringController initialized")
    
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
    
    def validate_coordinates(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate coordinates from request data
        
        Args:
            request_data: Request dictionary containing 'coordinates' key
        
        Returns:
            Validation response
        """
        try:
            coordinates = request_data.get('coordinates')
            
            if not coordinates:
                return self._create_response(
                    success=False,
                    message="Coordinates validation failed",
                    error="No coordinates provided in request",
                    data={'required_format': '[[lat1, lon1], [lat2, lon2], ...]'}
                )
            
            # Use service validation
            is_valid, validation_message, parsed_coordinates = self.service.validate_coordinates(coordinates)
            
            if not is_valid:
                return self._create_response(
                    success=False,
                    message="Coordinates validation failed",
                    error=validation_message,
                    data={'required_format': '[[lat1, lon1], [lat2, lon2], ...]'}
                )
            
            return self._create_response(
                success=True,
                message="Coordinates validation successful",
                data={
                    'coordinates': parsed_coordinates,
                    'count': len(parsed_coordinates),
                    'validation_message': validation_message
                },
                processing_info={
                    'coordinates_count': len(parsed_coordinates)
                }
            )
            
        except Exception as e:
            logger.error(f"Controller coordinates validation error: {e}")
            return self._create_response(
                success=False,
                message="Coordinates validation error",
                error=f"Controller error: {str(e)}"
            )
    
    def process_single_coordinate_features(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process post-disaster feature engineering for a single coordinate
        
        Args:
            request_data: Request dictionary containing 'weather_data' and optionally 'coordinate'
        
        Returns:
            Feature engineering response
        """
        try:
            # Extract request data
            weather_data = request_data.get('weather_data')
            coordinate = request_data.get('coordinate')
            global_stats = request_data.get('global_stats')
            
            if not weather_data:
                return self._create_response(
                    success=False,
                    message="Weather data required",
                    error="No weather_data provided in request",
                    data={'required_variables': self.service.get_input_variables()}
                )
            
            # Process using service
            result = self.service.process_single_coordinate_features(
                weather_data=weather_data,
                coordinate=coordinate,
                global_stats=global_stats
            )
            
            if result['success']:
                return self._create_response(
                    success=True,
                    message="Feature engineering completed successfully",
                    data={
                        'coordinate': result['coordinate'],
                        'features': result['features'],
                        'metadata': result['metadata']
                    },
                    processing_info={
                        'processing_time_seconds': result['processing_time_seconds'],
                        'features_count': len(result['features']) if result['features'] else 0,
                        'days_processed': len(next(iter(result['features'].values()), [])) if result['features'] else 0
                    }
                )
            else:
                return self._create_response(
                    success=False,
                    message="Feature engineering failed",
                    error=result['error'],
                    data={'coordinate': result['coordinate']},
                    processing_info={
                        'processing_time_seconds': result['processing_time_seconds']
                    }
                )
            
        except Exception as e:
            logger.error(f"Controller single coordinate processing error: {e}")
            return self._create_response(
                success=False,
                message="Single coordinate feature engineering error",
                error=f"Controller error: {str(e)}"
            )
    
    def process_batch_features(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process post-disaster feature engineering for multiple coordinates
        
        Args:
            request_data: Request dictionary containing 'weather_datasets' and optionally 'coordinates'
        
        Returns:
            Batch feature engineering response
        """
        try:
            # Extract request data
            weather_datasets = request_data.get('weather_datasets')
            coordinates = request_data.get('coordinates')
            
            if not weather_datasets:
                return self._create_response(
                    success=False,
                    message="Weather datasets required",
                    error="No weather_datasets provided in request",
                    data={'required_format': 'List of weather data dictionaries'}
                )
            
            if not isinstance(weather_datasets, list):
                return self._create_response(
                    success=False,
                    message="Invalid weather datasets format",
                    error="weather_datasets must be a list",
                    data={'required_format': 'List of weather data dictionaries'}
                )
            
            # Process using service
            result = self.service.process_batch_features(
                weather_datasets=weather_datasets,
                coordinates=coordinates
            )
            
            if result['success']:
                return self._create_response(
                    success=True,
                    message=f"Batch feature engineering completed: {result['successful_coordinates']}/{result['total_coordinates']} coordinates",
                    data={
                        'results': result['results'],
                        'global_statistics': result['global_statistics'],
                        'summary': {
                            'total_coordinates': result['total_coordinates'],
                            'successful_coordinates': result['successful_coordinates'],
                            'failed_coordinates': result['failed_coordinates'],
                            'success_rate_percent': result['success_rate_percent']
                        }
                    },
                    processing_info={
                        'processing_time_seconds': result['processing_time_seconds'],
                        'coordinates_count': result['total_coordinates']
                    }
                )
            else:
                return self._create_response(
                    success=False,
                    message="Batch feature engineering failed",
                    error=result['error'],
                    processing_info={
                        'processing_time_seconds': result['processing_time_seconds']
                    }
                )
            
        except Exception as e:
            logger.error(f"Controller batch processing error: {e}")
            return self._create_response(
                success=False,
                message="Batch feature engineering error",
                error=f"Controller error: {str(e)}"
            )
    
    def export_to_csv(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Export feature engineering results to CSV format
        
        Args:
            request_data: Request dictionary containing 'results' and optionally 'include_metadata'
        
        Returns:
            CSV export response
        """
        try:
            results = request_data.get('results')
            include_metadata = request_data.get('include_metadata', True)
            
            if not results:
                return self._create_response(
                    success=False,
                    message="Results required for CSV export",
                    error="No results provided in request"
                )
            
            # Export to DataFrame
            df = self.service.export_to_dataframe(results, include_metadata)
            
            if df is None:
                return self._create_response(
                    success=False,
                    message="CSV export failed",
                    error="Failed to create DataFrame from results"
                )
            
            # Convert to CSV string
            csv_string = df.to_csv(index=False)
            
            return self._create_response(
                success=True,
                message=f"CSV export completed: {len(df)} rows, {len(df.columns)} columns",
                data={
                    'csv_data': csv_string,
                    'row_count': len(df),
                    'column_count': len(df.columns),
                    'columns': df.columns.tolist()
                },
                processing_info={
                    'export_format': 'CSV',
                    'include_metadata': include_metadata
                }
            )
            
        except Exception as e:
            logger.error(f"Controller CSV export error: {e}")
            return self._create_response(
                success=False,
                message="CSV export error",
                error=f"Controller error: {str(e)}"
            )
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about input variables and output features
        
        Returns:
            Feature information response
        """
        try:
            feature_descriptions = self.service.get_feature_descriptions()
            input_variables = self.service.get_input_variables()
            output_variables = self.service.get_output_variables()
            
            return self._create_response(
                success=True,
                message="Feature information retrieved successfully",
                data={
                    'input_variables': {
                        'count': len(input_variables),
                        'variables': input_variables,
                        'description': 'Required weather variables for feature engineering'
                    },
                    'output_features': {
                        'count': len(output_variables),
                        'features': output_variables,
                        'descriptions': feature_descriptions,
                        'description': 'Engineered features created from weather data'
                    },
                    'processing_info': {
                        'days_per_coordinate': 60,
                        'feature_engineering_type': 'Post-disaster weather analysis'
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Controller feature info error: {e}")
            return self._create_response(
                success=False,
                message="Feature information error",
                error=f"Controller error: {str(e)}"
            )
    
    def get_service_health(self) -> Dict[str, Any]:
        """
        Get service health and performance statistics
        
        Returns:
            Service health response
        """
        try:
            health_info = self.service.get_service_health()
            
            if health_info.get('service_status') == 'healthy':
                return self._create_response(
                    success=True,
                    message="Service is healthy",
                    data=health_info
                )
            else:
                return self._create_response(
                    success=False,
                    message="Service health check failed",
                    error=health_info.get('error', 'Unknown health issue'),
                    data=health_info
                )
            
        except Exception as e:
            logger.error(f"Controller health check error: {e}")
            return self._create_response(
                success=False,
                message="Health check error",
                error=f"Controller error: {str(e)}"
            )
    
    def reset_statistics(self) -> Dict[str, Any]:
        """
        Reset service and model statistics
        
        Returns:
            Statistics reset response
        """
        try:
            reset_result = self.service.reset_statistics()
            
            if reset_result['status'] == 'success':
                return self._create_response(
                    success=True,
                    message="Statistics reset successfully",
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
    
    def validate_weather_input(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate weather data input format
        
        Args:
            request_data: Request dictionary containing 'weather_data'
        
        Returns:
            Weather data validation response
        """
        try:
            weather_data = request_data.get('weather_data')
            
            if not weather_data:
                return self._create_response(
                    success=False,
                    message="Weather data validation failed",
                    error="No weather_data provided in request",
                    data={'required_variables': self.service.get_input_variables()}
                )
            
            # Use service validation
            is_valid, validation_message, validated_weather = self.service.validate_weather_data(weather_data)
            
            if not is_valid:
                return self._create_response(
                    success=False,
                    message="Weather data validation failed",
                    error=validation_message,
                    data={'required_variables': self.service.get_input_variables()}
                )
            
            return self._create_response(
                success=True,
                message="Weather data validation successful",
                data={
                    'validation_message': validation_message,
                    'variables_count': len(validated_weather),
                    'days_per_variable': len(next(iter(validated_weather.values()), [])),
                    'detected_variables': list(validated_weather.keys())
                }
            )
            
        except Exception as e:
            logger.error(f"Controller weather validation error: {e}")
            return self._create_response(
                success=False,
                message="Weather data validation error",
                error=f"Controller error: {str(e)}"
            )