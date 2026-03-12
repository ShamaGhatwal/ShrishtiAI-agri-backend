"""
Feature Engineering Controller
Handles feature engineering operations and coordinates between service and API
"""
import logging
from typing import Dict, Any, List, Optional
from services.feature_engineering_service import FeatureEngineeringService
from models.feature_engineering_model import WeatherFeatureModel
from utils import create_error_response, create_success_response

class FeatureEngineeringController:
    """Controller for weather feature engineering operations"""
    
    def __init__(self, feature_service: FeatureEngineeringService):
        self.feature_service = feature_service
        self.logger = logging.getLogger(__name__)
    
    def process_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process weather data to compute engineered features
        
        Args:
            data: Request data containing weather_data and optional parameters
            
        Returns:
            Feature engineering response
        """
        try:
            # Validate required parameters
            if 'weather_data' not in data or not data['weather_data']:
                return create_error_response(
                    "Missing required parameter: 'weather_data'",
                    {"required_fields": ["weather_data"]}
                )
            
            weather_data = data['weather_data']
            event_duration = data.get('event_duration', 1.0)
            include_metadata = data.get('include_metadata', True)
            
            # Validate event duration
            try:
                event_duration = float(event_duration) if event_duration else 1.0
                if event_duration <= 0:
                    event_duration = 1.0
            except (ValueError, TypeError):
                return create_error_response(
                    "Invalid event_duration: must be a positive number",
                    {"event_duration": event_duration}
                )
            
            self.logger.info(f"Processing features for weather data with {len(weather_data)} fields, "
                           f"event_duration: {event_duration} days")
            
            # Process features
            success, result = self.feature_service.process_weather_features(
                weather_data, event_duration, include_metadata
            )
            
            if success:
                return create_success_response(result)
            else:
                return create_error_response(
                    "Failed to process engineered features",
                    result
                )
                
        except Exception as e:
            self.logger.error(f"Feature processing error: {str(e)}")
            return create_error_response(
                f"Failed to process features: {str(e)}"
            )
    
    def process_batch_features(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process multiple weather datasets for feature engineering
        
        Args:
            data: Request data containing batch of weather datasets
            
        Returns:
            Batch feature engineering response
        """
        try:
            # Validate batch request
            if 'batch_data' not in data or not isinstance(data['batch_data'], list):
                return create_error_response(
                    "Invalid batch request: 'batch_data' array required"
                )
            
            batch_data = data['batch_data']
            include_metadata = data.get('include_metadata', True)
            
            if len(batch_data) > 100:  # Limit batch size
                return create_error_response(
                    "Batch size too large: maximum 100 items allowed",
                    {"max_allowed": 100, "requested": len(batch_data)}
                )
            
            self.logger.info(f"Processing batch feature engineering for {len(batch_data)} datasets")
            
            # Process batch
            success, result = self.feature_service.process_batch_features(
                batch_data, include_metadata
            )
            
            if success:
                return create_success_response(result)
            else:
                return create_error_response(
                    "Failed to process batch features",
                    result
                )
                
        except Exception as e:
            self.logger.error(f"Batch feature processing error: {str(e)}")
            return create_error_response(
                f"Failed to process batch features: {str(e)}"
            )
    
    def create_feature_dataframe(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create DataFrame with weather data and engineered features
        
        Args:
            data: Request data containing weather_data, disaster_date, and days_before
            
        Returns:
            DataFrame creation response
        """
        try:
            # Validate required parameters
            required_fields = ['weather_data', 'disaster_date', 'days_before']
            missing_fields = [field for field in required_fields if field not in data or data[field] is None]
            
            if missing_fields:
                return create_error_response(
                    f"Missing required fields: {', '.join(missing_fields)}",
                    {"missing_fields": missing_fields}
                )
            
            weather_data = data['weather_data']
            disaster_date = str(data['disaster_date'])
            event_duration = data.get('event_duration', 1.0)
            
            try:
                days_before = int(data['days_before'])
                event_duration = float(event_duration)
            except (ValueError, TypeError) as e:
                return create_error_response(
                    f"Invalid parameter format: {str(e)}",
                    {"validation_error": str(e)}
                )
            
            self.logger.info(f"Creating feature DataFrame for {disaster_date}, "
                           f"{days_before} days, duration: {event_duration}")
            
            # Process features first
            success, feature_result = self.feature_service.process_weather_features(
                weather_data, event_duration, include_metadata=True
            )
            
            if not success:
                return create_error_response(
                    "Failed to process features for DataFrame",
                    feature_result
                )
            
            # Create DataFrame
            try:
                df = self.feature_service.create_feature_dataframe(
                    weather_data,
                    feature_result['engineered_features'],
                    disaster_date,
                    days_before
                )
                
                # Convert DataFrame to dict for JSON response
                dataframe_data = {
                    'dates': df['date'].tolist(),
                    'weather_data': {
                        col: df[col].tolist()
                        for col in df.columns 
                        if col in WeatherFeatureModel.WEATHER_FIELDS
                    },
                    'engineered_features': {
                        col: df[col].tolist()
                        for col in df.columns
                        if col in WeatherFeatureModel.ENGINEERED_FEATURES
                    }
                }
                
                return create_success_response({
                    'dataframe': dataframe_data,
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'metadata': feature_result.get('metadata', {}),
                    'validation': feature_result.get('validation', {})
                })
                
            except Exception as e:
                return create_error_response(
                    f"Failed to create DataFrame: {str(e)}"
                )
                
        except Exception as e:
            self.logger.error(f"DataFrame creation error: {str(e)}")
            return create_error_response(
                f"Failed to create feature DataFrame: {str(e)}"
            )
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about available engineered features"""
        try:
            feature_info = self.feature_service.get_feature_info()
            
            return create_success_response({
                'feature_info': feature_info,
                'service_status': self.feature_service.get_service_status()
            })
            
        except Exception as e:
            self.logger.error(f"Feature info error: {str(e)}")
            return create_error_response(
                f"Failed to get feature info: {str(e)}"
            )
    
    def validate_weather_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate weather data for feature engineering
        
        Args:
            data: Request data containing weather_data
            
        Returns:
            Validation response
        """
        try:
            if 'weather_data' not in data or not data['weather_data']:
                return create_error_response(
                    "Missing required parameter: 'weather_data'",
                    {"required_fields": ["weather_data"]}
                )
            
            weather_data = data['weather_data']
            
            # Validate data
            is_valid, validation = self.feature_service.validate_input_data(weather_data)
            
            validation_result = {
                'validation': validation,
                'is_valid': is_valid,
                'ready_for_processing': is_valid
            }
            
            if is_valid:
                return create_success_response(validation_result)
            else:
                return create_error_response(
                    "Weather data validation failed",
                    validation_result
                )
                
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            return create_error_response(
                f"Failed to validate weather data: {str(e)}"
            )
    
    def process_and_export(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process features and export in specified format
        
        Args:
            data: Request data with weather_data, disaster_date, days_before, and export options
            
        Returns:
            Export response
        """
        try:
            # Validate required parameters
            required_fields = ['weather_data', 'disaster_date', 'days_before']
            missing_fields = [field for field in required_fields if field not in data or data[field] is None]
            
            if missing_fields:
                return create_error_response(
                    f"Missing required fields: {', '.join(missing_fields)}",
                    {"missing_fields": missing_fields}
                )
            
            weather_data = data['weather_data']
            disaster_date = str(data['disaster_date'])
            event_duration = data.get('event_duration', 1.0)
            export_format = data.get('export_format', 'dict').lower()
            
            try:
                days_before = int(data['days_before'])
                event_duration = float(event_duration)
            except (ValueError, TypeError) as e:
                return create_error_response(
                    f"Invalid parameter format: {str(e)}",
                    {"validation_error": str(e)}
                )
            
            # Validate export format
            valid_formats = ['dict', 'dataframe', 'json']
            if export_format not in valid_formats:
                return create_error_response(
                    f"Invalid export format: {export_format}",
                    {"valid_formats": valid_formats}
                )
            
            self.logger.info(f"Processing and exporting features in '{export_format}' format")
            
            # Process and export
            success, result = self.feature_service.process_and_export(
                weather_data, disaster_date, days_before, event_duration, export_format
            )
            
            if success:
                # Handle DataFrame special case for JSON response
                if export_format == 'dataframe' and 'export' in result:
                    export_data = result['export']
                    if 'dataframe' in export_data:
                        # Convert DataFrame to dict for JSON serialization
                        df = export_data['dataframe']
                        export_data['dataframe_dict'] = df.to_dict(orient='list')
                        # Remove actual DataFrame object for JSON response
                        del export_data['dataframe']
                
                return create_success_response(result)
            else:
                return create_error_response(
                    "Failed to process and export features",
                    result
                )
                
        except Exception as e:
            self.logger.error(f"Process and export error: {str(e)}")
            return create_error_response(
                f"Failed to process and export: {str(e)}"
            )
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get feature engineering service status and health"""
        try:
            service_status = self.feature_service.get_service_status()
            
            return create_success_response({
                'controller': 'Feature Engineering Controller',
                'service': service_status,
                'health': 'healthy' if service_status.get('initialized') else 'unhealthy',
                'available_operations': [
                    'process_features',
                    'process_batch_features', 
                    'create_feature_dataframe',
                    'validate_weather_data',
                    'process_and_export',
                    'get_feature_info'
                ]
            })
            
        except Exception as e:
            self.logger.error(f"Service status error: {str(e)}")
            return create_error_response(
                f"Failed to get service status: {str(e)}"
            )