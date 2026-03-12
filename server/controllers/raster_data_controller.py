"""
Raster Data Controller for HazardGuard System
API request coordination and response formatting for raster data operations
"""

import logging
from typing import Dict, List, Optional, Any, Union
from flask import request, jsonify
import pandas as pd

from services.raster_data_service import RasterDataService

logger = logging.getLogger(__name__)

class RasterDataController:
    """Controller layer for raster data API operations"""
    
    def __init__(self, raster_config: Optional[Dict[str, Any]] = None):
        """Initialize raster data controller"""
        self.service = RasterDataService(raster_config)
        self.request_count = 0
        
    def process_raster_extraction(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle raster extraction API request"""
        try:
            self.request_count += 1
            logger.info(f"Processing raster extraction request #{self.request_count}")
            
            # Validate request structure
            if not isinstance(request_data, dict):
                return self._error_response("Request must be a JSON object", 400)
            
            # Extract coordinates
            coordinates = request_data.get('coordinates', [])
            if not coordinates:
                return self._error_response("'coordinates' field is required and must be non-empty", 400)
            
            # Validate coordinates format
            is_valid, validation_message = self.service.validate_coordinates(coordinates)
            if not is_valid:
                return self._error_response(f"Invalid coordinates: {validation_message}", 400)
            
            # Extract optional features filter
            features = request_data.get('features')
            if features and not isinstance(features, list):
                return self._error_response("'features' must be a list of feature names", 400)
            
            # Process extraction
            result = self.service.process_raster_extraction(coordinates, features)
            
            if result['success']:
                return self._success_response(
                    data=result['data'],
                    metadata=result['metadata'],
                    message=f"Successfully extracted raster data for {len(coordinates)} coordinates"
                )
            else:
                return self._error_response(result['error'], 500)
                
        except Exception as e:
            logger.error(f"Error in raster extraction controller: {e}")
            return self._error_response(f"Internal server error: {str(e)}", 500)
    
    def process_batch_extraction(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle batch raster extraction API request"""
        try:
            self.request_count += 1
            logger.info(f"Processing batch raster extraction request #{self.request_count}")
            
            # Validate request structure
            if not isinstance(request_data, dict):
                return self._error_response("Request must be a JSON object", 400)
            
            # Extract coordinates
            coordinates = request_data.get('coordinates', [])
            if not coordinates:
                return self._error_response("'coordinates' field is required and must be non-empty", 400)
            
            # Validate coordinates format
            is_valid, validation_message = self.service.validate_coordinates(coordinates)
            if not is_valid:
                return self._error_response(f"Invalid coordinates: {validation_message}", 400)
            
            # Extract batch size
            batch_size = request_data.get('batch_size', 100)
            if not isinstance(batch_size, int) or batch_size <= 0:
                return self._error_response("'batch_size' must be a positive integer", 400)
            
            # Extract optional features filter
            features = request_data.get('features')
            if features and not isinstance(features, list):
                return self._error_response("'features' must be a list of feature names", 400)
            
            # Process batch extraction
            result = self.service.process_batch_extraction(coordinates, batch_size, features)
            
            if result['success']:
                return self._success_response(
                    data=result['data'],
                    metadata=result['metadata'],
                    message=f"Successfully processed batch extraction for {len(coordinates)} coordinates"
                )
            else:
                return self._error_response(result['error'], 500)
                
        except Exception as e:
            logger.error(f"Error in batch raster extraction controller: {e}")
            return self._error_response(f"Internal server error: {str(e)}", 500)
    
    def create_dataframe(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle DataFrame creation API request"""
        try:
            self.request_count += 1
            logger.info(f"Processing DataFrame creation request #{self.request_count}")
            
            # Validate request structure
            if not isinstance(request_data, dict):
                return self._error_response("Request must be a JSON object", 400)
            
            # Extract coordinates
            coordinates = request_data.get('coordinates', [])
            if not coordinates:
                return self._error_response("'coordinates' field is required and must be non-empty", 400)
            
            # Validate coordinates format
            is_valid, validation_message = self.service.validate_coordinates(coordinates)
            if not is_valid:
                return self._error_response(f"Invalid coordinates: {validation_message}", 400)
            
            # Extract optional features filter
            features = request_data.get('features')
            if features and not isinstance(features, list):
                return self._error_response("'features' must be a list of feature names", 400)
            
            # Create DataFrame
            result = self.service.create_raster_dataframe(coordinates, features)
            
            if result['success']:
                # Convert DataFrame to dict for JSON response
                df_dict = result['dataframe'].to_dict('records') if result['dataframe'] is not None else []
                
                return self._success_response(
                    data=df_dict,
                    metadata=result['metadata'],
                    message=f"Successfully created DataFrame with {len(df_dict)} rows"
                )
            else:
                return self._error_response(result['error'], 500)
                
        except Exception as e:
            logger.error(f"Error in DataFrame creation controller: {e}")
            return self._error_response(f"Internal server error: {str(e)}", 500)
    
    def export_data(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data export API request"""
        try:
            self.request_count += 1
            logger.info(f"Processing data export request #{self.request_count}")
            
            # Validate request structure
            if not isinstance(request_data, dict):
                return self._error_response("Request must be a JSON object", 400)
            
            # Extract coordinates
            coordinates = request_data.get('coordinates', [])
            if not coordinates:
                return self._error_response("'coordinates' field is required and must be non-empty", 400)
            
            # Validate coordinates format
            is_valid, validation_message = self.service.validate_coordinates(coordinates)
            if not is_valid:
                return self._error_response(f"Invalid coordinates: {validation_message}", 400)
            
            # Extract export format
            export_format = request_data.get('format', 'json').lower()
            if export_format not in ['json', 'csv', 'excel']:
                return self._error_response("'format' must be one of: json, csv, excel", 400)
            
            # Extract optional features filter
            features = request_data.get('features')
            if features and not isinstance(features, list):
                return self._error_response("'features' must be a list of feature names", 400)
            
            # Export data
            result = self.service.export_raster_data(coordinates, export_format, features)
            
            if result['success']:
                # Handle different export formats
                if export_format == 'excel':
                    # Convert DataFrame to dict for JSON response
                    exported_data = result['data'].to_dict('records') if hasattr(result['data'], 'to_dict') else result['data']
                else:
                    exported_data = result['data']
                
                return self._success_response(
                    data=exported_data,
                    metadata=result['metadata'],
                    message=f"Successfully exported data in {export_format} format"
                )
            else:
                return self._error_response(result['error'], 500)
                
        except Exception as e:
            logger.error(f"Error in data export controller: {e}")
            return self._error_response(f"Internal server error: {str(e)}", 500)
    
    def validate_coordinates(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle coordinate validation API request"""
        try:
            self.request_count += 1
            logger.info(f"Processing coordinate validation request #{self.request_count}")
            
            # Validate request structure
            if not isinstance(request_data, dict):
                return self._error_response("Request must be a JSON object", 400)
            
            # Extract coordinates
            coordinates = request_data.get('coordinates', [])
            if not coordinates:
                return self._error_response("'coordinates' field is required", 400)
            
            # Validate coordinates
            is_valid, validation_message = self.service.validate_coordinates(coordinates)
            
            return self._success_response(
                data={
                    'valid': is_valid,
                    'message': validation_message,
                    'coordinate_count': len(coordinates)
                },
                metadata={
                    'validation_timestamp': logger.handlers[0].formatter.formatTime() if logger.handlers else None
                },
                message="Coordinate validation completed"
            )
                
        except Exception as e:
            logger.error(f"Error in coordinate validation controller: {e}")
            return self._error_response(f"Internal server error: {str(e)}", 500)
    
    def get_available_features(self) -> Dict[str, Any]:
        """Handle get available features API request"""
        try:
            self.request_count += 1
            logger.info(f"Processing get available features request #{self.request_count}")
            
            result = self.service.get_available_features()
            
            if result['success']:
                return self._success_response(
                    data=result['features'],
                    metadata={
                        'availability': result['availability'],
                        'configuration': result['metadata']
                    },
                    message="Successfully retrieved available features"
                )
            else:
                return self._error_response(result['error'], 500)
                
        except Exception as e:
            logger.error(f"Error in get available features controller: {e}")
            return self._error_response(f"Internal server error: {str(e)}", 500)
    
    def get_service_status(self) -> Dict[str, Any]:
        """Handle service status API request"""
        try:
            self.request_count += 1
            logger.info(f"Processing service status request #{self.request_count}")
            
            stats = self.service.get_processing_statistics()
            validation = self.service.validate_raster_configuration()
            
            status_data = {
                'service_health': 'healthy',
                'request_count': self.request_count,
                'processing_statistics': stats['statistics'] if stats['success'] else None,
                'configuration_validation': validation['validation'] if validation['success'] else None
            }
            
            # Determine overall health
            if not stats['success'] or not validation['success']:
                status_data['service_health'] = 'degraded'
            elif validation['success'] and validation['summary']['readable_sources'] == 0:
                status_data['service_health'] = 'no_data'
            
            return self._success_response(
                data=status_data,
                metadata={
                    'timestamp': logger.handlers[0].formatter.formatTime() if logger.handlers else None
                },
                message="Service status retrieved successfully"
            )
                
        except Exception as e:
            logger.error(f"Error in service status controller: {e}")
            return self._error_response(f"Internal server error: {str(e)}", 500)
    
    def test_extraction(self) -> Dict[str, Any]:
        """Handle test extraction API request"""
        try:
            self.request_count += 1
            logger.info(f"Processing test extraction request #{self.request_count}")
            
            result = self.service.test_raster_extraction()
            
            if result['success']:
                return self._success_response(
                    data=result.get('test_data'),
                    metadata={
                        'processing_time': result.get('processing_time'),
                        'test_coordinates': [{'longitude': 121.0, 'latitude': 14.0}]
                    },
                    message=result['message']
                )
            else:
                return self._error_response(result.get('error', 'Test extraction failed'), 500)
                
        except Exception as e:
            logger.error(f"Error in test extraction controller: {e}")
            return self._error_response(f"Internal server error: {str(e)}", 500)
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Handle feature information API request"""
        try:
            self.request_count += 1
            logger.info(f"Processing feature info request #{self.request_count}")
            
            result = self.service.get_available_features()
            
            if result['success']:
                # Restructure response for better API usability
                feature_info = {}
                for feature_name, feature_details in result['features'].items():
                    feature_info[feature_name] = {
                        **feature_details,
                        'available': result['availability'][feature_name]['available'],
                        'path_configured': result['availability'][feature_name]['path_configured']
                    }
                
                return self._success_response(
                    data=feature_info,
                    metadata=result['metadata'],
                    message="Feature information retrieved successfully"
                )
            else:
                return self._error_response(result['error'], 500)
                
        except Exception as e:
            logger.error(f"Error in feature info controller: {e}")
            return self._error_response(f"Internal server error: {str(e)}", 500)
    
    def _success_response(self, data: Any = None, metadata: Dict[str, Any] = None, 
                         message: str = "Success", status_code: int = 200) -> Dict[str, Any]:
        """Create standardized success response"""
        return {
            'success': True,
            'message': message,
            'data': data,
            'metadata': metadata or {},
            'status_code': status_code
        }
    
    def _error_response(self, error_message: str, status_code: int = 400) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            'success': False,
            'error': error_message,
            'data': None,
            'metadata': {
                'request_count': self.request_count,
                'timestamp': logger.handlers[0].formatter.formatTime() if logger.handlers else None
            },
            'status_code': status_code
        }
    
    def get_request_statistics(self) -> Dict[str, Any]:
        """Get controller request statistics"""
        service_stats = self.service.get_processing_statistics()
        
        return {
            'success': True,
            'statistics': {
                'total_api_requests': self.request_count,
                'service_statistics': service_stats['statistics'] if service_stats['success'] else None
            }
        }