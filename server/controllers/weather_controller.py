"""
Weather Controller
Handles weather data operations and coordinates between service and API
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from services.weather_service import NASAPowerService
from models.weather_model import WeatherRequest, WeatherDataModel
from utils import create_error_response, create_success_response

class WeatherController:
    """Controller for weather data operations"""
    
    def __init__(self, weather_service: NASAPowerService):
        self.weather_service = weather_service
        self.logger = logging.getLogger(__name__)
    
    def get_weather_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get weather data for specific coordinates and date
        
        Args:
            data: Request data containing coordinates, date, and optional parameters
            
        Returns:
            Weather data response
        """
        try:
            # Validate required parameters
            required_fields = ['latitude', 'longitude', 'disaster_date']
            missing_fields = [field for field in required_fields if field not in data or data[field] is None]
            
            if missing_fields:
                return create_error_response(
                    f"Missing required fields: {', '.join(missing_fields)}",
                    {"missing_fields": missing_fields}
                )
            
            # Extract parameters
            try:
                latitude = float(data['latitude'])
                longitude = float(data['longitude'])
                disaster_date = str(data['disaster_date'])
                days_before = int(data.get('days_before', 60))
            except (ValueError, TypeError) as e:
                return create_error_response(
                    f"Invalid parameter format: {str(e)}",
                    {"validation_error": str(e)}
                )
            
            # Create weather request
            weather_request = WeatherRequest(
                latitude=latitude,
                longitude=longitude,
                disaster_date=disaster_date,
                days_before=days_before
            )
            
            # Validate request
            validation = weather_request.validate()
            if not validation['valid']:
                return create_error_response(
                    "Request validation failed",
                    {"validation_errors": validation['errors']}
                )
            
            self.logger.info(f"Fetching weather data for lat={latitude}, lon={longitude}, "
                           f"disaster_date={disaster_date}, days_before={days_before}")
            
            # Fetch weather data
            success, result = self.weather_service.fetch_weather_data(weather_request)
            
            if success:
                return create_success_response(result)
            else:
                return create_error_response(
                    "Failed to fetch weather data",
                    result
                )
                
        except Exception as e:
            self.logger.error(f"Weather data error: {str(e)}")
            return create_error_response(
                f"Failed to get weather data: {str(e)}"
            )
    
    def get_weather_time_series(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get weather data as time series DataFrame
        
        Args:
            data: Request data containing coordinates, date, and optional parameters
            
        Returns:
            Time series weather data response
        """
        try:
            # Get weather data first
            weather_result = self.get_weather_data(data)
            
            if weather_result.get('status') != 'success':
                return weather_result
            
            # Extract weather data
            weather_data = weather_result['data']['weather_data']
            disaster_date = data['disaster_date']
            days_before = int(data.get('days_before', 60))
            
            # Create time series DataFrame
            df = WeatherDataModel.create_time_series_dataframe(
                weather_data, disaster_date, days_before
            )
            
            # Convert DataFrame to dict for JSON response
            time_series_data = {
                'dates': df['date'].tolist(),
                'weather_data': {
                    col: df[col].tolist() 
                    for col in df.columns if col != 'date'
                }
            }
            
            return create_success_response({
                'time_series': time_series_data,
                'metadata': weather_result['data']['metadata'],
                'validation': weather_result['data']['validation']
            })
            
        except Exception as e:
            self.logger.error(f"Time series error: {str(e)}")
            return create_error_response(
                f"Failed to create time series: {str(e)}"
            )
    
    def batch_get_weather_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get weather data for multiple locations
        
        Args:
            data: Request data containing list of location/date combinations
            
        Returns:
            Batch weather data response
        """
        try:
            # Validate batch request
            if 'locations' not in data or not isinstance(data['locations'], list):
                return create_error_response(
                    "Invalid batch request: 'locations' array required"
                )
            
            locations = data['locations']
            if len(locations) > 100:  # Limit batch size
                return create_error_response(
                    "Batch size too large: maximum 100 locations allowed",
                    {"max_allowed": 100, "requested": len(locations)}
                )
            
            # Create weather requests
            weather_requests = []
            for i, location in enumerate(locations):
                try:
                    request = WeatherRequest(
                        latitude=float(location['latitude']),
                        longitude=float(location['longitude']),
                        disaster_date=str(location['disaster_date']),
                        days_before=int(location.get('days_before', 60))
                    )
                    weather_requests.append(request)
                except Exception as e:
                    return create_error_response(
                        f"Invalid location at index {i}: {str(e)}",
                        {"location_index": i, "error": str(e)}
                    )
            
            self.logger.info(f"Starting batch weather fetch for {len(weather_requests)} locations")
            
            # Batch fetch weather data
            batch_result = self.weather_service.batch_fetch_weather_data(weather_requests)
            
            return create_success_response(batch_result)
            
        except Exception as e:
            self.logger.error(f"Batch weather error: {str(e)}")
            return create_error_response(
                f"Failed to process batch weather request: {str(e)}"
            )
    
    def get_weather_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get weather data summary statistics
        
        Args:
            data: Request data containing coordinates, date, and optional parameters
            
        Returns:
            Weather summary response with statistics
        """
        try:
            # Get weather data first
            weather_result = self.get_weather_data(data)
            
            if weather_result.get('status') != 'success':
                return weather_result
            
            weather_data = weather_result['data']['weather_data']
            
            # Calculate summary statistics
            summary_stats = {}
            for field_name, values in weather_data.items():
                valid_values = [v for v in values if v is not None]
                
                if valid_values:
                    summary_stats[field_name] = {
                        'mean': sum(valid_values) / len(valid_values),
                        'min': min(valid_values),
                        'max': max(valid_values),
                        'count': len(valid_values),
                        'missing': len([v for v in values if v is None]),
                        'completeness': len(valid_values) / len(values) * 100
                    }
                else:
                    summary_stats[field_name] = {
                        'mean': None, 'min': None, 'max': None,
                        'count': 0, 'missing': len(values),
                        'completeness': 0.0
                    }
            
            return create_success_response({
                'summary_statistics': summary_stats,
                'metadata': weather_result['data']['metadata'],
                'data_quality': weather_result['data']['validation']['data_quality']
            })
            
        except Exception as e:
            self.logger.error(f"Weather summary error: {str(e)}")
            return create_error_response(
                f"Failed to create weather summary: {str(e)}"
            )
    
    def get_available_fields(self) -> Dict[str, Any]:
        """Get available weather fields and their descriptions"""
        try:
            field_descriptions = {
                'temperature_C': 'Temperature at 2 meters (°C)',
                'humidity_perc': 'Relative humidity at 2 meters (%)',
                'wind_speed_mps': 'Wind speed at 2 meters (m/s)',
                'precipitation_mm': 'Precipitation corrected (mm)',
                'surface_pressure_hPa': 'Surface pressure (hPa)',
                'solar_radiation_wm2': 'Solar radiation (W/m²)',
                'temperature_max_C': 'Maximum temperature (°C)',
                'temperature_min_C': 'Minimum temperature (°C)',
                'specific_humidity_g_kg': 'Specific humidity at 2m (g/kg)',
                'dew_point_C': 'Dew point temperature at 2m (°C)',
                'wind_speed_10m_mps': 'Wind speed at 10 meters (m/s)',
                'cloud_amount_perc': 'Cloud amount (%)',
                'sea_level_pressure_hPa': 'Sea level pressure (hPa)',
                'surface_soil_wetness_perc': 'Surface soil wetness (%)',
                'wind_direction_10m_degrees': 'Wind direction at 10m (degrees)',
                'evapotranspiration_wm2': 'Evapotranspiration energy flux (W/m²)',
                'root_zone_soil_moisture_perc': 'Root zone soil moisture (%)'
            }
            
            return create_success_response({
                'available_fields': field_descriptions,
                'field_count': len(field_descriptions),
                'nasa_power_fields': WeatherDataModel.WEATHER_FIELDS,
                'service_info': {
                    'data_source': 'NASA POWER API',
                    'temporal_resolution': 'daily',
                    'spatial_resolution': '0.5° x 0.625°',
                    'coverage': 'global',
                    'data_lag': '~7 days'
                }
            })
            
        except Exception as e:
            self.logger.error(f"Available fields error: {str(e)}")
            return create_error_response(
                f"Failed to get available fields: {str(e)}"
            )
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get weather service status and health"""
        try:
            service_status = self.weather_service.get_service_status()
            
            return create_success_response({
                'controller': 'Weather Controller',
                'service': service_status,
                'health': 'healthy' if service_status.get('initialized') else 'unhealthy'
            })
            
        except Exception as e:
            self.logger.error(f"Service status error: {str(e)}")
            return create_error_response(
                f"Failed to get service status: {str(e)}"
            )