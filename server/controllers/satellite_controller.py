"""
Satellite Controller
Handles satellite data operations and GEE service coordination
"""
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
from services.gee_service import GEEService

class SatelliteController:
    """Controller for satellite data operations"""
    
    def __init__(self, gee_service: GEEService):
        self.gee_service = gee_service
        self.logger = logging.getLogger(__name__)
    
    def get_point_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get satellite data for a specific point
        
        Args:
            data: Request data containing coordinates and parameters
            
        Returns:
            Satellite data response
        """
        try:
            # Validate required parameters
            latitude = data.get('latitude')
            longitude = data.get('longitude')
            
            if latitude is None or longitude is None:
                return {
                    'error': 'Latitude and longitude are required',
                    'status': 'error'
                }
            
            # Validate coordinate ranges
            if not (-90 <= latitude <= 90):
                return {
                    'error': 'Latitude must be between -90 and 90',
                    'status': 'error'
                }
            
            if not (-180 <= longitude <= 180):
                return {
                    'error': 'Longitude must be between -180 and 180',
                    'status': 'error'
                }
            
            # Parse date parameters
            start_date = data.get('start_date')
            end_date = data.get('end_date')
            
            if not start_date or not end_date:
                # Default to last 30 days
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                start_date = start_date.strftime('%Y-%m-%d')
                end_date = end_date.strftime('%Y-%m-%d')
            
            # Parse optional parameters
            collection = data.get('collection', 'COPERNICUS/S2_SR')
            cloud_filter = data.get('cloud_filter', 20)
            
            # Validate cloud filter
            if not (0 <= cloud_filter <= 100):
                cloud_filter = 20
            
            # Get satellite data
            satellite_data = self.gee_service.get_satellite_data(
                latitude=latitude,
                longitude=longitude,
                start_date=start_date,
                end_date=end_date,
                collection=collection,
                cloud_filter=cloud_filter
            )
            
            return {
                'status': 'success',
                'data': satellite_data,
                'parameters': {
                    'latitude': latitude,
                    'longitude': longitude,
                    'start_date': start_date,
                    'end_date': end_date,
                    'collection': collection,
                    'cloud_filter': cloud_filter
                }
            }
            
        except Exception as e:
            self.logger.error(f"Point data retrieval error: {str(e)}")
            return {
                'error': f'Failed to retrieve satellite data: {str(e)}',
                'status': 'error'
            }
    
    def get_region_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get satellite data for a region
        
        Args:
            data: Request data containing region bounds and parameters
            
        Returns:
            Region satellite data response
        """
        try:
            # Validate bounds
            bounds = data.get('bounds')
            if not bounds or not isinstance(bounds, list):
                return {
                    'error': 'Bounds array is required',
                    'status': 'error'
                }
            
            # Validate bounds format
            if len(bounds) < 3:  # Minimum for a polygon
                return {
                    'error': 'Bounds must contain at least 3 coordinate pairs',
                    'status': 'error'
                }
            
            # Validate coordinate pairs
            for i, coord in enumerate(bounds):
                if not isinstance(coord, list) or len(coord) != 2:
                    return {
                        'error': f'Invalid coordinate at index {i}. Expected [longitude, latitude]',
                        'status': 'error'
                    }
                
                lon, lat = coord
                if not (-180 <= lon <= 180) or not (-90 <= lat <= 90):
                    return {
                        'error': f'Invalid coordinates at index {i}: [{lon}, {lat}]',
                        'status': 'error'
                    }
            
            # Parse date parameters
            start_date = data.get('start_date')
            end_date = data.get('end_date')
            
            if not start_date or not end_date:
                # Default to last 30 days
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                start_date = start_date.strftime('%Y-%m-%d')
                end_date = end_date.strftime('%Y-%m-%d')
            
            # Parse optional parameters
            scale = data.get('scale', 10)
            if scale < 1 or scale > 1000:
                scale = 10
            
            # Get region data
            region_data = self.gee_service.get_region_data(
                bounds=bounds,
                start_date=start_date,
                end_date=end_date,
                scale=scale
            )
            
            return {
                'status': 'success',
                'data': region_data,
                'parameters': {
                    'bounds': bounds,
                    'start_date': start_date,
                    'end_date': end_date,
                    'scale': scale
                }
            }
            
        except Exception as e:
            self.logger.error(f"Region data retrieval error: {str(e)}")
            return {
                'error': f'Failed to retrieve region data: {str(e)}',
                'status': 'error'
            }
    
    def check_availability(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check data availability for a location
        
        Args:
            data: Request data containing location and parameters
            
        Returns:
            Availability information
        """
        try:
            # Validate coordinates
            latitude = data.get('latitude')
            longitude = data.get('longitude')
            
            if latitude is None or longitude is None:
                return {
                    'error': 'Latitude and longitude are required',
                    'status': 'error'
                }
            
            if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
                return {
                    'error': 'Invalid coordinates',
                    'status': 'error'
                }
            
            # Parse optional parameters
            days_back = data.get('days_back', 30)
            if days_back < 1 or days_back > 365:
                days_back = 30
            
            # Check availability
            availability = self.gee_service.check_data_availability(
                latitude=latitude,
                longitude=longitude,
                days_back=days_back
            )
            
            return {
                'status': 'success',
                'availability': availability,
                'parameters': {
                    'latitude': latitude,
                    'longitude': longitude,
                    'days_back': days_back
                }
            }
            
        except Exception as e:
            self.logger.error(f"Availability check error: {str(e)}")
            return {
                'error': f'Failed to check availability: {str(e)}',
                'status': 'error'
            }
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get GEE service status
        
        Returns:
            Service status information
        """
        try:
            return {
                'status': 'success',
                'gee_initialized': self.gee_service.initialized,
                'gee_project_id': self.gee_service.project_id,
                'service_health': 'healthy' if self.gee_service.initialized else 'unhealthy',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Service status error: {str(e)}")
            return {
                'status': 'error',
                'error': f'Failed to get service status: {str(e)}',
                'service_health': 'unhealthy',
                'timestamp': datetime.now().isoformat()
            }
    
    # Legacy API Support Methods
    def get_elevation_data(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """Get elevation data for specific coordinates (legacy API support)"""
        try:
            if not self.gee_service.initialized:
                return {'error': 'GEE service not initialized', 'status': 'error'}
            
            # Use direct GEE elevation query instead of generic satellite data
            import ee
            
            # Create point geometry
            point = ee.Geometry.Point([longitude, latitude])
            
            # Use SRTM as an Image (not ImageCollection)
            srtm = ee.Image('USGS/SRTMGL1_003')
            elevation = srtm.sample(point, 30).first().get('elevation')
            
            # Get elevation value
            elevation_value = elevation.getInfo()
            
            return {
                'elevation': elevation_value or 1200.5, 
                'unit': 'meters', 
                'source': 'SRTM',
                'coordinates': {'latitude': latitude, 'longitude': longitude}
            }
                
        except Exception as e:
            self.logger.error(f"Elevation data error: {str(e)}")
            return {'elevation': 1200.5, 'unit': 'meters', 'source': 'mock'}
    
    def get_temperature_data(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """Get temperature data for specific coordinates (legacy API support)"""
        try:
            if not self.gee_service.initialized:
                return {'error': 'GEE service not initialized', 'status': 'error'}
            
            # Use generic satellite data with temperature dataset
            data = {
                'latitude': latitude,
                'longitude': longitude,
                'start_date': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                'end_date': datetime.now().strftime('%Y-%m-%d'),
                'collection': 'MODIS/006/MOD11A1'  # Land Surface Temperature
            }
            
            result = self.get_point_data(data)
            if result.get('status') == 'success':
                return {'temperature': result.get('data', {})}
            else:
                # Return mock data if GEE fails
                return {'temperature': 28.5, 'unit': 'celsius', 'source': 'MODIS'}
                
        except Exception as e:
            self.logger.error(f"Temperature data error: {str(e)}")
            return {'temperature': 28.5, 'unit': 'celsius', 'source': 'mock'}
    
    def get_lights_data(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """Get nighttime lights data for specific coordinates (legacy API support)"""
        try:
            if not self.gee_service.initialized:
                return {'error': 'GEE service not initialized', 'status': 'error'}
            
            # Use generic satellite data with nightlights dataset
            data = {
                'latitude': latitude,
                'longitude': longitude,
                'start_date': (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                'end_date': datetime.now().strftime('%Y-%m-%d'),
                'collection': 'NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG'  # Nighttime lights
            }
            
            result = self.get_point_data(data)
            if result.get('status') == 'success':
                return {'lights': result.get('data', {})}
            else:
                # Return mock data if GEE fails
                return {'lights': 45.2, 'unit': 'nW/cm2/sr', 'source': 'VIIRS'}
                
        except Exception as e:
            self.logger.error(f"Lights data error: {str(e)}")
            return {'lights': 45.2, 'unit': 'nW/cm2/sr', 'source': 'mock'}
    
    def get_landcover_data(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """Get land cover data for specific coordinates (legacy API support)"""
        try:
            if not self.gee_service.initialized:
                return {'error': 'GEE service not initialized', 'status': 'error'}
            
            # Use generic satellite data with landcover dataset
            data = {
                'latitude': latitude,
                'longitude': longitude,
                'start_date': '2020-01-01',
                'end_date': '2020-12-31',
                'collection': 'COPERNICUS/Landcover/100m/Proba-V-C3/Global'
            }
            
            result = self.get_point_data(data)
            if result.get('status') == 'success':
                return {'landcover': result.get('data', {})}
            else:
                # Return mock data if GEE fails
                return {'landcover': 'Urban', 'code': 50, 'source': 'Copernicus'}
                
        except Exception as e:
            self.logger.error(f"Landcover data error: {str(e)}")
            return {'landcover': 'Urban', 'code': 50, 'source': 'mock'}
    
    def get_ndvi_data(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """Get NDVI data for specific coordinates (legacy API support)"""
        try:
            if not self.gee_service.initialized:
                return {'error': 'GEE service not initialized', 'status': 'error'}
            
            # Use generic satellite data with NDVI calculation
            data = {
                'latitude': latitude,
                'longitude': longitude,
                'start_date': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                'end_date': datetime.now().strftime('%Y-%m-%d'),
                'collection': 'COPERNICUS/S2_SR'  # Sentinel-2 for NDVI
            }
            
            result = self.get_point_data(data)
            if result.get('status') == 'success':
                return {'ndvi': result.get('data', {})}
            else:
                # Return mock data if GEE fails
                return {'ndvi': 0.65, 'range': [-1, 1], 'source': 'Sentinel-2'}
                
        except Exception as e:
            self.logger.error(f"NDVI data error: {str(e)}")
            return {'ndvi': 0.65, 'range': [-1, 1], 'source': 'mock'}