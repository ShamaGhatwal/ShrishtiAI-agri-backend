"""
GeoVision Fusion Prediction Controller
API request coordination and response formatting for GeoVision fusion predictions
"""

import logging
from typing import Dict, Optional, Any, Tuple
from datetime import datetime

from services.geovision_fusion_service import GeoVisionFusionService

logger = logging.getLogger(__name__)


class GeoVisionFusionController:
    """Controller for GeoVision fusion prediction API operations."""

    def __init__(self, service: Optional[GeoVisionFusionService] = None):
        """Initialize the controller with a service instance."""
        self.service = service or GeoVisionFusionService()
        self.controller_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0
        }
        logger.info("[GEOVISION_CTRL] Controller initialized")

    def initialize_controller(self) -> Dict[str, Any]:
        """Initialize by setting up the service."""
        try:
            success, message = self.service.initialize_service()
            return self._create_response(
                success=success,
                message=message if success else "Controller initialization failed",
                data={'service_status': 'ready' if success else 'failed'},
                error=None if success else message
            )
        except Exception as e:
            return self._create_response(
                success=False,
                message="Controller initialization error",
                error=str(e)
            )

    def predict_fusion(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Primary endpoint: run GeoVision fusion prediction for a location.

        Expected request_data:
            latitude: float     (-90 to 90)
            longitude: float    (-180 to 180)
        """
        self.controller_stats['total_requests'] += 1

        try:
            # Reject if TF models are still warming up in the background thread
            if not self.service.service_stats.get('models_loaded'):
                return self._create_response(
                    success=False,
                    message="Service is warming up, please retry in a moment",
                    error="models_not_ready"
                )

            # Validate required fields
            is_valid, msg, parsed = self._validate_request(request_data)
            if not is_valid:
                return self._create_response(
                    success=False,
                    message="Validation failed",
                    error=msg,
                    data={
                        'required_fields': ['latitude', 'longitude']
                    }
                )

            latitude, longitude = parsed

            # Call service (no reference_date — service auto-selects most recent)
            result = self.service.predict_for_location(latitude, longitude)

            if result.get('success'):
                self.controller_stats['successful_requests'] += 1
                return self._create_response(
                    success=True,
                    message="GeoVision fusion prediction completed",
                    data={
                        'location': result['location'],
                        'prediction': result['prediction'],
                        'intermediate': result.get('intermediate', {}),
                        'metadata': result.get('metadata', {}),
                        'data_collection_summary': {
                            'weather_data': result['data_collection']['weather']['success'],
                            'feature_engineering': result['data_collection']['features']['success'],
                            'raster_data': result['data_collection']['raster']['success'],
                        }
                    },
                    processing_info={
                        'processing_time_seconds': result['processing_time_seconds'],
                        'disaster_prediction': result['prediction']['disaster_prediction'],
                        'weather_prediction': result['prediction']['weather_prediction'],
                        'models_used': result.get('intermediate', {}).get('models_used', [])
                    }
                )
            else:
                self.controller_stats['failed_requests'] += 1
                return self._create_response(
                    success=False,
                    message="Fusion prediction failed",
                    error=result.get('error', 'Unknown error'),
                    data={
                        'location': result.get('location'),
                        'data_collection': result.get('data_collection')
                    }
                )

        except Exception as e:
            self.controller_stats['failed_requests'] += 1
            logger.error(f"[GEOVISION_CTRL] Error: {e}")
            return self._create_response(
                success=False,
                message="Fusion prediction error",
                error=str(e)
            )

    def get_service_status(self) -> Dict[str, Any]:
        """Return service health and model status."""
        status = self.service.get_service_status()
        return self._create_response(
            success=True,
            message="GeoVision service status",
            data=status
        )

    # ────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ────────────────────────────────────────────────────────
    def _validate_request(self, data: Dict[str, Any]) -> Tuple[bool, str, Optional[Tuple]]:
        """Validate prediction request."""
        if 'latitude' not in data:
            return False, "Missing required field: 'latitude'", None
        if 'longitude' not in data:
            return False, "Missing required field: 'longitude'", None
        try:
            lat = float(data['latitude'])
            lon = float(data['longitude'])
        except (ValueError, TypeError):
            return False, "latitude/longitude must be numeric", None
        if not (-90 <= lat <= 90):
            return False, f"Invalid latitude {lat}", None
        if not (-180 <= lon <= 180):
            return False, f"Invalid longitude {lon}", None

        return True, "OK", (lat, lon)

    def _create_response(self, success: bool = True, message: str = '',
                         data: Any = None, error: str = None,
                         processing_info: Optional[Dict] = None) -> Dict[str, Any]:
        """Build standardized response."""
        response = {
            'success': success,
            'message': message,
            'data': data,
            'timestamp': datetime.now().isoformat(),
            'service': 'geovision_fusion'
        }
        if error:
            response['error'] = error
        if processing_info:
            response['processing_info'] = processing_info
        return response
