"""
Forest Department Service
Handles GEE-based forestry analysis including:
1. NDVI Analysis - Vegetation health (global layer)
2. Crop Classification - Classify crops using Random Forest
3. Soil Moisture / Irrigation Analysis - Dry vs irrigated areas
4. Fire Risk / Wildfire Map - Score fire risk
5. Plantation Suitability - Best places to plant trees
6. Compensatory Plantation Planner - Suggest new planting areas
7. Tree Growth / Regrowth Estimation - Estimate years to restore canopy
8. Species Recommendation Engine - Suggest plant species
"""

import ee
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional


class ForestDepartmentService:
    """Service class for Forest Department GEE operations"""

    def __init__(self, gee_service):
        """Initialize with existing GEE service instance"""
        self.gee_service = gee_service
        self.logger = logging.getLogger(__name__)

    def _ensure_initialized(self):
        """Ensure GEE is initialized before operations"""
        if not self.gee_service.initialized:
            raise Exception("GEE not initialized")

    def _get_tile_url(self, image, vis_params):
        """Generate a tile URL for an Earth Engine image"""
        try:
            map_id = image.getMapId(vis_params)
            return map_id["tile_fetcher"].url_format
        except Exception as e:
            self.logger.error(f"Error generating tile URL: {str(e)}")
            return None

    # ========================================
    # 1. NDVI ANALYSIS (Global Layer)
    # ========================================
    def get_ndvi_tiles(self) -> Dict[str, Any]:
        """
        Get global NDVI tiles for vegetation health visualization.

        Returns:
            Dictionary with tile URL and metadata
        """
        self._ensure_initialized()

        try:
            # Use last 3 months for good global coverage
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)

            collection = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterDate(
                    start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
                )
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
                .select(["B8", "B4"])
            )

            image = collection.median()
            ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")

            vis_params = {
                "min": 0.0,
                "max": 1.0,
                "palette": ["brown", "yellow", "green"],
            }

            tile_url = self._get_tile_url(ndvi, vis_params)

            return {
                "status": "success",
                "tile_url": tile_url,
                "metadata": {
                    "title": "NDVI (Vegetation Health)",
                    "description": "Normalized Difference Vegetation Index showing plant health",
                    "source": "Copernicus Sentinel-2 SR Harmonized",
                    "date_range": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                    "resolution": "10m",
                },
                "legend": [
                    {
                        "color": "#A52A2A",
                        "label": "No vegetation",
                        "value": "0.0 - 0.3",
                    },
                    {
                        "color": "#FFFF00",
                        "label": "Sparse vegetation",
                        "value": "0.3 - 0.6",
                    },
                    {
                        "color": "#008000",
                        "label": "Healthy vegetation",
                        "value": "0.6 - 1.0",
                    },
                ],
            }

        except Exception as e:
            self.logger.error(f"Error getting NDVI tiles: {str(e)}")
            return {"status": "error", "message": str(e)}

    # ========================================
    # 2. CROP CLASSIFICATION
    # ========================================
    def classify_crops(
        self, coordinates: List[List[float]], season: str = "kharif"
    ) -> Dict[str, Any]:
        """
        Classify crops using spectral indices and unsupervised classification.

        Since we don't have training data, we use k-means clustering combined
        with spectral signatures to identify different crop types.

        Args:
            coordinates: List of [lng, lat] coordinates defining the area
            season: 'kharif' (Jun-Oct), 'rabi' (Nov-Mar), or 'zaid' (Mar-Jun)

        Returns:
            Dictionary with crop classification results
        """
        self._ensure_initialized()

        try:
            polygon = ee.Geometry.Polygon([coordinates])
            total_area = polygon.area().getInfo()

            # Set date range based on season
            now = datetime.now()
            if season == "kharif":
                start_str, end_str = f"{now.year}-06-01", f"{now.year}-10-31"
            elif season == "rabi":
                start_str, end_str = f"{now.year}-11-01", f"{now.year + 1}-03-31"
            else:  # zaid
                start_str, end_str = f"{now.year}-03-01", f"{now.year}-06-30"

            # Get Sentinel-2 imagery - try current season first
            collection = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(polygon)
                .filterDate(start_str, end_str)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            )

            count = collection.size().getInfo()
            data_source = "current_season"
            actual_start = start_str
            actual_end = end_str

            # If no data for current season, search backwards for latest available
            if count == 0:
                self.logger.info(
                    f"No imagery for {season} season, searching for latest available..."
                )

                # Search last 12 months for any available imagery
                search_end = datetime.now()
                search_start = search_end - timedelta(days=365)

                collection = (
                    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                    .filterBounds(polygon)
                    .filterDate(
                        search_start.strftime("%Y-%m-%d"),
                        search_end.strftime("%Y-%m-%d"),
                    )
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
                    .sort("system:time_start", False)  # Sort by date descending
                )

                count = collection.size().getInfo()

                if count == 0:
                    # Still no data, try with higher cloud tolerance
                    collection = (
                        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                        .filterBounds(polygon)
                        .filterDate(
                            search_start.strftime("%Y-%m-%d"),
                            search_end.strftime("%Y-%m-%d"),
                        )
                        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
                        .sort("system:time_start", False)
                    )
                    count = collection.size().getInfo()

                if count == 0:
                    return {
                        "status": "no_data",
                        "message": "No satellite imagery available for this area in the past 12 months",
                    }

                # Get actual date range of available imagery
                dates = collection.aggregate_array("system:time_start").getInfo()
                if dates:
                    min_date = datetime.fromtimestamp(min(dates) / 1000)
                    max_date = datetime.fromtimestamp(max(dates) / 1000)
                    actual_start = min_date.strftime("%Y-%m-%d")
                    actual_end = max_date.strftime("%Y-%m-%d")

                data_source = "latest_available"

            # Create median composite
            s2 = collection.median()

            # Calculate spectral indices
            ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
            ndwi = s2.normalizedDifference(["B3", "B8"]).rename("NDWI")

            # EVI (Enhanced Vegetation Index)
            evi = s2.expression(
                "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
                {
                    "NIR": s2.select("B8"),
                    "RED": s2.select("B4"),
                    "BLUE": s2.select("B2"),
                },
            ).rename("EVI")

            # Stack bands for classification
            input_image = s2.select(["B2", "B3", "B4", "B8", "B11", "B12"]).addBands(
                [ndvi, ndwi, evi]
            )

            # Use unsupervised k-means clustering (5 classes)
            # This groups similar spectral signatures together
            training = input_image.sample(
                region=polygon, scale=10, numPixels=5000, seed=42
            )

            clusterer = ee.Clusterer.wekaKMeans(5).train(training)
            classified = input_image.cluster(clusterer).rename("crop_class")

            # Get cluster statistics to interpret classes
            scale = 10
            class_stats = (
                classified.reduceRegion(
                    reducer=ee.Reducer.frequencyHistogram(),
                    geometry=polygon,
                    scale=scale,
                    maxPixels=1e9,
                )
                .get("crop_class")
                .getInfo()
                or {}
            )

            # Get mean NDVI per cluster to help interpret
            cluster_ndvi = {}
            for i in range(5):
                mask = classified.eq(i)
                mean_ndvi = (
                    ndvi.updateMask(mask)
                    .reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=polygon,
                        scale=scale,
                        maxPixels=1e9,
                    )
                    .get("NDVI")
                    .getInfo()
                    or 0
                )
                cluster_ndvi[i] = mean_ndvi

            # Sort clusters by NDVI to assign interpretations
            # Higher NDVI = more vigorous crops (rice, sugarcane)
            # Lower NDVI = bare/fallow or sparse crops (wheat stubble)
            sorted_clusters = sorted(
                cluster_ndvi.items(), key=lambda x: x[1], reverse=True
            )

            # Create interpretation based on NDVI ranking
            crop_labels = {
                sorted_clusters[0][0]: {"name": "Rice/Dense Crops", "color": "#2E7D32"},
                sorted_clusters[1][0]: {
                    "name": "Sugarcane/Plantation",
                    "color": "#4CAF50",
                },
                sorted_clusters[2][0]: {"name": "Wheat/Cereals", "color": "#FFC107"},
                sorted_clusters[3][0]: {
                    "name": "Mixed/Other Crops",
                    "color": "#FF9800",
                },
                sorted_clusters[4][0]: {"name": "Fallow/Bare Land", "color": "#795548"},
            }

            # Calculate percentages
            total_pixels = sum(class_stats.values())
            classification = {}
            for class_id, pixels in class_stats.items():
                class_num = int(class_id)
                if class_num in crop_labels:
                    label_info = crop_labels[class_num]
                    classification[label_info["name"]] = {
                        "percentage": round((pixels / total_pixels) * 100, 2),
                        "area_hectares": round(
                            (pixels / total_pixels) * (total_area / 10000), 2
                        ),
                        "color": label_info["color"],
                    }

            # Generate tile URL
            vis_params = {
                "min": 0,
                "max": 4,
                "palette": ["#795548", "#FF9800", "#FFC107", "#4CAF50", "#2E7D32"],
            }
            tile_url = self._get_tile_url(classified.clip(polygon), vis_params)

            return {
                "status": "success",
                "total_area_hectares": round(total_area / 10000, 2),
                "season": season,
                "classification": classification,
                "cluster_ndvi": {
                    crop_labels.get(k, {}).get("name", f"Class {k}"): round(v, 4)
                    for k, v in cluster_ndvi.items()
                },
                "tile_url": tile_url,
                "legend": [
                    {"color": "#2E7D32", "label": "Rice/Dense Crops"},
                    {"color": "#4CAF50", "label": "Sugarcane/Plantation"},
                    {"color": "#FFC107", "label": "Wheat/Cereals"},
                    {"color": "#FF9800", "label": "Mixed/Other Crops"},
                    {"color": "#795548", "label": "Fallow/Bare Land"},
                ],
                "date_range": {"start": actual_start, "end": actual_end},
                "images_used": count,
                "data_source": data_source,
                "note": "Using latest available imagery"
                if data_source == "latest_available"
                else None,
            }

        except Exception as e:
            self.logger.error(f"Error classifying crops: {str(e)}")
            return {"status": "error", "message": str(e)}

    # ========================================
    # 3. SOIL MOISTURE (Global Layer)
    # ========================================
    def get_soil_moisture_tiles(self) -> Dict[str, Any]:
        """
        Get global soil moisture heatmap tiles.
        Uses NDMI (Normalized Difference Moisture Index) from Sentinel-2.

        Returns:
            Dictionary with tile URL and metadata
        """
        self._ensure_initialized()

        try:
            # Use last 3 months for good global coverage
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)

            collection = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterDate(
                    start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
                )
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
                .select(["B8", "B11"])
            )

            image = collection.median()

            # NDMI = (NIR - SWIR) / (NIR + SWIR) - measures vegetation water content
            ndmi = image.normalizedDifference(["B8", "B11"]).rename("NDMI")

            vis_params = {
                "min": -0.5,
                "max": 0.5,
                "palette": ["#8B4513", "#D2691E", "#FFD700", "#90EE90", "#006400"],
            }

            tile_url = self._get_tile_url(ndmi, vis_params)

            return {
                "status": "success",
                "tile_url": tile_url,
                "metadata": {
                    "title": "Soil Moisture (NDMI)",
                    "description": "Normalized Difference Moisture Index showing soil/vegetation water content",
                    "source": "Copernicus Sentinel-2 SR Harmonized",
                    "date_range": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                    "resolution": "20m",
                },
                "legend": [
                    {
                        "color": "#8B4513",
                        "label": "Very Dry",
                        "value": "-0.5 to -0.2",
                    },
                    {
                        "color": "#D2691E",
                        "label": "Dry",
                        "value": "-0.2 to 0.0",
                    },
                    {
                        "color": "#FFD700",
                        "label": "Moderate",
                        "value": "0.0 to 0.2",
                    },
                    {
                        "color": "#90EE90",
                        "label": "Good Moisture",
                        "value": "0.2 to 0.4",
                    },
                    {
                        "color": "#006400",
                        "label": "High Moisture",
                        "value": "0.4 to 0.5+",
                    },
                ],
            }

        except Exception as e:
            self.logger.error(f"Error getting soil moisture tiles: {str(e)}")
            return {"status": "error", "message": str(e)}

    # Legacy method for polygon-based analysis (kept for compatibility)
    def analyze_soil_moisture(self, coordinates: List[List[float]]) -> Dict[str, Any]:
        """
        Analyze soil moisture and irrigation status.

        Uses NDVI + NDWI + Land Surface Temperature as proxies
        since SMAP has coarse resolution (9km).

        Args:
            coordinates: List of [lng, lat] coordinates defining the area

        Returns:
            Dictionary with soil moisture analysis
        """
        self._ensure_initialized()

        try:
            polygon = ee.Geometry.Polygon([coordinates])
            total_area = polygon.area().getInfo()
            scale = 30  # Use 30m for this analysis

            # Date range - last 3 months
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            # Get Sentinel-2 for vegetation indices
            s2_collection = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(polygon)
                .filterDate(start_str, end_str)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            )

            s2_count = s2_collection.size().getInfo()
            if s2_count == 0:
                return {
                    "status": "no_data",
                    "message": "No satellite imagery available",
                }

            s2 = s2_collection.median()

            # Calculate indices
            ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
            ndwi = s2.normalizedDifference(["B3", "B8"]).rename("NDWI")

            # NDMI (Normalized Difference Moisture Index) - better for soil moisture
            ndmi = s2.normalizedDifference(["B8", "B11"]).rename("NDMI")

            # Get MODIS Land Surface Temperature
            lst_collection = (
                ee.ImageCollection("MODIS/061/MOD11A2")
                .filterBounds(polygon)
                .filterDate(start_str, end_str)
            )

            lst_count = lst_collection.size().getInfo()
            has_lst = lst_count > 0

            if has_lst:
                lst = lst_collection.mean().select("LST_Day_1km")
                # Convert to Celsius
                lst_celsius = lst.multiply(0.02).subtract(273.15).rename("LST")
            else:
                lst_celsius = ee.Image.constant(25).rename("LST")

            # Create moisture proxy index
            # Higher NDMI + Higher NDVI + Lower LST = More moisture
            # Normalize each component to 0-1 range
            ndmi_norm = ndmi.add(1).divide(2)  # -1 to 1 -> 0 to 1
            ndvi_norm = ndvi.clamp(0, 1)
            lst_norm = (
                lst_celsius.subtract(50).multiply(-1).divide(30).clamp(0, 1)
            )  # Invert: cooler = higher

            # Weighted moisture index
            moisture_index = (
                ndmi_norm.multiply(0.5)
                .add(ndvi_norm.multiply(0.3))
                .add(lst_norm.multiply(0.2))
            ).rename("moisture_index")

            # Classify irrigation status
            # 0 = Dry, 1 = Moderate, 2 = Well Irrigated
            irrigation_class = (
                ee.Image(0)
                .where(moisture_index.gt(0.3), 1)  # Moderate
                .where(moisture_index.gt(0.5), 2)  # Well irrigated
                .rename("irrigation")
            )

            # Detect water stress (low NDVI + low moisture)
            stress = ndvi.lt(0.3).And(moisture_index.lt(0.3)).rename("stress")

            # Get statistics
            moisture_stats = moisture_index.reduceRegion(
                reducer=ee.Reducer.mean()
                .combine(ee.Reducer.minMax(), sharedInputs=True)
                .combine(ee.Reducer.stdDev(), sharedInputs=True),
                geometry=polygon,
                scale=scale,
                maxPixels=1e9,
            ).getInfo()

            # Get class distribution
            class_histogram = (
                irrigation_class.reduceRegion(
                    reducer=ee.Reducer.frequencyHistogram(),
                    geometry=polygon,
                    scale=scale,
                    maxPixels=1e9,
                )
                .get("irrigation")
                .getInfo()
                or {}
            )

            total_pixels = sum(class_histogram.values()) or 1
            dry_pct = (class_histogram.get("0", 0) / total_pixels) * 100
            moderate_pct = (class_histogram.get("1", 0) / total_pixels) * 100
            irrigated_pct = (class_histogram.get("2", 0) / total_pixels) * 100

            # Get stress percentage
            stress_sum = (
                stress.reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=polygon,
                    scale=scale,
                    maxPixels=1e9,
                )
                .get("stress")
                .getInfo()
                or 0
            )
            stress_pct = (stress_sum / total_pixels) * 100

            # Generate tile URL for moisture index
            vis_params = {
                "min": 0,
                "max": 1,
                "palette": ["#8B4513", "#D2691E", "#FFD700", "#90EE90", "#006400"],
            }
            tile_url = self._get_tile_url(moisture_index.clip(polygon), vis_params)

            # Determine overall status
            if irrigated_pct > 50:
                overall_status = "Well Irrigated"
            elif moderate_pct > 40:
                overall_status = "Moderately Irrigated"
            elif dry_pct > 50:
                overall_status = "Dry / Under-irrigated"
            else:
                overall_status = "Mixed Conditions"

            return {
                "status": "success",
                "total_area_hectares": round(total_area / 10000, 2),
                "overall_status": overall_status,
                "moisture_index": {
                    "mean": round(moisture_stats.get("moisture_index_mean", 0) or 0, 4),
                    "min": round(moisture_stats.get("moisture_index_min", 0) or 0, 4),
                    "max": round(moisture_stats.get("moisture_index_max", 0) or 0, 4),
                },
                "irrigation_distribution": {
                    "dry": {"percentage": round(dry_pct, 1), "color": "#8B4513"},
                    "moderate": {
                        "percentage": round(moderate_pct, 1),
                        "color": "#FFD700",
                    },
                    "well_irrigated": {
                        "percentage": round(irrigated_pct, 1),
                        "color": "#006400",
                    },
                },
                "water_stress": {
                    "percentage": round(stress_pct, 1),
                    "severity": "High"
                    if stress_pct > 30
                    else "Moderate"
                    if stress_pct > 15
                    else "Low",
                },
                "tile_url": tile_url,
                "legend": [
                    {"color": "#8B4513", "label": "Dry"},
                    {"color": "#D2691E", "label": "Low Moisture"},
                    {"color": "#FFD700", "label": "Moderate"},
                    {"color": "#90EE90", "label": "Good Moisture"},
                    {"color": "#006400", "label": "Well Irrigated"},
                ],
                "date_range": {"start": start_str, "end": end_str},
                "has_lst_data": has_lst,
            }

        except Exception as e:
            self.logger.error(f"Error analyzing soil moisture: {str(e)}")
            return {"status": "error", "message": str(e)}

    # ========================================
    # 4. ACTIVE FIRES / THERMAL ANOMALIES (Global Layer using FIRMS)
    # ========================================
    def get_active_fires_tiles(self) -> Dict[str, Any]:
        """
        Get global active fire/thermal anomaly hotspots using FIRMS data.
        FIRMS = Fire Information for Resource Management System.

        Returns:
            Dictionary with tile URL and metadata for fire hotspots
        """
        self._ensure_initialized()

        try:
            # Use last 7 days for recent fire activity
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)

            # FIRMS dataset - thermal anomalies detected by MODIS/VIIRS
            firms = (
                ee.ImageCollection("FIRMS")
                .filterDate(
                    start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
                )
                .select(["T21"])  # Brightness temperature (fire intensity)
            )

            # Get count of fire pixels
            fire_count = firms.count().rename("fire_count")

            # Get max brightness temperature (fire intensity indicator)
            fire_intensity = firms.max().rename("fire_intensity")

            # Create a composite showing fire locations
            # Higher values = more intense/persistent fires
            fire_hotspots = (
                fire_count.multiply(10)
                .add(fire_intensity.subtract(300).multiply(0.1))
                .rename("fire_hotspots")
            )

            # Visualization for fire hotspots
            vis_params = {
                "min": 0,
                "max": 100,
                "palette": ["#FFFF00", "#FFA500", "#FF4500", "#FF0000", "#8B0000"],
            }

            tile_url = self._get_tile_url(fire_hotspots, vis_params)

            return {
                "status": "success",
                "tile_url": tile_url,
                "metadata": {
                    "title": "Active Fires (FIRMS)",
                    "description": "Thermal anomalies and active fire hotspots detected by MODIS/VIIRS satellites",
                    "source": "NASA FIRMS (Fire Information for Resource Management System)",
                    "date_range": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                    "resolution": "1km",
                    "update_frequency": "Near real-time (updated every few hours)",
                },
                "legend": [
                    {
                        "color": "#FFFF00",
                        "label": "Low activity",
                        "value": "Thermal anomaly detected",
                    },
                    {
                        "color": "#FFA500",
                        "label": "Moderate",
                        "value": "Active fire likely",
                    },
                    {
                        "color": "#FF4500",
                        "label": "High",
                        "value": "Confirmed active fire",
                    },
                    {
                        "color": "#FF0000",
                        "label": "Very High",
                        "value": "Intense fire activity",
                    },
                    {
                        "color": "#8B0000",
                        "label": "Extreme",
                        "value": "Major fire event",
                    },
                ],
            }

        except Exception as e:
            self.logger.error(f"Error getting active fires tiles: {str(e)}")
            return {"status": "error", "message": str(e)}

    # Legacy method for polygon-based fire risk analysis (kept for compatibility)
    def analyze_fire_risk(self, coordinates: List[List[float]]) -> Dict[str, Any]:
        """
        Analyze fire risk based on temperature, dry vegetation, and slope.

        Args:
            coordinates: List of [lng, lat] coordinates defining the area

        Returns:
            Dictionary with fire risk analysis
        """
        self._ensure_initialized()

        try:
            polygon = ee.Geometry.Polygon([coordinates])
            total_area = polygon.area().getInfo()
            scale = 30

            # Date range
            end_date = datetime.now()
            start_date = end_date - timedelta(
                days=30
            )  # Last month for recent conditions
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            # Get MODIS Land Surface Temperature
            lst_collection = (
                ee.ImageCollection("MODIS/061/MOD11A2")
                .filterBounds(polygon)
                .filterDate(start_str, end_str)
            )

            lst_count = lst_collection.size().getInfo()
            if lst_count > 0:
                lst = lst_collection.mean().select("LST_Day_1km")
                temp_celsius = lst.multiply(0.02).subtract(273.15)
            else:
                # Fallback to annual average
                lst = (
                    ee.ImageCollection("MODIS/061/MOD11A2")
                    .filterBounds(polygon)
                    .filterDate(f"{end_date.year}-01-01", end_str)
                    .mean()
                    .select("LST_Day_1km")
                )
                temp_celsius = lst.multiply(0.02).subtract(273.15)

            # Get Sentinel-2 for NDVI
            s2_collection = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(polygon)
                .filterDate(start_str, end_str)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            )

            s2_count = s2_collection.size().getInfo()
            if s2_count == 0:
                # Try longer date range
                s2_collection = (
                    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                    .filterBounds(polygon)
                    .filterDate(start_date - timedelta(days=60), end_str)
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
                )
                s2_count = s2_collection.size().getInfo()

            if s2_count == 0:
                return {"status": "no_data", "message": "No imagery available"}

            s2 = s2_collection.median()
            ndvi = s2.normalizedDifference(["B8", "B4"])

            # Dry vegetation indicator (low NDVI)
            dry_veg = ndvi.lt(0.3).rename("dry_veg")

            # NDMI for moisture
            ndmi = s2.normalizedDifference(["B8", "B11"])
            dry_ndmi = ndmi.lt(0).rename("dry_ndmi")

            # Get slope from SRTM
            dem = ee.Image("USGS/SRTMGL1_003")
            slope = ee.Terrain.slope(dem)

            # Normalize components for risk calculation
            # Temperature: 20-50°C range -> 0-1
            temp_norm = temp_celsius.subtract(20).divide(30).clamp(0, 1)

            # Dry vegetation: binary 0/1
            dry_veg_norm = dry_veg.unmask(0)

            # Dry NDMI: binary 0/1
            dry_ndmi_norm = dry_ndmi.unmask(0)

            # Slope: 0-45 degrees -> 0-1
            slope_norm = slope.divide(45).clamp(0, 1)

            # Calculate fire risk score (weighted)
            # Temperature: 40%, Dry vegetation: 25%, Low moisture: 20%, Slope: 15%
            fire_risk = (
                temp_norm.multiply(0.40)
                .add(dry_veg_norm.multiply(0.25))
                .add(dry_ndmi_norm.multiply(0.20))
                .add(slope_norm.multiply(0.15))
            ).rename("fire_risk")

            # Classify risk levels
            # 0 = Low, 1 = Moderate, 2 = High, 3 = Extreme
            risk_class = (
                ee.Image(0)
                .where(fire_risk.gt(0.25), 1)
                .where(fire_risk.gt(0.50), 2)
                .where(fire_risk.gt(0.75), 3)
                .rename("risk_class")
            )

            # Get statistics
            risk_stats = fire_risk.reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    ee.Reducer.minMax(), sharedInputs=True
                ),
                geometry=polygon,
                scale=scale,
                maxPixels=1e9,
            ).getInfo()

            temp_stats = temp_celsius.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=polygon,
                scale=1000,  # MODIS resolution
                maxPixels=1e9,
            ).getInfo()

            # Get risk class distribution
            class_histogram = (
                risk_class.reduceRegion(
                    reducer=ee.Reducer.frequencyHistogram(),
                    geometry=polygon,
                    scale=scale,
                    maxPixels=1e9,
                )
                .get("risk_class")
                .getInfo()
                or {}
            )

            total_pixels = sum(class_histogram.values()) or 1
            low_pct = (class_histogram.get("0", 0) / total_pixels) * 100
            moderate_pct = (class_histogram.get("1", 0) / total_pixels) * 100
            high_pct = (class_histogram.get("2", 0) / total_pixels) * 100
            extreme_pct = (class_histogram.get("3", 0) / total_pixels) * 100

            # Get dry vegetation percentage
            dry_veg_sum = (
                dry_veg.reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=polygon,
                    scale=scale,
                    maxPixels=1e9,
                )
                .get("dry_veg")
                .getInfo()
                or 0
            )
            dry_veg_pct = (dry_veg_sum / total_pixels) * 100

            # Generate tile URL
            vis_params = {
                "min": 0,
                "max": 3,
                "palette": ["#4CAF50", "#FFEB3B", "#FF9800", "#F44336"],
            }
            tile_url = self._get_tile_url(risk_class.clip(polygon), vis_params)

            # Determine overall risk
            mean_risk = risk_stats.get("fire_risk_mean", 0) or 0
            if mean_risk > 0.75:
                overall_risk = "Extreme"
            elif mean_risk > 0.50:
                overall_risk = "High"
            elif mean_risk > 0.25:
                overall_risk = "Moderate"
            else:
                overall_risk = "Low"

            return {
                "status": "success",
                "total_area_hectares": round(total_area / 10000, 2),
                "overall_risk": overall_risk,
                "risk_score": round(mean_risk * 100, 1),
                "risk_distribution": {
                    "low": {"percentage": round(low_pct, 1), "color": "#4CAF50"},
                    "moderate": {
                        "percentage": round(moderate_pct, 1),
                        "color": "#FFEB3B",
                    },
                    "high": {"percentage": round(high_pct, 1), "color": "#FF9800"},
                    "extreme": {
                        "percentage": round(extreme_pct, 1),
                        "color": "#F44336",
                    },
                },
                "factors": {
                    "temperature": {
                        "mean_celsius": round(temp_stats.get("LST_Day_1km", 0) or 0, 1),
                    },
                    "dry_vegetation": {
                        "percentage": round(dry_veg_pct, 1),
                    },
                },
                "tile_url": tile_url,
                "legend": [
                    {"color": "#4CAF50", "label": "Low Risk"},
                    {"color": "#FFEB3B", "label": "Moderate Risk"},
                    {"color": "#FF9800", "label": "High Risk"},
                    {"color": "#F44336", "label": "Extreme Risk"},
                ],
                "date_range": {"start": start_str, "end": end_str},
            }

        except Exception as e:
            self.logger.error(f"Error analyzing fire risk: {str(e)}")
            return {"status": "error", "message": str(e)}

    # ========================================
    # 5. PLANTATION SUITABILITY
    # ========================================
    def analyze_plantation_suitability(
        self, coordinates: List[List[float]]
    ) -> Dict[str, Any]:
        """
        Analyze suitability for tree plantation.

        Considers: slope, soil moisture, built-up mask, water mask

        Args:
            coordinates: List of [lng, lat] coordinates defining the area

        Returns:
            Dictionary with plantation suitability analysis
        """
        self._ensure_initialized()

        try:
            polygon = ee.Geometry.Polygon([coordinates])
            total_area = polygon.area().getInfo()
            scale = 30

            # Date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            # Get Sentinel-2
            s2_collection = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(polygon)
                .filterDate(start_str, end_str)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            )

            s2_count = s2_collection.size().getInfo()
            if s2_count == 0:
                return {"status": "no_data", "message": "No imagery available"}

            s2 = s2_collection.median()

            # Calculate indices
            ndvi = s2.normalizedDifference(["B8", "B4"])
            ndwi = s2.normalizedDifference(["B3", "B8"])
            ndbi = s2.normalizedDifference(["B11", "B8"])
            ndmi = s2.normalizedDifference(["B8", "B11"])

            # Get slope from SRTM
            dem = ee.Image("USGS/SRTMGL1_003")
            slope = ee.Terrain.slope(dem)

            # Create masks
            # Built-up mask (NDBI > 0.2)
            built_mask = ndbi.gt(0.2)

            # Water mask (NDWI > 0.3)
            water_mask = ndwi.gt(0.3)

            # Existing forest mask (high NDVI)
            forest_mask = ndvi.gt(0.6)

            # Suitability criteria
            # Good slope (< 15 degrees)
            good_slope = slope.lt(15)

            # Good moisture (NDMI > -0.2)
            good_moisture = ndmi.gt(-0.2)

            # Not built-up
            not_built = built_mask.Not()

            # Not water
            not_water = water_mask.Not()

            # Not already forest
            not_forest = forest_mask.Not()

            # Calculate suitability score (0-100)
            # Each criterion contributes to the score
            suitability = (
                good_slope.multiply(30)
                .add(good_moisture.multiply(25))
                .add(not_built.multiply(20))
                .add(not_water.multiply(15))
                .add(not_forest.multiply(10))
            ).rename("suitability")

            # Binary suitable mask (score > 60)
            suitable = suitability.gt(60).rename("suitable")

            # Get statistics
            suit_stats = suitability.reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    ee.Reducer.minMax(), sharedInputs=True
                ),
                geometry=polygon,
                scale=scale,
                maxPixels=1e9,
            ).getInfo()

            # Get suitable area
            suitable_pixels = (
                suitable.reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=polygon,
                    scale=scale,
                    maxPixels=1e9,
                )
                .get("suitable")
                .getInfo()
                or 0
            )

            total_pixels = (
                suitable.unmask(0)
                .reduceRegion(
                    reducer=ee.Reducer.count(),
                    geometry=polygon,
                    scale=scale,
                    maxPixels=1e9,
                )
                .get("suitable")
                .getInfo()
                or 1
            )

            suitable_pct = (suitable_pixels / total_pixels) * 100
            suitable_area_ha = (suitable_pct / 100) * (total_area / 10000)

            # Get exclusion reasons
            built_pixels = (
                built_mask.reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=polygon,
                    scale=scale,
                    maxPixels=1e9,
                )
                .get("nd")
                .getInfo()
                or 0
            )

            water_pixels = (
                water_mask.reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=polygon,
                    scale=scale,
                    maxPixels=1e9,
                )
                .get("nd")
                .getInfo()
                or 0
            )

            steep_pixels = (
                slope.gt(15)
                .reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=polygon,
                    scale=scale,
                    maxPixels=1e9,
                )
                .get("slope")
                .getInfo()
                or 0
            )

            forest_pixels = (
                forest_mask.reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=polygon,
                    scale=scale,
                    maxPixels=1e9,
                )
                .get("nd")
                .getInfo()
                or 0
            )

            built_pct = (built_pixels / total_pixels) * 100
            water_pct = (water_pixels / total_pixels) * 100
            steep_pct = (steep_pixels / total_pixels) * 100
            forest_pct = (forest_pixels / total_pixels) * 100

            # Generate tile URL
            vis_params = {
                "min": 0,
                "max": 100,
                "palette": ["#F44336", "#FF9800", "#FFEB3B", "#8BC34A", "#4CAF50"],
            }
            tile_url = self._get_tile_url(suitability.clip(polygon), vis_params)

            # Determine overall suitability
            mean_suit = suit_stats.get("suitability_mean", 0) or 0
            if mean_suit > 70:
                overall = "Highly Suitable"
            elif mean_suit > 50:
                overall = "Moderately Suitable"
            elif mean_suit > 30:
                overall = "Marginally Suitable"
            else:
                overall = "Not Suitable"

            return {
                "status": "success",
                "total_area_hectares": round(total_area / 10000, 2),
                "overall_suitability": overall,
                "suitability_score": round(mean_suit, 1),
                "suitable_area": {
                    "percentage": round(suitable_pct, 1),
                    "hectares": round(suitable_area_ha, 2),
                },
                "exclusion_factors": {
                    "built_up": {
                        "percentage": round(built_pct, 1),
                        "reason": "Urban/developed area",
                    },
                    "water_bodies": {
                        "percentage": round(water_pct, 1),
                        "reason": "Water bodies",
                    },
                    "steep_slope": {
                        "percentage": round(steep_pct, 1),
                        "reason": "Slope > 15°",
                    },
                    "existing_forest": {
                        "percentage": round(forest_pct, 1),
                        "reason": "Already forested",
                    },
                },
                "tile_url": tile_url,
                "legend": [
                    {"color": "#F44336", "label": "Not Suitable (0-20)"},
                    {"color": "#FF9800", "label": "Poor (20-40)"},
                    {"color": "#FFEB3B", "label": "Moderate (40-60)"},
                    {"color": "#8BC34A", "label": "Good (60-80)"},
                    {"color": "#4CAF50", "label": "Excellent (80-100)"},
                ],
                "date_range": {"start": start_str, "end": end_str},
            }

        except Exception as e:
            self.logger.error(f"Error analyzing plantation suitability: {str(e)}")
            return {"status": "error", "message": str(e)}

    # ========================================
    # 6. COMPENSATORY PLANTATION PLANNER
    # ========================================
    def plan_compensatory_plantation(
        self,
        removal_coordinates: List[List[float]],
        search_coordinates: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        """
        Plan compensatory plantation for trees being removed.

        Args:
            removal_coordinates: Area where trees are being removed
            search_coordinates: Optional larger area to search for plantation sites
                              (defaults to 5km buffer around removal area)

        Returns:
            Dictionary with plantation plan
        """
        self._ensure_initialized()

        try:
            removal_polygon = ee.Geometry.Polygon([removal_coordinates])
            removal_area = removal_polygon.area().getInfo()
            scale = 30

            # Create search area (buffer if not specified)
            if search_coordinates:
                search_polygon = ee.Geometry.Polygon([search_coordinates])
            else:
                # 5km buffer around removal area
                search_polygon = removal_polygon.buffer(5000)

            search_area = search_polygon.area().getInfo()

            # Date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            # Get Sentinel-2
            s2_collection = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(search_polygon)
                .filterDate(start_str, end_str)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            )

            s2_count = s2_collection.size().getInfo()
            if s2_count == 0:
                return {"status": "no_data", "message": "No imagery available"}

            s2 = s2_collection.median()

            # Calculate indices
            ndvi = s2.normalizedDifference(["B8", "B4"])
            ndwi = s2.normalizedDifference(["B3", "B8"])
            ndbi = s2.normalizedDifference(["B11", "B8"])
            ndmi = s2.normalizedDifference(["B8", "B11"])

            # Detect forest in removal area
            forest = ndvi.gt(0.5)

            # Calculate forest area being lost
            forest_pixels = (
                forest.reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=removal_polygon,
                    scale=scale,
                    maxPixels=1e9,
                )
                .get("nd")
                .getInfo()
                or 0
            )

            total_removal_pixels = (
                forest.unmask(0)
                .reduceRegion(
                    reducer=ee.Reducer.count(),
                    geometry=removal_polygon,
                    scale=scale,
                    maxPixels=1e9,
                )
                .get("nd")
                .getInfo()
                or 1
            )

            forest_pct = (forest_pixels / total_removal_pixels) * 100
            lost_forest_area_ha = (forest_pct / 100) * (removal_area / 10000)

            # Get slope
            dem = ee.Image("USGS/SRTMGL1_003")
            slope = ee.Terrain.slope(dem)

            # Create suitability map for search area
            built_mask = ndbi.gt(0.2)
            water_mask = ndwi.gt(0.3)
            existing_forest = ndvi.gt(0.5)

            # Suitable = gentle slope, not built, not water, not already forest
            suitable = (
                slope.lt(15)
                .And(built_mask.Not())
                .And(water_mask.Not())
                .And(existing_forest.Not())
                .And(ndmi.gt(-0.3))  # Some moisture
            )

            # Distance from removal area (priority: closer is better)
            distance = suitable.fastDistanceTransform().sqrt().multiply(scale)

            # Create priority score: suitable * (1 / distance)
            # Higher score = more suitable AND closer
            priority = suitable.multiply(
                distance.add(100).pow(-1).multiply(10000)  # Normalize
            ).rename("priority")

            # Get suitable area in search region
            suitable_pixels = (
                suitable.reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=search_polygon,
                    scale=scale,
                    maxPixels=1e9,
                )
                .get("slope")
                .getInfo()
                or 0
            )

            total_search_pixels = (
                suitable.unmask(0)
                .reduceRegion(
                    reducer=ee.Reducer.count(),
                    geometry=search_polygon,
                    scale=scale,
                    maxPixels=1e9,
                )
                .get("slope")
                .getInfo()
                or 1
            )

            suitable_area_ha = (suitable_pixels / total_search_pixels) * (
                search_area / 10000
            )

            # Generate tile URLs
            # Removal area forest
            forest_vis = {"min": 0, "max": 1, "palette": ["#FFFFFF00", "#228B22"]}
            removal_tile = self._get_tile_url(forest.clip(removal_polygon), forest_vis)

            # Priority areas
            priority_vis = {
                "min": 0,
                "max": 100,
                "palette": ["#FFFFFF00", "#90EE90", "#228B22"],
            }
            priority_tile = self._get_tile_url(
                priority.clip(search_polygon), priority_vis
            )

            # Determine if enough suitable area exists
            required_area_ha = lost_forest_area_ha * 2  # Usually 2:1 compensation ratio
            can_compensate = suitable_area_ha >= required_area_ha

            return {
                "status": "success",
                "removal_area": {
                    "total_hectares": round(removal_area / 10000, 2),
                    "forest_percentage": round(forest_pct, 1),
                    "forest_loss_hectares": round(lost_forest_area_ha, 2),
                },
                "compensation_required": {
                    "ratio": "2:1",
                    "required_hectares": round(required_area_ha, 2),
                },
                "search_area": {
                    "total_hectares": round(search_area / 10000, 2),
                    "suitable_hectares": round(suitable_area_ha, 2),
                    "suitable_percentage": round(
                        (suitable_pixels / total_search_pixels) * 100, 1
                    ),
                },
                "feasibility": {
                    "can_compensate": can_compensate,
                    "message": (
                        f"Sufficient suitable land available ({round(suitable_area_ha, 1)} ha)"
                        if can_compensate
                        else f"Insufficient suitable land. Need {round(required_area_ha, 1)} ha, only {round(suitable_area_ha, 1)} ha available"
                    ),
                },
                "tile_urls": {
                    "removal_forest": removal_tile,
                    "priority_areas": priority_tile,
                },
                "legend": [
                    {"color": "#228B22", "label": "Existing Forest / High Priority"},
                    {"color": "#90EE90", "label": "Medium Priority"},
                    {"color": "#FFFFFF", "label": "Not Suitable"},
                ],
                "date_range": {"start": start_str, "end": end_str},
            }

        except Exception as e:
            self.logger.error(f"Error planning compensatory plantation: {str(e)}")
            return {"status": "error", "message": str(e)}

    # ========================================
    # 7. VEGETATION TREND ANALYSIS (10 years)
    # ========================================
    def estimate_tree_growth(self, coordinates: List[List[float]]) -> Dict[str, Any]:
        """
        Analyze vegetation NDVI trends over 10 years.

        Uses NDVI time series to show historical vegetation changes with charts.

        Args:
            coordinates: List of [lng, lat] coordinates defining the area

        Returns:
            Dictionary with 10-year NDVI trend analysis
        """
        self._ensure_initialized()

        try:
            polygon = ee.Geometry.Polygon([coordinates])
            total_area = polygon.area().getInfo()
            scale = 30

            current_year = datetime.now().year

            # Get NDVI for each year (last 10 years)
            years = list(range(current_year - 9, current_year + 1))
            yearly_ndvi = {}

            for year in years:
                start_str = f"{year}-01-01"
                end_str = f"{year}-12-31"

                collection = (
                    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                    .filterBounds(polygon)
                    .filterDate(start_str, end_str)
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
                )

                count = collection.size().getInfo()
                if count > 0:
                    median = collection.median()
                    ndvi = median.normalizedDifference(["B8", "B4"])

                    mean_ndvi = (
                        ndvi.reduceRegion(
                            reducer=ee.Reducer.mean(),
                            geometry=polygon,
                            scale=scale,
                            maxPixels=1e9,
                        )
                        .get("nd")
                        .getInfo()
                        or 0
                    )
                    yearly_ndvi[year] = mean_ndvi

            if len(yearly_ndvi) < 2:
                return {
                    "status": "insufficient_data",
                    "message": "Not enough historical data to estimate growth",
                }

            # Calculate linear trend
            years_list = sorted(yearly_ndvi.keys())
            ndvi_values = [yearly_ndvi[y] for y in years_list]

            # Simple linear regression
            n = len(years_list)
            sum_x = sum(years_list)
            sum_y = sum(ndvi_values)
            sum_xy = sum(x * y for x, y in zip(years_list, ndvi_values))
            sum_x2 = sum(x * x for x in years_list)

            # Slope (annual NDVI change)
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            intercept = (sum_y - slope * sum_x) / n

            # Current NDVI (most recent)
            current_ndvi = ndvi_values[-1]

            # Determine growth status
            if slope > 0.02:
                growth_status = "Rapid Growth"
            elif slope > 0.005:
                growth_status = "Moderate Growth"
            elif slope > 0:
                growth_status = "Slow Growth"
            elif slope > -0.005:
                growth_status = "Stable"
            else:
                growth_status = "Declining"

            # Get current land cover assessment
            if current_ndvi < 0.2:
                current_status = "Bare/Degraded Land"
            elif current_ndvi < 0.4:
                current_status = "Sparse Vegetation"
            elif current_ndvi < 0.6:
                current_status = "Moderate Vegetation"
            else:
                current_status = "Dense Vegetation/Forest"

            # Create chart data format for frontend visualization
            sorted_years = sorted(yearly_ndvi.keys())
            chart_data = [
                {"year": year, "ndvi": round(yearly_ndvi[year], 4), "label": str(year)}
                for year in sorted_years
            ]

            # Calculate additional statistics
            ndvi_values_list = [yearly_ndvi[y] for y in sorted_years]
            min_ndvi = min(ndvi_values_list)
            max_ndvi = max(ndvi_values_list)
            avg_ndvi = sum(ndvi_values_list) / len(ndvi_values_list)

            # Find best and worst years
            best_year = max(yearly_ndvi.keys(), key=lambda y: yearly_ndvi[y])
            worst_year = min(yearly_ndvi.keys(), key=lambda y: yearly_ndvi[y])

            # Calculate percentage change over the period
            first_ndvi = ndvi_values_list[0]
            last_ndvi = ndvi_values_list[-1]
            total_change_percent = (
                ((last_ndvi - first_ndvi) / first_ndvi * 100) if first_ndvi > 0 else 0
            )

            return {
                "status": "success",
                "total_area_hectares": round(total_area / 10000, 2),
                "current_status": current_status,
                "current_ndvi": round(current_ndvi, 4),
                "growth_analysis": {
                    "status": growth_status,
                    "annual_ndvi_change": round(slope, 5),
                    "trend": "Improving"
                    if slope > 0
                    else "Declining"
                    if slope < 0
                    else "Stable",
                },
                "historical_ndvi": {
                    str(k): round(v, 4) for k, v in sorted(yearly_ndvi.items())
                },
                "years_analyzed": len(yearly_ndvi),
                # New fields for chart visualization
                "chart_data": chart_data,
                "statistics": {
                    "min_ndvi": round(min_ndvi, 4),
                    "max_ndvi": round(max_ndvi, 4),
                    "avg_ndvi": round(avg_ndvi, 4),
                    "best_year": best_year,
                    "worst_year": worst_year,
                    "total_change_percent": round(total_change_percent, 2),
                    "start_year": sorted_years[0],
                    "end_year": sorted_years[-1],
                },
            }

        except Exception as e:
            self.logger.error(f"Error estimating tree growth: {str(e)}")
            return {"status": "error", "message": str(e)}

    # ========================================
    # 8. SPECIES RECOMMENDATION ENGINE
    # ========================================
    def recommend_species(self, coordinates: List[List[float]]) -> Dict[str, Any]:
        """
        Recommend plant species based on environmental conditions.

        Args:
            coordinates: List of [lng, lat] coordinates defining the area

        Returns:
            Dictionary with species recommendations
        """
        self._ensure_initialized()

        try:
            polygon = ee.Geometry.Polygon([coordinates])
            total_area = polygon.area().getInfo()
            scale = 1000  # Coarser scale for regional data

            # Get centroid for regional assessment
            centroid = polygon.centroid()

            # Date range
            end_date = datetime.now()
            current_year = end_date.year

            # === RAINFALL DATA (CHIRPS) ===
            chirps = (
                ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                .filterBounds(polygon)
                .filterDate(f"{current_year - 1}-01-01", f"{current_year}-01-01")
            )

            annual_rainfall = chirps.sum().rename("rainfall")
            rainfall_mm = (
                annual_rainfall.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=polygon,
                    scale=scale,
                    maxPixels=1e9,
                )
                .get("rainfall")
                .getInfo()
                or 1000  # Default
            )

            # === TEMPERATURE DATA (MODIS LST) ===
            lst = (
                ee.ImageCollection("MODIS/061/MOD11A2")
                .filterBounds(polygon)
                .filterDate(f"{current_year - 1}-01-01", f"{current_year}-01-01")
                .mean()
                .select("LST_Day_1km")
            )

            temp_kelvin = (
                lst.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=polygon,
                    scale=scale,
                    maxPixels=1e9,
                )
                .get("LST_Day_1km")
                .getInfo()
                or 300
            )
            temp_celsius = temp_kelvin * 0.02 - 273.15

            # === SLOPE DATA ===
            dem = ee.Image("USGS/SRTMGL1_003")
            slope = ee.Terrain.slope(dem)

            slope_mean = (
                slope.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=polygon,
                    scale=30,
                    maxPixels=1e9,
                )
                .get("slope")
                .getInfo()
                or 5
            )

            # === SOIL MOISTURE PROXY (NDMI) ===
            s2_collection = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(polygon)
                .filterDate(f"{current_year}-01-01", end_date.strftime("%Y-%m-%d"))
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            )

            s2_count = s2_collection.size().getInfo()
            if s2_count > 0:
                s2 = s2_collection.median()
                ndmi = s2.normalizedDifference(["B8", "B11"])
                moisture = (
                    ndmi.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=polygon,
                        scale=30,
                        maxPixels=1e9,
                    )
                    .get("nd")
                    .getInfo()
                    or 0
                )
            else:
                moisture = 0

            # === SPECIES RECOMMENDATION RULES ===
            recommendations = []

            # Rule-based species matching
            # High rainfall + gentle slope
            if rainfall_mm > 1500 and slope_mean < 10:
                recommendations.append(
                    {
                        "species": "Teak (Tectona grandis)",
                        "confidence": 0.85,
                        "reason": "High rainfall and gentle slope ideal for teak",
                        "growth_rate": "Moderate",
                        "time_to_maturity": "25-30 years",
                    }
                )

            if rainfall_mm > 2000 and moisture > 0.2:
                recommendations.append(
                    {
                        "species": "Sal (Shorea robusta)",
                        "confidence": 0.80,
                        "reason": "Very high rainfall and good moisture",
                        "growth_rate": "Slow",
                        "time_to_maturity": "40-50 years",
                    }
                )

            # Moderate rainfall
            if 800 < rainfall_mm < 1500:
                recommendations.append(
                    {
                        "species": "Neem (Azadirachta indica)",
                        "confidence": 0.82,
                        "reason": "Adaptable to moderate rainfall conditions",
                        "growth_rate": "Fast",
                        "time_to_maturity": "10-15 years",
                    }
                )

            # Low rainfall / arid
            if rainfall_mm < 800:
                recommendations.append(
                    {
                        "species": "Acacia (Acacia nilotica)",
                        "confidence": 0.88,
                        "reason": "Drought-tolerant, ideal for low rainfall areas",
                        "growth_rate": "Fast",
                        "time_to_maturity": "8-12 years",
                    }
                )
                recommendations.append(
                    {
                        "species": "Prosopis (Prosopis juliflora)",
                        "confidence": 0.85,
                        "reason": "Extremely drought-tolerant",
                        "growth_rate": "Fast",
                        "time_to_maturity": "5-8 years",
                    }
                )

            # High moisture areas
            if moisture > 0.3:
                recommendations.append(
                    {
                        "species": "Bamboo (Bambusa spp.)",
                        "confidence": 0.80,
                        "reason": "High moisture availability suitable for bamboo",
                        "growth_rate": "Very Fast",
                        "time_to_maturity": "3-5 years",
                    }
                )

            # Coastal/high moisture
            if moisture > 0.4 and temp_celsius > 25:
                recommendations.append(
                    {
                        "species": "Mangrove (Rhizophora spp.)",
                        "confidence": 0.75,
                        "reason": "High moisture and warm temperature (verify coastal location)",
                        "growth_rate": "Moderate",
                        "time_to_maturity": "15-20 years",
                    }
                )

            # Fruit trees for moderate conditions
            if 20 < temp_celsius < 35 and rainfall_mm > 600:
                recommendations.append(
                    {
                        "species": "Mango (Mangifera indica)",
                        "confidence": 0.78,
                        "reason": "Suitable temperature and rainfall for fruit trees",
                        "growth_rate": "Moderate",
                        "time_to_maturity": "5-8 years for fruiting",
                    }
                )

            # Sort by confidence
            recommendations.sort(key=lambda x: x["confidence"], reverse=True)

            # Environmental profile
            profile = {
                "rainfall_mm": round(rainfall_mm, 0),
                "rainfall_category": (
                    "High (>1500mm)"
                    if rainfall_mm > 1500
                    else "Moderate (800-1500mm)"
                    if rainfall_mm > 800
                    else "Low (<800mm)"
                ),
                "temperature_celsius": round(temp_celsius, 1),
                "temperature_category": (
                    "Hot (>30°C)"
                    if temp_celsius > 30
                    else "Warm (20-30°C)"
                    if temp_celsius > 20
                    else "Cool (<20°C)"
                ),
                "slope_degrees": round(slope_mean, 1),
                "slope_category": (
                    "Steep (>15°)"
                    if slope_mean > 15
                    else "Moderate (5-15°)"
                    if slope_mean > 5
                    else "Gentle (<5°)"
                ),
                "moisture_index": round(moisture, 3),
                "moisture_category": (
                    "High" if moisture > 0.3 else "Moderate" if moisture > 0 else "Low"
                ),
            }

            return {
                "status": "success",
                "total_area_hectares": round(total_area / 10000, 2),
                "environmental_profile": profile,
                "recommendations": recommendations[:5],  # Top 5
                "note": "Recommendations are based on satellite-derived environmental data. Ground verification recommended.",
            }

        except Exception as e:
            self.logger.error(f"Error recommending species: {str(e)}")
            return {"status": "error", "message": str(e)}
