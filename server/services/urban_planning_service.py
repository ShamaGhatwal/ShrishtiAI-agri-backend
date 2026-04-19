"""
Urban Planning Service
Handles GEE-based urban planning analysis including:
1. Plot Measurement - Calculate polygon area
2. Road Width Measurement - Calculate line length
3. Land-Use Classification - Classify area using spectral indices
4. Built-Up Detection - Extract urban areas using multiple indices
5. Suitability Analysis - Score locations for building suitability
"""

import ee
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional


def _subtract_months(dt: datetime, months: int) -> datetime:
    """
    Safely subtract months from a datetime, handling edge cases like
    end-of-month dates that don't exist in earlier months.

    E.g., March 31 - 1 month = February 28/29 (not an error)
    """
    # Use timedelta to go back approximately the right number of days
    # 6 months ≈ 182 days
    return dt - timedelta(days=months * 30)


class UrbanPlanningService:
    """Service class for Urban Planning GEE operations"""

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
    # 1. PLOT MEASUREMENT
    # ========================================
    def calculate_plot_area(self, coordinates: List[List[float]]) -> Dict[str, Any]:
        """
        Calculate the area of a polygon defined by coordinates.

        Args:
            coordinates: List of [lng, lat] coordinates defining the polygon

        Returns:
            Dictionary with area in square meters and hectares
        """
        self._ensure_initialized()

        try:
            # Create polygon geometry from coordinates
            polygon = ee.Geometry.Polygon([coordinates])

            # Calculate area in square meters
            area_m2 = polygon.area().getInfo()

            # Convert to hectares
            area_hectares = area_m2 / 10000

            # Calculate perimeter
            perimeter_m = polygon.perimeter().getInfo()

            return {
                "status": "success",
                "area_m2": round(area_m2, 2),
                "area_hectares": round(area_hectares, 4),
                "area_acres": round(area_hectares * 2.471, 4),
                "area_km2": round(area_m2 / 1000000, 6),
                "perimeter_m": round(perimeter_m, 2),
                "perimeter_km": round(perimeter_m / 1000, 4),
                "coordinates": coordinates,
                "num_vertices": len(coordinates),
            }

        except Exception as e:
            self.logger.error(f"Error calculating plot area: {str(e)}")
            return {"status": "error", "message": str(e)}

    # ========================================
    # 2. ROAD WIDTH MEASUREMENT
    # ========================================
    def calculate_road_length(self, coordinates: List[List[float]]) -> Dict[str, Any]:
        """
        Calculate the length of a line/road defined by coordinates.

        Args:
            coordinates: List of [lng, lat] coordinates defining the line

        Returns:
            Dictionary with length in meters and kilometers
        """
        self._ensure_initialized()

        try:
            # Create LineString geometry from coordinates
            line = ee.Geometry.LineString(coordinates)

            # Calculate length in meters
            length_m = line.length().getInfo()

            # Convert to other units
            length_km = length_m / 1000
            length_miles = length_km * 0.621371
            length_feet = length_m * 3.28084

            return {
                "status": "success",
                "length_m": round(length_m, 2),
                "length_km": round(length_km, 4),
                "length_miles": round(length_miles, 4),
                "length_feet": round(length_feet, 2),
                "coordinates": coordinates,
                "num_points": len(coordinates),
            }

        except Exception as e:
            self.logger.error(f"Error calculating road length: {str(e)}")
            return {"status": "error", "message": str(e)}

    # ========================================
    # 3. LAND-USE CLASSIFICATION (IMPROVED)
    # ========================================
    def classify_land_use(
        self, coordinates: List[List[float]], date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Classify land use within a polygon using improved spectral indices.

        IMPROVED Classification rules with better thresholds:
        - Water: NDWI > 0.1 AND NDVI < 0.2 (water bodies)
        - Vegetation: NDVI > 0.25 (green areas)
        - Built-up/Urban: NDBI > -0.1 AND NDVI < 0.25 AND NDWI < 0.1
        - Bare land: Everything else

        Args:
            coordinates: List of [lng, lat] coordinates defining the area
            date: Optional date string (YYYY-MM-DD), defaults to recent

        Returns:
            Dictionary with land use classification percentages and tile URL for overlay
        """
        self._ensure_initialized()

        try:
            # Create polygon geometry
            polygon = ee.Geometry.Polygon([coordinates])

            # Set date range (last 6 months if not specified for better imagery)
            if date:
                end_date = datetime.strptime(date, "%Y-%m-%d")
            else:
                end_date = datetime.now()

            # Go back 6 months for better chance of cloud-free imagery
            start_date = _subtract_months(end_date, 6)

            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            # Get Sentinel-2 imagery with relaxed cloud filter
            collection = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(polygon)
                .filterDate(start_str, end_str)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            )

            count = collection.size().getInfo()
            if count == 0:
                # Try with even more relaxed cloud filter
                collection = (
                    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                    .filterBounds(polygon)
                    .filterDate(start_str, end_str)
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
                )
                count = collection.size().getInfo()

            if count == 0:
                return {
                    "status": "no_data",
                    "message": "No satellite imagery available for the specified area and date range",
                }

            # Create median composite
            image = collection.median()

            # Calculate spectral indices
            ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
            ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
            ndbi = image.normalizedDifference(["B11", "B8"]).rename("NDBI")

            # IMPROVED thresholds for more accurate classification
            # Water: positive NDWI and low NDVI
            water_mask = ndwi.gt(0.1).And(ndvi.lt(0.2))

            # Vegetation: moderate to high NDVI
            vegetation_mask = ndvi.gt(0.25).And(water_mask.Not())

            # Built-up: NDBI > -0.1 (lowered threshold), low vegetation, not water
            builtup_mask = ndbi.gt(-0.1).And(ndvi.lt(0.25)).And(water_mask.Not())

            # Bare land: everything else
            bare_mask = (
                vegetation_mask.Not().And(water_mask.Not()).And(builtup_mask.Not())
            )

            # Create classified image (for visualization)
            # 1 = Water (blue), 2 = Vegetation (green), 3 = Built-up (red), 4 = Bare (yellow)
            classified = (
                ee.Image(0)
                .where(water_mask, 1)
                .where(vegetation_mask, 2)
                .where(builtup_mask, 3)
                .where(bare_mask, 4)
                .rename("classification")
            )

            # Calculate area for each class
            scale = 10  # 10m resolution for Sentinel-2

            # Get total area
            total_area = polygon.area().getInfo()

            # Calculate pixel counts for each class using the classified image
            class_stats = (
                classified.reduceRegion(
                    reducer=ee.Reducer.frequencyHistogram(),
                    geometry=polygon,
                    scale=scale,
                    maxPixels=1e9,
                )
                .get("classification")
                .getInfo()
                or {}
            )

            # Parse pixel counts
            water_pixels = class_stats.get("1", 0)
            vegetation_pixels = class_stats.get("2", 0)
            builtup_pixels = class_stats.get("3", 0)
            bare_pixels = class_stats.get("4", 0)

            # Calculate total pixels
            total_pixels = (
                water_pixels + vegetation_pixels + builtup_pixels + bare_pixels
            )

            if total_pixels == 0:
                return {
                    "status": "error",
                    "message": "Could not calculate land use classification",
                }

            # Calculate percentages
            vegetation_pct = (vegetation_pixels / total_pixels) * 100
            water_pct = (water_pixels / total_pixels) * 100
            builtup_pct = (builtup_pixels / total_pixels) * 100
            bare_pct = (bare_pixels / total_pixels) * 100

            # Get mean index values
            mean_indices = (
                image.addBands([ndvi, ndwi, ndbi])
                .select(["NDVI", "NDWI", "NDBI"])
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=polygon,
                    scale=scale,
                    maxPixels=1e9,
                )
                .getInfo()
            )

            # Generate tile URL for the classification overlay
            vis_params = {
                "min": 1,
                "max": 4,
                "palette": [
                    "0000FF",
                    "00FF00",
                    "FF0000",
                    "FFFF00",
                ],  # Blue, Green, Red, Yellow
            }
            tile_url = self._get_tile_url(classified.clip(polygon), vis_params)

            return {
                "status": "success",
                "total_area_m2": round(total_area, 2),
                "total_area_hectares": round(total_area / 10000, 4),
                "classification": {
                    "vegetation": {
                        "percentage": round(vegetation_pct, 2),
                        "area_m2": round((vegetation_pct / 100) * total_area, 2),
                        "color": "#00FF00",
                    },
                    "water": {
                        "percentage": round(water_pct, 2),
                        "area_m2": round((water_pct / 100) * total_area, 2),
                        "color": "#0000FF",
                    },
                    "built_up": {
                        "percentage": round(builtup_pct, 2),
                        "area_m2": round((builtup_pct / 100) * total_area, 2),
                        "color": "#FF0000",
                    },
                    "bare_land": {
                        "percentage": round(bare_pct, 2),
                        "area_m2": round((bare_pct / 100) * total_area, 2),
                        "color": "#FFFF00",
                    },
                },
                "indices": {
                    "mean_ndvi": round(mean_indices.get("NDVI", 0) or 0, 4),
                    "mean_ndwi": round(mean_indices.get("NDWI", 0) or 0, 4),
                    "mean_ndbi": round(mean_indices.get("NDBI", 0) or 0, 4),
                },
                "tile_url": tile_url,
                "legend": {
                    "Water": "#0000FF",
                    "Vegetation": "#00FF00",
                    "Built-up": "#FF0000",
                    "Bare Land": "#FFFF00",
                },
                "date_range": {"start": start_str, "end": end_str},
                "images_used": count,
            }

        except Exception as e:
            self.logger.error(f"Error classifying land use: {str(e)}")
            return {"status": "error", "message": str(e)}

    # ========================================
    # 4. NDBI ANALYSIS (Pure NDBI Index Display)
    # ========================================
    def detect_built_up(
        self, coordinates: List[List[float]], date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze NDBI (Normalized Difference Built-up Index) for an area.

        Returns pure NDBI statistics and visualization without built-up classification.
        NDBI = (SWIR - NIR) / (SWIR + NIR)

        NDBI interpretation:
        - High positive (>0.2): Dense built-up areas
        - Moderate positive (0 to 0.2): Built-up/urban areas
        - Around zero (-0.1 to 0): Mixed/transitional areas
        - Negative (-0.2 to -0.1): Vegetation
        - Very negative (<-0.2): Water bodies

        Args:
            coordinates: List of [lng, lat] coordinates defining the area
            date: Optional date string (YYYY-MM-DD)

        Returns:
            Dictionary with NDBI statistics and tile URL for visualization
        """
        self._ensure_initialized()

        try:
            polygon = ee.Geometry.Polygon([coordinates])

            # Set date range (6 months for better imagery)
            if date:
                end_date = datetime.strptime(date, "%Y-%m-%d")
            else:
                end_date = datetime.now()

            start_date = _subtract_months(end_date, 6)

            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            # Get Sentinel-2 imagery
            collection = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(polygon)
                .filterDate(start_str, end_str)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            )

            count = collection.size().getInfo()
            if count == 0:
                # Try with more relaxed cloud filter
                collection = (
                    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                    .filterBounds(polygon)
                    .filterDate(start_str, end_str)
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
                )
                count = collection.size().getInfo()

            if count == 0:
                return {
                    "status": "no_data",
                    "message": "No satellite imagery available for the specified area",
                }

            # Create median composite
            image = collection.median()

            # Calculate NDBI: (SWIR - NIR) / (SWIR + NIR)
            # Sentinel-2: B11 = SWIR, B8 = NIR
            ndbi = image.normalizedDifference(["B11", "B8"]).rename("NDBI")

            # Calculate statistics
            scale = 10
            total_area = polygon.area().getInfo()

            # Get NDBI statistics
            ndbi_stats = ndbi.reduceRegion(
                reducer=ee.Reducer.mean()
                .combine(ee.Reducer.minMax(), sharedInputs=True)
                .combine(ee.Reducer.stdDev(), sharedInputs=True),
                geometry=polygon,
                scale=scale,
                maxPixels=1e9,
            ).getInfo()

            # Generate tile URL for NDBI visualization
            # Color palette: Blue (water) -> Green (vegetation) -> Yellow -> Orange -> Red (built-up)
            vis_params = {
                "min": -0.5,
                "max": 0.5,
                "palette": ["0000FF", "00FFFF", "00FF00", "FFFF00", "FFA500", "FF0000"],
            }
            tile_url = self._get_tile_url(ndbi.clip(polygon), vis_params)

            return {
                "status": "success",
                "total_area_m2": round(total_area, 2),
                "total_area_hectares": round(total_area / 10000, 4),
                "ndbi_statistics": {
                    "mean": round(ndbi_stats.get("NDBI_mean", 0) or 0, 4),
                    "min": round(ndbi_stats.get("NDBI_min", 0) or 0, 4),
                    "max": round(ndbi_stats.get("NDBI_max", 0) or 0, 4),
                    "std_dev": round(ndbi_stats.get("NDBI_stdDev", 0) or 0, 4),
                },
                "tile_url": tile_url,
                "legend": {
                    "Blue (-0.5)": "Water bodies",
                    "Cyan (-0.3)": "Low reflectance",
                    "Green (-0.1)": "Vegetation",
                    "Yellow (0)": "Mixed/Bare",
                    "Orange (0.2)": "Built-up",
                    "Red (0.5)": "Dense built-up",
                },
                "date_range": {"start": start_str, "end": end_str},
                "images_used": count,
            }

        except Exception as e:
            self.logger.error(f"Error analyzing NDBI: {str(e)}")
            return {"status": "error", "message": str(e)}

    # ========================================
    # 5. SUITABILITY ANALYSIS (IMPROVED WATER DETECTION)
    # ========================================
    def analyze_suitability(
        self, coordinates: List[List[float]], date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze building suitability with IMPROVED water body detection.

        Uses multiple water detection methods:
        1. JRC Global Surface Water dataset (most reliable)
        2. NDWI and MNDWI from Sentinel-2
        3. AWEInsh (Automated Water Extraction Index)

        Scoring methodology:
        - Slope: Lower slopes are better (flat land preferred)
        - Water: Multi-source detection for accurate flood risk
        - NDBI: Existing built-up areas score higher (infrastructure nearby)
        - NDVI: Lower vegetation is preferred (less clearing needed)

        Args:
            coordinates: List of [lng, lat] coordinates defining the area
            date: Optional date string for satellite imagery

        Returns:
            Dictionary with suitability scores and analysis
        """
        self._ensure_initialized()

        try:
            polygon = ee.Geometry.Polygon([coordinates])
            total_area = polygon.area().getInfo()
            scale = 10  # Use 10m for better accuracy

            # Set date range for satellite imagery (6 months)
            if date:
                end_date = datetime.strptime(date, "%Y-%m-%d")
            else:
                end_date = datetime.now()

            start_date = _subtract_months(end_date, 6)

            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            # ---- SLOPE ANALYSIS ----
            # Get SRTM DEM
            dem = ee.Image("USGS/SRTMGL1_003")

            # Calculate slope in degrees
            slope = ee.Terrain.slope(dem)

            # Get slope statistics
            slope_stats = slope.reduceRegion(
                reducer=ee.Reducer.mean()
                .combine(ee.Reducer.minMax(), sharedInputs=True)
                .combine(ee.Reducer.stdDev(), sharedInputs=True),
                geometry=polygon,
                scale=30,  # SRTM resolution
                maxPixels=1e9,
            ).getInfo()

            # Get elevation statistics
            elevation_stats = dem.reduceRegion(
                reducer=ee.Reducer.mean().combine(
                    ee.Reducer.minMax(), sharedInputs=True
                ),
                geometry=polygon,
                scale=30,
                maxPixels=1e9,
            ).getInfo()

            # ---- JRC GLOBAL SURFACE WATER (Primary water detection) ----
            # This dataset provides historical water occurrence data
            jrc_water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")

            # Get water occurrence (0-100 = percentage of time water present)
            water_occurrence = jrc_water.select("occurrence")

            # Get seasonal water
            seasonality = jrc_water.select("seasonality")

            # Calculate water statistics from JRC
            jrc_stats = water_occurrence.reduceRegion(
                reducer=ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True),
                geometry=polygon,
                scale=30,
                maxPixels=1e9,
            ).getInfo()

            jrc_occurrence_mean = jrc_stats.get("occurrence_mean", 0) or 0
            jrc_occurrence_max = jrc_stats.get("occurrence_max", 0) or 0

            # Count pixels with significant water presence (>10% of time)
            water_present_mask = water_occurrence.gt(10)
            jrc_water_pixels = (
                water_present_mask.reduceRegion(
                    reducer=ee.Reducer.sum(), geometry=polygon, scale=30, maxPixels=1e9
                )
                .get("occurrence")
                .getInfo()
                or 0
            )

            jrc_total_pixels = (
                water_present_mask.unmask(0)
                .reduceRegion(
                    reducer=ee.Reducer.count(),
                    geometry=polygon,
                    scale=30,
                    maxPixels=1e9,
                )
                .get("occurrence")
                .getInfo()
                or 1
            )

            jrc_water_percentage = (jrc_water_pixels / jrc_total_pixels) * 100

            # ---- LAND COVER ANALYSIS ----
            # Get Sentinel-2 imagery
            collection = (
                ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                .filterBounds(polygon)
                .filterDate(start_str, end_str)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            )

            image_count = collection.size().getInfo()

            if image_count == 0:
                # Try with relaxed cloud filter
                collection = (
                    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                    .filterBounds(polygon)
                    .filterDate(start_str, end_str)
                    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))
                )
                image_count = collection.size().getInfo()

            if image_count > 0:
                image = collection.median()

                # Calculate indices
                ndvi = image.normalizedDifference(["B8", "B4"])
                ndbi = image.normalizedDifference(["B11", "B8"])

                # IMPROVED water detection using multiple indices
                # NDWI (McFeeters) - uses Green and NIR
                ndwi = image.normalizedDifference(["B3", "B8"])
                # MNDWI (Modified NDWI) - uses Green and SWIR, better for urban areas
                mndwi = image.normalizedDifference(["B3", "B11"])

                # AWEInsh (Automated Water Extraction Index - no shadow)
                # AWEInsh = 4 * (Green - SWIR1) - (0.25 * NIR + 2.75 * SWIR2)
                # For Sentinel-2: Green=B3, NIR=B8, SWIR1=B11, SWIR2=B12
                green = image.select("B3").divide(10000)  # Scale to reflectance
                nir = image.select("B8").divide(10000)
                swir1 = image.select("B11").divide(10000)
                swir2 = image.select("B12").divide(10000)
                awei = (
                    green.subtract(swir1)
                    .multiply(4)
                    .subtract(nir.multiply(0.25).add(swir2.multiply(2.75)))
                    .rename("AWEI")
                )

                # Get index statistics
                ndvi_mean = (
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

                ndbi_mean = (
                    ndbi.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=polygon,
                        scale=scale,
                        maxPixels=1e9,
                    )
                    .get("nd")
                    .getInfo()
                    or 0
                )

                ndwi_mean = (
                    ndwi.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=polygon,
                        scale=scale,
                        maxPixels=1e9,
                    )
                    .get("nd")
                    .getInfo()
                    or 0
                )

                mndwi_mean = (
                    mndwi.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=polygon,
                        scale=scale,
                        maxPixels=1e9,
                    )
                    .get("nd")
                    .getInfo()
                    or 0
                )

                awei_mean = (
                    awei.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=polygon,
                        scale=scale,
                        maxPixels=1e9,
                    )
                    .get("AWEI")
                    .getInfo()
                    or 0
                )

                # IMPROVED: Calculate water body percentage using multiple methods
                # More sensitive thresholds: NDWI > -0.1 OR MNDWI > -0.1 OR AWEI > 0
                water_mask_s2 = ndwi.gt(-0.1).Or(mndwi.gt(-0.1)).Or(awei.gt(0))
                water_pixels_s2 = (
                    water_mask_s2.reduceRegion(
                        reducer=ee.Reducer.sum(),
                        geometry=polygon,
                        scale=scale,
                        maxPixels=1e9,
                    )
                    .get("nd")
                    .getInfo()
                    or 0
                )

                total_pixels_s2 = (
                    water_mask_s2.unmask(0)
                    .reduceRegion(
                        reducer=ee.Reducer.count(),
                        geometry=polygon,
                        scale=scale,
                        maxPixels=1e9,
                    )
                    .get("nd")
                    .getInfo()
                    or 1
                )

                water_percentage_s2 = (water_pixels_s2 / total_pixels_s2) * 100

                has_land_cover_data = True
            else:
                ndvi_mean = 0
                ndbi_mean = 0
                ndwi_mean = 0
                mndwi_mean = 0
                awei_mean = 0
                water_percentage_s2 = 0
                has_land_cover_data = False

            # ---- COMBINED WATER ASSESSMENT ----
            # Use the maximum of JRC and Sentinel-2 water detection
            # This ensures we don't miss water bodies
            water_percentage = max(
                jrc_water_percentage, water_percentage_s2 if has_land_cover_data else 0
            )

            # Also check max water indices for current water presence
            max_water_index = max(ndwi_mean, mndwi_mean) if has_land_cover_data else 0

            # ---- SUITABILITY SCORING ----
            # Weights for different factors
            slope_weight = 0.35
            vegetation_weight = 0.20
            builtup_weight = 0.15
            water_weight = 0.30  # Increased weight for water/flood risk

            # Calculate component scores (0-100 scale)
            mean_slope = slope_stats.get("slope_mean", 0) or 0

            # Slope: Lower is better for building
            # Display value = slope level (0-100 where 100 = very steep)
            # Score = suitability (100 = flat/good, 10 = steep/bad)
            if mean_slope <= 5:
                slope_score = 100
                slope_display = 20  # Low slope shown as small bar
                slope_category = "Flat terrain"
            elif mean_slope <= 15:
                slope_score = 70
                slope_display = 40
                slope_category = "Gentle slope"
            elif mean_slope <= 30:
                slope_score = 40
                slope_display = 70
                slope_category = "Steep slope"
            else:
                slope_score = 10
                slope_display = 100  # Very steep shown as full bar
                slope_category = "Very steep"

            # Vegetation: Show actual vegetation level
            # High bar = high vegetation = needs clearing
            if ndvi_mean < 0.2:
                vegetation_score = 100  # Good for building
                vegetation_display = 20  # Low vegetation
                vegetation_category = "Minimal vegetation"
            elif ndvi_mean < 0.4:
                vegetation_score = 70
                vegetation_display = 45
                vegetation_category = "Low vegetation"
            elif ndvi_mean < 0.6:
                vegetation_score = 40
                vegetation_display = 70
                vegetation_category = "Moderate vegetation"
            else:
                vegetation_score = 20  # Bad for building
                vegetation_display = 95  # High vegetation shown as full bar
                vegetation_category = "Dense vegetation (clearing needed)"

            # Infrastructure: Show actual development level
            # High bar = high infrastructure = good
            if ndbi_mean > 0.1:
                builtup_score = 100
                builtup_display = 90  # Well developed
                builtup_category = "Well developed"
            elif ndbi_mean > -0.1:
                builtup_score = 70
                builtup_display = 60
                builtup_category = "Partially developed"
            elif ndbi_mean > -0.2:
                builtup_score = 50
                builtup_display = 35
                builtup_category = "Sparse development"
            else:
                builtup_score = 30
                builtup_display = 15  # Undeveloped shown as small bar
                builtup_category = "Undeveloped area"

            # Flood Risk: Show actual risk level
            # High bar = high risk = bad
            if (
                water_percentage > 20
                or jrc_occurrence_max > 80
                or max_water_index > 0.3
            ):
                water_score = 0
                water_display = 100  # Very high risk
                water_category = "Very high risk (water body)"
            elif (
                water_percentage > 10
                or jrc_occurrence_max > 50
                or max_water_index > 0.2
            ):
                water_score = 10
                water_display = 85
                water_category = "High flood risk"
            elif (
                water_percentage > 5 or jrc_occurrence_max > 30 or max_water_index > 0.1
            ):
                water_score = 30
                water_display = 60
                water_category = "Moderate flood risk"
            elif (
                water_percentage > 2 or jrc_occurrence_mean > 10 or max_water_index > 0
            ):
                water_score = 50
                water_display = 40
                water_category = "Some flood risk"
            elif (
                water_percentage > 0.5
                or jrc_occurrence_mean > 5
                or max_water_index > -0.1
            ):
                water_score = 70
                water_display = 20
                water_category = "Low flood risk"
            else:
                water_score = 100
                water_display = 5  # No risk shown as tiny bar
                water_category = "Minimal flood risk"

            # Calculate weighted total score
            total_score = (
                slope_score * slope_weight
                + vegetation_score * vegetation_weight
                + builtup_score * builtup_weight
                + water_score * water_weight
            )

            # No heatmap generation - removed for simplicity

            # Determine overall suitability
            if water_score == 0:
                overall_suitability = "Not Suitable"
                recommendation = "High flood risk area. This location is not suitable for construction."
            elif total_score >= 80:
                overall_suitability = "Highly Suitable"
                recommendation = (
                    "This area is excellent for construction with minimal constraints."
                )
            elif total_score >= 60:
                overall_suitability = "Suitable"
                recommendation = (
                    "This area is suitable for construction with some considerations."
                )
            elif total_score >= 40:
                overall_suitability = "Moderately Suitable"
                recommendation = (
                    "This area may require additional site preparation or engineering."
                )
            else:
                overall_suitability = "Not Recommended"
                recommendation = (
                    "This area has significant constraints for construction."
                )

            return {
                "status": "success",
                "total_area_m2": round(total_area, 2),
                "total_area_hectares": round(total_area / 10000, 4),
                "suitability": {
                    "overall_score": round(total_score, 1),
                    "category": overall_suitability,
                    "recommendation": recommendation,
                },
                "factors": {
                    "slope": {
                        "score": slope_score,
                        "display": slope_display,
                        "weight": slope_weight,
                        "category": slope_category,
                        "mean_degrees": round(mean_slope, 2),
                        "min_degrees": round(slope_stats.get("slope_min", 0) or 0, 2),
                        "max_degrees": round(slope_stats.get("slope_max", 0) or 0, 2),
                    },
                    "vegetation": {
                        "score": vegetation_score,
                        "display": vegetation_display,
                        "weight": vegetation_weight,
                        "category": vegetation_category,
                        "ndvi_mean": round(ndvi_mean, 4),
                    },
                    "infrastructure": {
                        "score": builtup_score,
                        "display": builtup_display,
                        "weight": builtup_weight,
                        "category": builtup_category,
                        "ndbi_mean": round(ndbi_mean, 4),
                    },
                    "flood_risk": {
                        "score": water_score,
                        "display": water_display,
                        "weight": water_weight,
                        "category": water_category,
                        "ndwi_mean": round(ndwi_mean, 4)
                        if has_land_cover_data
                        else None,
                        "mndwi_mean": round(mndwi_mean, 4)
                        if has_land_cover_data
                        else None,
                        "awei_mean": round(awei_mean, 4)
                        if has_land_cover_data
                        else None,
                        "water_percentage": round(water_percentage, 2),
                        "jrc_occurrence_mean": round(jrc_occurrence_mean, 2),
                        "jrc_occurrence_max": round(jrc_occurrence_max, 2),
                    },
                },
                "terrain": {
                    "mean_elevation_m": round(
                        elevation_stats.get("elevation_mean", 0) or 0, 2
                    ),
                    "min_elevation_m": round(
                        elevation_stats.get("elevation_min", 0) or 0, 2
                    ),
                    "max_elevation_m": round(
                        elevation_stats.get("elevation_max", 0) or 0, 2
                    ),
                    "elevation_range_m": round(
                        (elevation_stats.get("elevation_max", 0) or 0)
                        - (elevation_stats.get("elevation_min", 0) or 0),
                        2,
                    ),
                },
                "has_land_cover_data": has_land_cover_data,
                "date_range": {"start": start_str, "end": end_str},
                "images_used": image_count,
            }

        except Exception as e:
            self.logger.error(f"Error analyzing suitability: {str(e)}")
            return {"status": "error", "message": str(e)}
