"""
Backwards Compatibility Routes
Legacy API endpoints for frontend compatibility
Matches the original app_test.py behavior with direct GEE tile URL generation
"""
from flask import Blueprint, request, jsonify
from controllers.satellite_controller import SatelliteController
from datetime import datetime
import ee
import json
import logging

# Create blueprint for legacy routes
legacy_bp = Blueprint('legacy', __name__, url_prefix='/api')
logger = logging.getLogger(__name__)

# Controller will be injected via factory
satellite_controller: SatelliteController = None

# AI model for chat (will be initialized in init_legacy_routes)
_chat_model = None
_ai_available = False

# GEE Chat System Prompt (from original app_test.py)
GEE_CHAT_SYSTEM_PROMPT = """
You are a Google Earth Engine (GEE) expert and geospatial AI assistant for GEO VISION platform. Your task is to receive a user's natural language query and convert it into a valid JSON object for map visualization.

You MUST respond with ONLY a single, valid JSON object. Do not wrap it in markdown or any other text.

The JSON object MUST contain the following keys:
- 'gee_code': A string containing the Python code to generate the GEE Image object (e.g., "ee.Image(...)"). This is mandatory.
- 'vis_params': A JSON object of visualization parameters for the GEE image. This is mandatory.
- 'response_text': A user-friendly string explaining what you have done. This is mandatory.
- 'legend': An optional JSON object describing the legend for the map. It should have a 'title' (string) and 'items' (an array of objects, each with 'color' and 'label' keys).

**Primary Rules**:
1. The entire output MUST be a single, valid JSON object.
2. The `gee_code` string must be syntactically perfect Python code. Use Python's `None`, not `null`.
3. If the user does not specify a date range, you MUST assume they mean the last 30 days from today's date.
4. If the user's request is unclear or cannot be mapped to a GEE dataset, set 'gee_code' to null and explain why in the 'response_text'.

**Dataset-Specific Instructions**:
* **NDVI/Vegetation**: Use 'COPERNICUS/S2_SR_HARMONIZED' or 'MODIS/061/MOD13A1' dataset.
* **Country Filtering**: Use the 'FAO/GAUL/2015/level0' feature collection. For example: `.clip(ee.FeatureCollection('FAO/GAUL/2015/level0').filter(ee.Filter.eq('ADM0_NAME', 'India')))`.
* **Water/Waterbodies**: Use the 'JRC/GSW1_4/GlobalSurfaceWater' dataset and select the 'occurrence' band.
* **Fire/Wildfire**: Use the 'MODIS/061/MCD64A1' dataset and select the 'BurnDate' band.
* **Sentinel-1 Radar**: Use 'COPERNICUS/S1_GRD', filter by `instrumentMode` (usually 'IW').
* **Sentinel-2 Optical**: Use 'COPERNICUS/S2_SR_HARMONIZED'. For true-color, select bands 'B4', 'B3', 'B2'.
* **Temperature**: Use 'MODIS/061/MOD11A1' and select 'LST_Day_1km' band.
* **Precipitation**: Use 'UCSB-CHG/CHIRPS/DAILY' dataset.
* **Elevation**: Use 'MERIT/DEM/v1_0_3' dataset.
* **Nighttime Lights**: Use 'NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG' dataset.
* **Land Cover**: Use 'ESA/WorldCover/v200/2021' dataset.

Example response:
{
  "gee_code": "ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterDate('2025-07-30', '2025-08-30').filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',20)).median().clip(ee.FeatureCollection('FAO/GAUL/2015/level0').filter(ee.Filter.eq('ADM0_NAME', 'India')))",
  "vis_params": {
    "min": 0.0,
    "max": 3000,
    "bands": ["B4", "B3", "B2"]
  },
  "response_text": "Here is a Sentinel-2 true-color image for India from the last 30 days.",
  "legend": null
}
"""

def init_legacy_routes(controller: SatelliteController):
    """Initialize legacy routes with controller and AI chat model"""
    global satellite_controller, _chat_model, _ai_available
    satellite_controller = controller
    
    # Initialize the Gemini AI for chat (matching original app_test.py)
    try:
        import google.generativeai as genai
        from config import get_config
        config = get_config()
        api_key = config.GEMINI_API_KEY
        
        if api_key:
            genai.configure(api_key=api_key)
            
            model_names = [
                'models/gemini-2.5-flash',
                'models/gemini-2.0-flash',
                'models/gemini-flash-latest',
                'models/gemini-pro-latest',
                'models/gemini-2.5-pro',
                'models/gemini-2.0-pro-exp'
            ]
            
            for model_name in model_names:
                try:
                    logger.info(f"Trying chat model: {model_name}")
                    _chat_model = genai.GenerativeModel(model_name, system_instruction=GEE_CHAT_SYSTEM_PROMPT)
                    test_response = _chat_model.generate_content("Hello, respond with just 'OK'")
                    logger.info(f"Chat model configured: {model_name} - {test_response.text}")
                    _ai_available = True
                    break
                except Exception as model_error:
                    logger.warning(f"Model {model_name} failed: {model_error}")
                    continue
            
            if not _ai_available:
                logger.error("All chat models failed. AI chat will be disabled.")
        else:
            logger.error("No GEMINI_API_KEY configured. AI chat will be disabled.")
    except ImportError:
        logger.error("google-generativeai not installed. AI chat will be disabled.")
    except Exception as e:
        logger.error(f"Error configuring chat AI: {e}")


# =============================================================================
# Health check route (legacy /api/health) - direct JSON response
# =============================================================================
@legacy_bp.route('/health', methods=['GET'])
def legacy_health():
    """Legacy health check - returns JSON directly (matching old app_test.py)"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gee_status": "initialized" if (satellite_controller and satellite_controller.gee_service.initialized) else "not_initialized",
        "ai_status": "available" if _ai_available and _chat_model else "not_available",
        "user": "ShrishtiAI"
    })


# =============================================================================
# AI CHATBOT ENDPOINT (matching original app_test.py exactly)
# =============================================================================
@legacy_bp.route('/chat', methods=['POST'])
def chat():
    """AI Chat endpoint - generates GEE visualizations from natural language"""
    global _chat_model, _ai_available
    
    if not _ai_available or not _chat_model:
        return jsonify({
            'error': 'AI chatbot is not available. Please install required dependencies: pip install google-generativeai',
            'success': False
        }), 503

    try:
        data = request.get_json()
        user_message = data.get('message')
        history = data.get('history', [])

        if not user_message:
            return jsonify({'error': 'No message provided.', 'success': False}), 400

        # Add today's date context
        today = datetime.utcnow().strftime('%Y-%m-%d')
        contextual_message = f"Today's date is {today}. User (ShrishtiAI) query: {user_message}"

        conversation = history + [{'role': 'user', 'parts': [{'text': contextual_message}]}]

        # Retry logic with maximum 3 attempts
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"AI Attempt {attempt + 1} for user query: {user_message}")
                
                # Generate content with Gemini
                response = _chat_model.generate_content(conversation)
                raw_response_text = response.text.strip()
                
                # Clean potential markdown wrappers
                if raw_response_text.startswith("```json"):
                    raw_response_text = raw_response_text[7:-3].strip()
                elif raw_response_text.startswith("```"):
                    raw_response_text = raw_response_text[3:-3].strip()
                
                # Parse JSON response
                result_dict = json.loads(raw_response_text)
                
                gee_code_str = result_dict.get('gee_code')
                vis_params = result_dict.get('vis_params')
                response_text = result_dict.get('response_text', "Here is the map you requested.")
                legend = result_dict.get('legend')

                # If no GEE code, just return the text response
                if not gee_code_str or str(gee_code_str).lower() == 'null' or gee_code_str is None:
                    return jsonify({
                        'response_text': response_text,
                        'legend': legend,
                        'tile_url': None,
                        'success': True,
                        'gee_code': None,
                        'metadata': {
                            'title': 'AI Response',
                            'description': response_text,
                            'source': 'Geo Vision AI Assistant',
                            'timestamp': datetime.now().isoformat()
                        }
                    })

                # Try to execute the GEE code
                try:
                    # Replace 'null' with 'None' for Python compatibility
                    safe_gee_code_str = str(gee_code_str).replace('null', 'None')
                    logger.info(f"Executing GEE code: {safe_gee_code_str}")
                    
                    # Execute the code to get the GEE image
                    image = eval(safe_gee_code_str, {'ee': ee})
                    
                    # Generate map tiles
                    if image and vis_params:
                        map_id = image.getMapId(vis_params)
                        tile_url = map_id['tile_fetcher'].url_format
                        
                        # Create metadata
                        metadata = {
                            'title': f'AI Generated: {response_text[:50]}...' if len(response_text) > 50 else f'AI Generated: {response_text}',
                            'description': response_text,
                            'source': 'Geo Vision AI Assistant + Google Earth Engine',
                            'timestamp': datetime.now().isoformat(),
                            'gee_code': safe_gee_code_str,
                            'legend': legend.get('items', []) if legend else []
                        }
                        
                        return jsonify({
                            'tile_url': tile_url,
                            'response_text': response_text,
                            'legend': legend,
                            'success': True,
                            'gee_code': safe_gee_code_str,
                            'metadata': metadata
                        })
                    else:
                        raise Exception("Failed to generate map tiles - image or vis_params invalid")
                        
                except Exception as gee_error:
                    logger.error(f"GEE execution error on attempt {attempt + 1}: {gee_error}")
                    
                    if attempt < max_retries - 1:
                        error_message = f"The previous GEE code failed with error: {str(gee_error)}. Please provide corrected code that will work. Generate only valid JSON with working GEE code. Fix the syntax and dataset issues."
                        conversation.append({'role': 'user', 'parts': [{'text': error_message}]})
                        continue
                    else:
                        return jsonify({
                            'error': f'Failed to execute GEE code after {max_retries} attempts. Last error: {str(gee_error)}',
                            'response_text': response_text,
                            'success': False
                        }), 500

            except json.JSONDecodeError as json_error:
                logger.error(f"JSON parsing error on attempt {attempt + 1}: {json_error}")
                if attempt < max_retries - 1:
                    error_message = "Your response was not valid JSON. Please respond with ONLY a valid JSON object containing gee_code, vis_params, and response_text. Do not include any markdown formatting."
                    conversation.append({'role': 'user', 'parts': [{'text': error_message}]})
                    continue
                else:
                    return jsonify({
                        'error': f'Failed to parse AI response after {max_retries} attempts.',
                        'success': False
                    }), 500
                    
            except Exception as general_error:
                logger.error(f"General error on attempt {attempt + 1}: {general_error}")
                if attempt < max_retries - 1:
                    continue
                else:
                    return jsonify({
                        'error': f'An internal error occurred after {max_retries} attempts: {str(general_error)}',
                        'success': False
                    }), 500

        return jsonify({
            'error': 'Maximum retry attempts exceeded',
            'success': False
        }), 500

    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        return jsonify({
            'error': f'Chat service error: {str(e)}',
            'success': False
        }), 500


# =============================================================================
# GEE DATA ENDPOINTS - Direct tile URL generation (matching original app_test.py)
# =============================================================================

@legacy_bp.route('/gee/ndvi', methods=['GET'])
def get_ndvi_tiles():
    """Get NDVI tiles with direct GEE tile URL generation"""
    try:
        collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterDate("2024-06-01", "2024-06-10") \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
            .select(['B8', 'B4'])

        image = collection.median()
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')

        vis_params = {
            'min': 0.0,
            'max': 1.0,
            'palette': ['brown', 'yellow', 'green']
        }

        map_id = ndvi.getMapId(vis_params)
        
        return jsonify({
            'success': True,
            'tile_url': map_id['tile_fetcher'].url_format,
            'metadata': {
                'title': 'NDVI (Vegetation Health)',
                'description': 'Normalized Difference Vegetation Index showing plant health',
                'source': 'Copernicus Sentinel-2 SR Harmonized',
                'date_range': '2024-05-01 to 2024-06-10',
                'resolution': '10m',
                'legend': [
                    {'color': 'brown', 'label': 'No vegetation', 'value': '0.0 - 0.3'},
                    {'color': 'yellow', 'label': 'Sparse vegetation', 'value': '0.3 - 0.6'},
                    {'color': 'green', 'label': 'Healthy vegetation', 'value': '0.6 - 1.0'}
                ]
            }
        })
        
    except Exception as e:
        logger.error(f"Error in NDVI endpoint: {str(e)}")
        return jsonify({
            'success': False, 
            'error': f'Failed to process NDVI data: {str(e)}'
        }), 500


@legacy_bp.route('/gee/elevation', methods=['GET'])
def get_elevation_tiles():
    """Get elevation tiles with direct GEE tile URL generation"""
    try:
        import traceback as _tb
        
        # Debug: check ee session state
        logger.info("[ELEVATION_DEBUG] Starting elevation tile request...")
        
        try:
            _proj = ee.data._cloud_api_user_project
            logger.info(f"[ELEVATION_DEBUG] ee cloud_api_user_project = {_proj}")
        except Exception:
            logger.info("[ELEVATION_DEBUG] Could not read cloud_api_user_project")
        
        try:
            _creds = ee.data.getAsset('MERIT/DEM/v1_0_3')
            logger.info(f"[ELEVATION_DEBUG] getAsset succeeded — credentials work for reads")
        except Exception as asset_err:
            logger.error(f"[ELEVATION_DEBUG] getAsset FAILED: {type(asset_err).__name__}: {asset_err}")
        
        # Try a minimal getInfo first (cheaper than getMapId)
        try:
            _val = ee.Number(1).getInfo()
            logger.info(f"[ELEVATION_DEBUG] ee.Number(1).getInfo() = {_val}  (basic auth OK)")
        except Exception as info_err:
            logger.error(f"[ELEVATION_DEBUG] ee.Number(1).getInfo() FAILED: {type(info_err).__name__}: {info_err}")
        
        elevation = ee.Image("MERIT/DEM/v1_0_3").select('dem')

        vis_params = {
            'min': 0,
            'max': 6000,
            'palette': ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF8000', '#FF0000', '#800080']
        }

        logger.info("[ELEVATION_DEBUG] Calling getMapId...")
        try:
            map_id = elevation.getMapId(vis_params)
            logger.info(f"[ELEVATION_DEBUG] getMapId succeeded, tile_url prefix: {map_id['tile_fetcher'].url_format[:80]}...")
        except Exception as map_err:
            logger.error(f"[ELEVATION_DEBUG] getMapId FAILED: {type(map_err).__name__}: {map_err}")
            logger.error(f"[ELEVATION_DEBUG] Full traceback:\n{_tb.format_exc()}")
            raise
        
        return jsonify({
            'success': True,
            'tile_url': map_id['tile_fetcher'].url_format,
            'metadata': {
                'title': 'Digital Elevation Model',
                'description': 'Terrain elevation above sea level',
                'source': 'MERIT DEM',
                'resolution': '90m',
                'legend': [
                    {'color': '#000080', 'label': 'Sea level', 'value': '0m'},
                    {'color': '#0000FF', 'label': 'Low elevation', 'value': '0-500m'},
                    {'color': '#00FFFF', 'label': 'Hills', 'value': '500-1000m'},
                    {'color': '#FFFF00', 'label': 'Mountains', 'value': '1000-2000m'},
                    {'color': '#FF8000', 'label': 'High mountains', 'value': '2000-4000m'},
                    {'color': '#FF0000', 'label': 'Very high peaks', 'value': '4000-6000m'},
                    {'color': '#800080', 'label': 'Extreme peaks', 'value': '>6000m'}
                ]
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@legacy_bp.route('/gee/lights', methods=['GET'])
def get_nighttime_lights_tiles():
    """Get nighttime lights tiles with stunning visualization"""
    try:
        image = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG') \
            .filterDate('2023-01-01', '2023-12-31') \
            .select('avg_rad') \
            .median()
        
        # Apply gamma correction for better visual appeal
        gamma_corrected = image.pow(0.8).multiply(5)
        
        # Ultra-dramatic visualization
        vis_params = {
            'min': 0,
            'max': 15,
            'palette': [
                '#000000',
                '#1a0033',
                '#330066',
                '#4d0099',
                '#6600cc',
                '#7f00ff',
                '#9933ff',
                '#cc66ff',
                '#ff99ff',
                '#ffccff',
                '#ffffff'
            ]
        }

        map_id = gamma_corrected.getMapId(vis_params)
        
        return jsonify({
            'success': True,
            'tile_url': map_id['tile_fetcher'].url_format,
            'metadata': {
                'title': 'Nighttime Lights (Dramatic)',
                'description': 'Dramatic visualization of global human activity at night',
                'source': 'NOAA VIIRS DNB (Gamma Enhanced)',
                'date_range': '2023 Annual Composite',
                'processing': 'Gamma correction and contrast enhancement',
                'legend': [
                    {'color': '#000000', 'label': 'No activity', 'value': 'Uninhabited'},
                    {'color': '#1a0033', 'label': 'Minimal', 'value': 'Rural'},
                    {'color': '#330066', 'label': 'Low', 'value': 'Small towns'},
                    {'color': '#6600cc', 'label': 'Medium', 'value': 'Cities'},
                    {'color': '#9933ff', 'label': 'High', 'value': 'Major cities'},
                    {'color': '#cc66ff', 'label': 'Very high', 'value': 'Mega cities'},
                    {'color': '#ffccff', 'label': 'Extreme', 'value': 'Urban centers'},
                    {'color': '#ffffff', 'label': 'Maximum', 'value': 'City cores'}
                ]
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@legacy_bp.route('/gee/landcover', methods=['GET'])
def get_landcover_tiles():
    """Get land cover tiles with proper ESA WorldCover visualization"""
    try:
        image = ee.Image("ESA/WorldCover/v200/2021")

        vis_params = {
            'min': 10,
            'max': 100,
            'palette': [
                '#006400', '#ffbb22', '#ffff4c', '#f096ff', '#fa0000',
                '#b4b4b4', '#f0f0f0', '#0064c8', '#0096a0', '#00cf75'
            ]
        }

        map_id = image.getMapId(vis_params)
        
        return jsonify({
            'success': True,
            'tile_url': map_id['tile_fetcher'].url_format,
            'metadata': {
                'title': 'Land Cover Classification',
                'description': 'Global land cover types at 10m resolution',
                'source': 'ESA WorldCover 2021',
                'resolution': '10m',
                'legend': [
                    {'color': '#006400', 'label': 'Tree cover', 'value': '10'},
                    {'color': '#ffbb22', 'label': 'Shrubland', 'value': '20'},
                    {'color': '#ffff4c', 'label': 'Grassland', 'value': '30'},
                    {'color': '#f096ff', 'label': 'Cropland', 'value': '40'},
                    {'color': '#fa0000', 'label': 'Built-up', 'value': '50'},
                    {'color': '#b4b4b4', 'label': 'Bare/sparse vegetation', 'value': '60'},
                    {'color': '#f0f0f0', 'label': 'Snow and ice', 'value': '70'},
                    {'color': '#0064c8', 'label': 'Permanent water bodies', 'value': '80'},
                    {'color': '#0096a0', 'label': 'Herbaceous wetland', 'value': '90'},
                    {'color': '#00cf75', 'label': 'Mangroves', 'value': '95'}
                ]
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@legacy_bp.route('/gee/temperature', methods=['GET'])
def get_temperature_tiles():
    """Get temperature tiles with proper MODIS visualization"""
    try:
        collection = ee.ImageCollection('MODIS/061/MOD11A1') \
            .filterDate('2023-06-01', '2023-08-31') \
            .select('LST_Day_1km')
        
        # Convert from Kelvin to Celsius
        temp_celsius = collection.median().multiply(0.02).subtract(273.15)

        vis_params = {
            'min': 15,
            'max': 45,
            'palette': ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000']
        }

        map_id = temp_celsius.getMapId(vis_params)
        
        return jsonify({
            'success': True,
            'tile_url': map_id['tile_fetcher'].url_format,
            'metadata': {
                'title': 'Land Surface Temperature',
                'description': 'Daytime land surface temperature',
                'source': 'MODIS Terra',
                'date_range': 'Summer 2023',
                'legend': [
                    {'color': '#000080', 'label': 'Very cold', 'value': '< 20\u00b0C'},
                    {'color': '#0000FF', 'label': 'Cold', 'value': '20-25\u00b0C'},
                    {'color': '#00FFFF', 'label': 'Cool', 'value': '25-30\u00b0C'},
                    {'color': '#00FF00', 'label': 'Moderate', 'value': '30-35\u00b0C'},
                    {'color': '#FFFF00', 'label': 'Warm', 'value': '35-40\u00b0C'},
                    {'color': '#FF8000', 'label': 'Hot', 'value': '40-45\u00b0C'},
                    {'color': '#FF0000', 'label': 'Very hot', 'value': '> 45\u00b0C'}
                ]
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================================================================
# SATELLITE TIMELAPSE ENDPOINT - generates tile URLs for multiple time periods
# =============================================================================

@legacy_bp.route('/gee/timelapse', methods=['GET', 'POST'])
def get_timelapse_tiles():
    """
    Generate GEE tile URLs for multiple time periods to enable map animation.
    Supports Sentinel-2 true-color, NDVI, and MODIS temperature.
    Query params / JSON body:
      dataset  – sentinel2 | ndvi | temperature  (default sentinel2)
      frames   – number of time frames 2-8  (default 4)
    Returns an array of {date_label, tile_url, start_date, end_date} per frame.
    """
    from datetime import timedelta as _td

    try:
        if request.method == 'POST':
            body = request.get_json(silent=True) or {}
        else:
            body = {}
        dataset = body.get('dataset') or request.args.get('dataset', 'sentinel2')
        frames  = int(body.get('frames') or request.args.get('frames', 4))
        frames  = max(2, min(frames, 24))

        # Optional custom date range  (YYYY-MM format)
        start_ym = body.get('start_year_month') or request.args.get('start_year_month')
        end_ym   = body.get('end_year_month')   or request.args.get('end_year_month')

        logger.info(f"[TIMELAPSE] dataset={dataset}, frames={frames}, range={start_ym}→{end_ym}")

        windows = []
        if start_ym and end_ym:
            # Build one window per calendar month in the requested range
            from dateutil.relativedelta import relativedelta
            cur = datetime.strptime(start_ym, '%Y-%m')
            end_dt = datetime.strptime(end_ym,   '%Y-%m')
            while cur <= end_dt:
                win_start = cur
                win_end   = cur + relativedelta(months=1)
                windows.append((win_start.strftime('%Y-%m-%d'), win_end.strftime('%Y-%m-%d')))
                cur += relativedelta(months=1)
            # cap at 24 frames
            windows = windows[:24]
        else:
            # Default: most-recent N months, going backwards from today
            # Nightlights (VIIRS monthly) lags ~3-4 months; use bigger offset
            if dataset == 'nightlights':
                lag_days = 120  # ~4 months back from today
            else:
                lag_days = 14
            end_anchor = datetime.utcnow() - _td(days=lag_days)
            for i in range(frames):
                win_end   = end_anchor - _td(days=30 * i)
                win_start = win_end   - _td(days=30)
                windows.append((win_start.strftime('%Y-%m-%d'), win_end.strftime('%Y-%m-%d')))
            windows.reverse()   # oldest → newest

        results = []
        for start_date, end_date in windows:
            tile_url = _build_timelapse_tile(dataset, start_date, end_date)
            # Friendly label: "Jan 2026"
            mid = datetime.strptime(start_date, '%Y-%m-%d') + _td(days=15)
            label = mid.strftime('%b %Y')
            results.append({
                'date_label': label,
                'start_date': start_date,
                'end_date':   end_date,
                'tile_url':   tile_url,
            })

        # windows are already oldest→newest (both branches handle ordering)

        return jsonify({
            'success': True,
            'dataset': dataset,
            'frames': results,
            'metadata': {
                'title': _timelapse_title(dataset),
                'description': f'{len(results)}-frame timelapse animation',
                'source': _timelapse_source(dataset),
            }
        })
    except Exception as e:
        logger.error(f"[TIMELAPSE] Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def _build_timelapse_tile(dataset: str, start: str, end: str) -> str:
    """Return a GEE XYZ tile URL for the given dataset and date window."""
    if dataset == 'sentinel2':
        col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterDate(start, end) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 25)) \
            .select(['B4', 'B3', 'B2'])
        image = col.median()
        vis = {'min': 0, 'max': 3000, 'bands': ['B4', 'B3', 'B2']}
    elif dataset == 'ndvi':
        col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterDate(start, end) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 25)) \
            .select(['B8', 'B4'])
        image = col.median().normalizedDifference(['B8', 'B4']).rename('NDVI')
        vis = {'min': 0.0, 'max': 1.0, 'palette': ['brown', 'yellow', 'green']}
    elif dataset == 'temperature':
        col = ee.ImageCollection('MODIS/061/MOD11A1') \
            .filterDate(start, end) \
            .select('LST_Day_1km')
        image = col.median().multiply(0.02).subtract(273.15)
        vis = {'min': -10, 'max': 45, 'palette': ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000']}
    elif dataset == 'nightlights':
        # VIIRS monthly — anchor already shifted 4 months back so data exists
        col = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG') \
            .filterDate(start, end) \
            .select('avg_rad')
        raw = col.median()
        # Clamp negatives then apply gamma
        clamped = raw.max(ee.Image.constant(0))
        image = clamped.pow(0.8).multiply(5)
        vis = {'min': 0, 'max': 15, 'palette': ['#000000', '#1a0033', '#330066', '#6600cc', '#9933ff', '#cc66ff', '#ffccff', '#ffffff']}
    elif dataset == 'precipitation':
        col = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
            .filterDate(start, end) \
            .select('precipitation')
        image = col.sum()  # total rainfall over the period
        vis = {'min': 0, 'max': 400, 'palette': ['#f7fbff', '#deebf7', '#9ecae1', '#4292c6', '#2171b5', '#08519c', '#08306b']}
    elif dataset == 'vegetation_modis':
        col = ee.ImageCollection('MODIS/061/MOD13A1') \
            .filterDate(start, end) \
            .select('NDVI')
        image = col.median().multiply(0.0001)
        vis = {'min': 0.0, 'max': 0.8, 'palette': ['#8B4513', '#D2B48C', '#FFFF00', '#90EE90', '#228B22', '#006400']}
    else:
        raise ValueError(f'Unknown timelapse dataset: {dataset}')

    map_id = image.getMapId(vis)
    return map_id['tile_fetcher'].url_format


_TIMELAPSE_META = {
    'sentinel2':      ('Sentinel-2 True Color',    'Copernicus Sentinel-2 SR'),
    'ndvi':           ('Vegetation Index (NDVI)',   'Copernicus Sentinel-2 SR'),
    'temperature':    ('Surface Temperature',       'MODIS Terra MOD11A1'),
    'nightlights':    ('Nighttime Lights',          'NOAA VIIRS DNB Monthly'),
    'precipitation':  ('Precipitation',             'CHIRPS Daily Rainfall'),
    'vegetation_modis': ('Vegetation (MODIS)',      'MODIS Terra MOD13A1'),
}


def _timelapse_title(ds: str) -> str:
    return _TIMELAPSE_META.get(ds, (ds, ''))[0]


def _timelapse_source(ds: str) -> str:
    return _TIMELAPSE_META.get(ds, ('', 'Google Earth Engine'))[1]


GEE_SNIPPET_CATALOG = [
    {
        'id': 'worldcereal_models_v100',
        'name': 'WorldCereal Models',
        'title': 'ESA WorldCereal Crop Classification',
        'description': 'WorldCereal model-based crop classification and confidence.',
        'category': 'Agriculture',
        'source': 'ESA/WorldCereal/2021/MODELS/v100',
    },
    {
        'id': 'worldcereal_markers_v100',
        'name': 'WorldCereal Markers',
        'title': 'Active Cropland Marker',
        'description': 'WorldCereal marker layer for active cropland mapping.',
        'category': 'Agriculture',
        'source': 'ESA/WorldCereal/2021/MARKERS/v100',
    },
    {
        'id': 'wapor_et_ratio',
        'name': 'WAPOR ET Ratio',
        'title': 'Evapotranspiration Ratio (ETa/ETo)',
        'description': 'FAO WAPOR derived evapotranspiration efficiency ratio.',
        'category': 'Water & Agriculture',
        'source': 'FAO/WAPOR/3/L1_AETI_D + FAO/WAPOR/3/L1_RET_D',
    },
    {
        'id': 'gfsad_cropland_extent',
        'name': 'GFSAD Cropland Extent',
        'title': 'Global Cropland Extent',
        'description': 'USGS global cropland extent / landcover map.',
        'category': 'Agriculture',
        'source': 'USGS/GFSAD1000_V1',
    },
    {
        'id': 'fao_drained_organic_soils',
        'name': 'FAO Drained Organic Soils',
        'title': 'Drained Organic Soil Area',
        'description': 'Latest FAO drained organic soils layer.',
        'category': 'Climate',
        'source': 'FAO/GHG/1/DROSA_A',
    },
    {
        'id': 'forest_loss_drivers_wri_gdm',
        'name': 'Forest Loss Drivers',
        'title': 'Dominant Driver of Forest Loss',
        'description': 'Driver class map for primary forest loss.',
        'category': 'Hazards',
        'source': 'projects/landandcarbon/assets/wri_gdm_drivers_forest_loss_1km/v1_2_2001_2024',
    },
    {
        'id': 'chirps_rainfall_anomaly',
        'name': 'CHIRPS Rainfall Anomaly',
        'title': 'Rainfall Anomaly vs Baseline',
        'description': 'Recent rainfall minus long-term baseline using CHIRPS.',
        'category': 'Climate',
        'source': 'UCSB-CHG/CHIRPS/DAILY',
    },
    {
        'id': 'gfs_forecast_panel',
        'name': 'GFS Forecast Panel',
        'title': '24h Forecast Temperature',
        'description': 'NOAA GFS near-term 2m temperature forecast.',
        'category': 'Weather',
        'source': 'NOAA/GFS0P25',
    },
    {
        'id': 'soil_moisture_smap',
        'name': 'SMAP Soil Moisture',
        'title': 'Surface Soil Moisture',
        'description': 'NASA SMAP surface soil moisture climatology.',
        'category': 'Soil & Water',
        'source': 'NASA/SMAP/SPL3SMP_E/006',
    },
    {
        'id': 'spei_drought_index',
        'name': 'SPEI Drought Index',
        'title': 'SPEI 12-Month Drought Index',
        'description': 'Standardized drought index from CSIC SPEI.',
        'category': 'Drought',
        'source': 'CSIC/SPEI/2_10',
    },
    {
        'id': 'kbdi_drought_index',
        'name': 'KBDI Drought Index',
        'title': 'Keetch-Byram Drought Index',
        'description': 'KBDI wildfire-related drought indicator.',
        'category': 'Drought',
        'source': 'UTOKYO/WTLAB/KBDI/v1',
    },
    {
        'id': 'dynamic_land_cover_change',
        'name': 'Dynamic Land Cover Change',
        'title': 'Dynamic World Crops Probability',
        'description': 'Dynamic World crop probability mean composite.',
        'category': 'Land Cover',
        'source': 'GOOGLE/DYNAMICWORLD/V1',
    },
    {
        'id': 'era5_land_heat_stress',
        'name': 'ERA5 Land Heat Stress',
        'title': 'ERA5-Land Mean 2m Temperature',
        'description': 'Mean near-surface temperature from ERA5-Land hourly data.',
        'category': 'Heat',
        'source': 'ECMWF/ERA5_LAND/HOURLY',
    },
    {
        'id': 'soilgrids_baseline',
        'name': 'SoilGrids Baseline',
        'title': 'Volumetric Soil Water Content Baseline',
        'description': 'Static SoilGrids volumetric water content layer.',
        'category': 'Soil & Water',
        'source': 'ISRIC/SoilGrids250m/v2_0/wv0010',
    },
    {
        'id': 'gpm_imerg_precip',
        'name': 'GPM IMERG Precipitation',
        'title': 'IMERG Mean Precipitation',
        'description': 'Mean precipitation from NASA GPM IMERG V07.',
        'category': 'Weather',
        'source': 'NASA/GPM_L3/IMERG_V07',
    },
]


@legacy_bp.route('/gee/catalog', methods=['GET'])
def get_gee_catalog():
    """Catalog endpoint for dataset discovery modal."""
    return jsonify({
        'success': True,
        'count': len(GEE_SNIPPET_CATALOG),
        'datasets': GEE_SNIPPET_CATALOG,
    })


def _build_dynamic_dataset(dataset_id: str):
    """Build image + visualization + metadata for dynamic /api/gee/<dataset_id>."""
    ds = (dataset_id or '').strip().lower()

    # Existing frontend aliases
    if ds in ('vegetation', 'ndvi'):
        collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterDate("2024-06-01", "2024-06-10") \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
            .select(['B8', 'B4'])
        image = collection.median().normalizedDifference(['B8', 'B4']).rename('NDVI')
        vis = {'min': 0.0, 'max': 1.0, 'palette': ['brown', 'yellow', 'green']}
        meta = {'title': 'NDVI (Vegetation Health)', 'description': 'Normalized Difference Vegetation Index.', 'source': 'COPERNICUS/S2_SR_HARMONIZED'}
        return image, vis, meta

    if ds in ('terrain', 'elevation'):
        image = ee.Image("MERIT/DEM/v1_0_3").select('dem')
        vis = {'min': 0, 'max': 6000, 'palette': ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF8000', '#FF0000', '#800080']}
        meta = {'title': 'Digital Elevation Model', 'description': 'Terrain elevation above sea level.', 'source': 'MERIT/DEM/v1_0_3'}
        return image, vis, meta

    if ds in ('nightlights', 'lights'):
        image = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG').filterDate('2023-01-01', '2023-12-31').select('avg_rad').median().pow(0.8).multiply(5)
        vis = {'min': 0, 'max': 15, 'palette': ['#000000', '#1a0033', '#330066', '#6600cc', '#9933ff', '#cc66ff', '#ffccff', '#ffffff']}
        meta = {'title': 'Nighttime Lights', 'description': 'Nighttime radiance composite.', 'source': 'NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG'}
        return image, vis, meta

    if ds == 'landcover':
        image = ee.Image("ESA/WorldCover/v200/2021")
        vis = {'min': 10, 'max': 100, 'palette': ['#006400', '#ffbb22', '#ffff4c', '#f096ff', '#fa0000', '#b4b4b4', '#f0f0f0', '#0064c8', '#0096a0', '#00cf75']}
        meta = {'title': 'Land Cover Classification', 'description': 'Global land cover types.', 'source': 'ESA/WorldCover/v200/2021'}
        return image, vis, meta

    if ds == 'temperature':
        image = ee.ImageCollection('MODIS/061/MOD11A1').filterDate('2023-06-01', '2023-08-31').select('LST_Day_1km').median().multiply(0.02).subtract(273.15)
        vis = {'min': 15, 'max': 45, 'palette': ['#000080', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000']}
        meta = {'title': 'Land Surface Temperature', 'description': 'Daytime land surface temperature.', 'source': 'MODIS/061/MOD11A1'}
        return image, vis, meta

    if ds in ('rainfall', 'gpm_imerg_precip'):
        image = ee.ImageCollection('NASA/GPM_L3/IMERG_V07').filterDate('2024-06-01', '2024-06-30').select('precipitation').mean()
        vis = {'min': 0, 'max': 10, 'palette': ['f7fbff', 'c6dbef', '6baed6', '2171b5', '08306b']}
        meta = {'title': 'IMERG Precipitation', 'description': 'Mean precipitation composite.', 'source': 'NASA/GPM_L3/IMERG_V07'}
        return image, vis, meta

    if ds == 'ocean_temp':
        image = ee.ImageCollection('NASA/OCEANDATA/MODIS-Aqua/L3SMI').filterDate('2023-01-01', '2023-12-31').select('sst').mean()
        vis = {'min': -2, 'max': 35, 'palette': ['#053061', '#2166ac', '#4393c3', '#d1e5f0', '#f4a582', '#b2182b']}
        meta = {'title': 'Ocean Surface Temperature', 'description': 'Sea-surface temperature mean composite.', 'source': 'NASA/OCEANDATA/MODIS-Aqua/L3SMI'}
        return image, vis, meta

    if ds == 'wildfire_risk':
        image = ee.ImageCollection('MODIS/061/MCD64A1').filterDate('2023-01-01', '2023-12-31').select('BurnDate').mean()
        vis = {'min': 0, 'max': 366, 'palette': ['#ffffb2', '#fecc5c', '#fd8d3c', '#f03b20', '#bd0026']}
        meta = {'title': 'Wildfire Risk Proxy', 'description': 'Mean burn-date intensity from MODIS burned area.', 'source': 'MODIS/061/MCD64A1'}
        return image, vis, meta

    if ds == 'population':
        image = ee.ImageCollection('CIESIN/GPWv411/GPW_Population_Density').filterDate('2020-01-01', '2020-12-31').first().select('population_density')
        vis = {'min': 0, 'max': 2000, 'palette': ['#fcfbfd', '#dadaeb', '#bcbddc', '#9e9ac8', '#756bb1', '#54278f']}
        meta = {'title': 'Population Density', 'description': 'Population density estimate.', 'source': 'CIESIN/GPWv411/GPW_Population_Density'}
        return image, vis, meta

    # Snippet ids (backend/gee_test_snippets)
    if ds == 'worldcereal_models_v100':
        first = ee.ImageCollection('ESA/WorldCereal/2021/MODELS/v100').first()
        image = ee.Image(first).select('classification')
        vis = {'min': 0, 'max': 100, 'palette': ['000000', '00ff00']}
        meta = {'title': 'WorldCereal Classification', 'description': 'Model-based crop classification.', 'source': 'ESA/WorldCereal/2021/MODELS/v100'}
        return image, vis, meta

    if ds == 'worldcereal_markers_v100':
        first = ee.ImageCollection('ESA/WorldCereal/2021/MARKERS/v100').first()
        image = ee.Image(first).select('classification')
        vis = {'min': 0, 'max': 100, 'palette': ['000000', '00bfff']}
        meta = {'title': 'WorldCereal Markers', 'description': 'Active cropland marker.', 'source': 'ESA/WorldCereal/2021/MARKERS/v100'}
        return image, vis, meta

    if ds == 'wapor_et_ratio':
        aeti = ee.ImageCollection('FAO/WAPOR/3/L1_AETI_D').filterDate('2024-01-01', '2024-12-31').mean()
        ret = ee.ImageCollection('FAO/WAPOR/3/L1_RET_D').filterDate('2024-01-01', '2024-12-31').mean()
        image = aeti.select('L1-AETI-D').divide(ret.select('L1-RET-D')).rename('ETa_ETo_ratio')
        vis = {'min': 0, 'max': 1.5, 'palette': ['8b0000', 'ffa500', 'ffff00', '00ff00']}
        meta = {'title': 'WAPOR ET Ratio', 'description': 'ETa/ETo water-use efficiency ratio.', 'source': 'FAO/WAPOR/3'}
        return image, vis, meta

    if ds == 'gfsad_cropland_extent':
        image = ee.Image('USGS/GFSAD1000_V1').select('landcover')
        vis = {'min': 0, 'max': 5, 'palette': ['000000', 'ff8c00', '8b4513', '00a650', '7fff00', 'ffff00']}
        meta = {'title': 'GFSAD Cropland Extent', 'description': 'Global cropland extent landcover.', 'source': 'USGS/GFSAD1000_V1'}
        return image, vis, meta

    if ds == 'fao_drained_organic_soils':
        latest = ee.ImageCollection('FAO/GHG/1/DROSA_A').sort('system:time_start', False).first()
        image = ee.Image(latest).select(0)
        vis = {'min': 0, 'max': 50, 'palette': ['f7fcf5', '74c476', '00441b']}
        meta = {'title': 'FAO Drained Organic Soils', 'description': 'Latest drained organic soil area layer.', 'source': 'FAO/GHG/1/DROSA_A'}
        return image, vis, meta

    if ds == 'forest_loss_drivers_wri_gdm':
        image = ee.Image('projects/landandcarbon/assets/wri_gdm_drivers_forest_loss_1km/v1_2_2001_2024').select(0)
        vis = {'min': 1, 'max': 7, 'palette': ['fdae61', 'd7191c', 'abdda4', '2b83ba', 'f46d43', '8073ac', '999999']}
        meta = {'title': 'Forest Loss Drivers', 'description': 'Dominant driver of forest loss.', 'source': 'WRI GDM forest loss drivers'}
        return image, vis, meta

    if ds == 'chirps_rainfall_anomaly':
        recent = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filterDate('2024-06-01', '2024-08-31').select('precipitation').sum()
        baseline = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filterDate('2014-06-01', '2023-08-31').select('precipitation').mean()
        image = recent.subtract(baseline).rename('rain_anomaly_mm')
        vis = {'min': -200, 'max': 200, 'palette': ['8b0000', 'fdd49e', 'f7f7f7', '9ecae1', '08519c']}
        meta = {'title': 'CHIRPS Rainfall Anomaly', 'description': 'Recent rainfall anomaly vs baseline.', 'source': 'UCSB-CHG/CHIRPS/DAILY'}
        return image, vis, meta

    if ds == 'gfs_forecast_panel':
        gfs = ee.ImageCollection('NOAA/GFS0P25').filter(ee.Filter.gte('forecast_hours', 0)).filter(ee.Filter.lte('forecast_hours', 24)).sort('system:time_start', False)
        image = ee.Image(gfs.first()).select('temperature_2m_above_ground').subtract(273.15)
        vis = {'min': 10, 'max': 45, 'palette': ['313695', '74add1', 'fee090', 'f46d43', 'a50026']}
        meta = {'title': 'GFS Forecast Temperature', 'description': 'Near-term 2m temperature forecast.', 'source': 'NOAA/GFS0P25'}
        return image, vis, meta

    if ds == 'soil_moisture_smap':
        image = ee.ImageCollection('NASA/SMAP/SPL3SMP_E/006').filterDate('2024-01-01', '2024-12-31').select('soil_moisture_am').mean()
        vis = {'min': 0, 'max': 0.6, 'palette': ['f46d43', 'fdae61', 'abd9e9', '2c7bb6']}
        meta = {'title': 'SMAP Soil Moisture', 'description': 'Surface soil moisture mean.', 'source': 'NASA/SMAP/SPL3SMP_E/006'}
        return image, vis, meta

    if ds == 'spei_drought_index':
        latest = ee.ImageCollection('CSIC/SPEI/2_10').filterDate('2022-01-01', '2023-01-01').sort('system:time_start', False).first()
        image = ee.Image(latest).select('SPEI_12_month')
        vis = {'min': -2.5, 'max': 2.5, 'palette': ['8b0000', 'f46d43', 'fee08b', 'd9ef8b', '66bd63', '1a9850']}
        meta = {'title': 'SPEI Drought Index', 'description': 'SPEI 12-month drought index.', 'source': 'CSIC/SPEI/2_10'}
        return image, vis, meta

    if ds == 'kbdi_drought_index':
        latest = ee.ImageCollection('UTOKYO/WTLAB/KBDI/v1').filterDate('2024-01-01', '2024-12-31').sort('system:time_start', False).first()
        image = ee.Image(latest).select('KBDI')
        vis = {'min': 0, 'max': 800, 'palette': ['313695', '74add1', 'fee090', 'f46d43', 'a50026']}
        meta = {'title': 'KBDI Drought Index', 'description': 'Keetch-Byram Drought Index.', 'source': 'UTOKYO/WTLAB/KBDI/v1'}
        return image, vis, meta

    if ds == 'dynamic_land_cover_change':
        image = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterDate('2023-01-01', '2023-12-31').select('crops').mean()
        vis = {'min': 0, 'max': 1, 'palette': ['ffffff', 'ffffb2', 'fecc5c', 'e31a1c']}
        meta = {'title': 'Dynamic World Crops Probability', 'description': 'Mean crops probability from Dynamic World.', 'source': 'GOOGLE/DYNAMICWORLD/V1'}
        return image, vis, meta

    if ds == 'era5_land_heat_stress':
        image = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').filterDate('2024-05-01', '2024-05-31').select('temperature_2m').mean().subtract(273.15)
        vis = {'min': 15, 'max': 45, 'palette': ['313695', '74add1', 'fee08b', 'f46d43', 'a50026']}
        meta = {'title': 'ERA5-Land Heat Stress', 'description': 'Mean 2m temperature heat-stress proxy.', 'source': 'ECMWF/ERA5_LAND/HOURLY'}
        return image, vis, meta

    if ds == 'soilgrids_baseline':
        image = ee.Image('ISRIC/SoilGrids250m/v2_0/wv0010').select('val_0_5cm_Q0_5')
        vis = {'min': 0.05, 'max': 0.6, 'palette': ['440154', '3b528b', '21918c', '5ec962', 'fde725']}
        meta = {'title': 'SoilGrids Baseline', 'description': 'Volumetric soil water content baseline.', 'source': 'ISRIC/SoilGrids250m/v2_0'}
        return image, vis, meta

    raise ValueError(f'Unsupported dataset id: {dataset_id}')


@legacy_bp.route('/gee/<dataset_id>', methods=['GET'])
def get_dynamic_dataset_tiles(dataset_id: str):
    """Dynamic dataset tiles endpoint used by dataset modal and cards."""
    try:
        image, vis_params, metadata = _build_dynamic_dataset(dataset_id)
        map_id = image.getMapId(vis_params)
        return jsonify({
            'success': True,
            'tile_url': map_id['tile_fetcher'].url_format,
            'metadata': {
                **metadata,
                'dataset_id': dataset_id,
                'timestamp': datetime.now().isoformat(),
            }
        })
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 404
    except Exception as e:
        logger.error(f"[GEE_DYNAMIC] Dataset {dataset_id} failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500