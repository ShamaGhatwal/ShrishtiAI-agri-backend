"""
GEO VISION Backend - Main Application
Professional Flask backend with MVC architecture
"""

# ============================================================================
# CRITICAL: PROJ/GDAL Setup MUST be done BEFORE any rasterio imports
# ============================================================================
import os
import sys
import time
from pathlib import Path

_startup_t0 = time.time()

def _elapsed():
    return f"{time.time() - _startup_t0:.1f}s"

# ── Local proj_data (copied from rasterio) lives right here in the backend dir ──
_BACKEND_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
proj_lib_path = _BACKEND_DIR / "proj_data"

# Fallback chain: backend/proj_data → rasterio package → pyproj package
# (cross-platform — works on Windows dev and Linux Render)
def _find_proj_data():
    """Find proj.db in known locations, cross-platform."""
    candidates = [
        _BACKEND_DIR / "proj_data",
    ]
    # Try rasterio's bundled proj_data
    try:
        import rasterio
        candidates.append(Path(rasterio.__file__).parent / "proj_data")
    except ImportError:
        pass
    # Try pyproj's bundled data
    try:
        import pyproj
        candidates.append(Path(pyproj.datadir.get_data_dir()))
    except (ImportError, AttributeError):
        pass
    # Try common system locations
    candidates.extend([
        Path("/usr/share/proj"),
        Path("/usr/local/share/proj"),
    ])
    for c in candidates:
        if c.exists() and (c / "proj.db").exists():
            return c
    return candidates[0]  # fallback

def _find_gdal_data():
    """Find GDAL data dir, cross-platform."""
    candidates = []
    try:
        from osgeo import gdal
        pkg_dir = Path(gdal.__file__).parent / "data" / "gdal"
        candidates.append(pkg_dir)
        # Also try osgeo package level
        candidates.append(Path(gdal.__file__).parent / "data")
    except ImportError:
        pass
    candidates.extend([
        Path("/usr/share/gdal"),
        Path("/usr/local/share/gdal"),
    ])
    for c in candidates:
        if c.exists():
            return c
    return candidates[0] if candidates else Path("/usr/share/gdal")

proj_lib_path = _find_proj_data()
gdal_data_path = _find_gdal_data()

if not (proj_lib_path / "proj.db").exists():
    print(f"[WARNING] proj.db not found in any known location!")
if not gdal_data_path.exists():
    print(f"[WARNING] GDAL data directory not found at {gdal_data_path}.")

# Set ALL PROJ env vars at once BEFORE any rasterio/pyproj imports.
# This is the only configuration needed — no need to import pyproj here.
os.environ["PROJ_LIB"]  = str(proj_lib_path)
os.environ["PROJ_DATA"] = str(proj_lib_path)   # newer PROJ versions
os.environ["GDAL_DATA"] = str(gdal_data_path)
os.environ["PROJ_IGNORE_CELESTIAL_BODY"] = "1"
os.environ["PROJ_NETWORK"] = "OFF"             # disable slow network grid lookups

# Enable GDAL HTTP access for Cloud Optimized GeoTIFF files on GCS
os.environ["GDAL_HTTP_UNSAFESSL"]              = "YES"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif,.tiff"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"]     = "EMPTY_DIR"
os.environ["VSI_CACHE"]                        = "TRUE"
os.environ["VSI_CACHE_SIZE"]                    = "67108864"  # 64 MB

print(f"[PROJ] PROJ_LIB set to: {os.environ['PROJ_LIB']}  ({_elapsed()})")
print(f"[PROJ] proj.db exists: {(proj_lib_path / 'proj.db').exists()}")

# ============================================================================
# Now safe to import Flask and other modules
# ============================================================================
print(f"[STARTUP] Importing Flask and service modules...  ({_elapsed()})")
from flask import Flask, jsonify
from flask_cors import CORS
import logging

# Add backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import MVC components
from config import get_config, Config
from config.raster_config import get_raster_config
from services import GEEService, AIService
from services.weather_service import NASAPowerService
from services.feature_engineering_service import FeatureEngineeringService
from services.raster_data_service import RasterDataService
from services.post_disaster_weather_service import PostDisasterWeatherService
from services.post_disaster_feature_engineering_service import PostDisasterFeatureEngineeringService
from services.hazardguard_prediction_service import HazardGuardPredictionService
print(f"[STARTUP] Importing WeatherWise (triggers TensorFlow load)...  ({_elapsed()})")
from services.weatherwise_prediction_service import WeatherWisePredictionService
print(f"[STARTUP] TensorFlow ready.  ({_elapsed()})")
from services.geovision_fusion_service import GeoVisionFusionService
from controllers import ChatController, SatelliteController
from controllers.weather_controller import WeatherController
from controllers.feature_engineering_controller import FeatureEngineeringController
from controllers.raster_data_controller import RasterDataController
from controllers.post_disaster_weather_controller import PostDisasterWeatherController
from controllers.post_disaster_feature_engineering_controller import PostDisasterFeatureEngineeringController
from controllers.hazardguard_prediction_controller import HazardGuardPredictionController
from controllers.weatherwise_prediction_controller import WeatherWisePredictionController
from controllers.geovision_fusion_controller import GeoVisionFusionController
from controllers.auth_controller import AuthController
from views import chat_bp, satellite_bp, legacy_bp, init_chat_routes, init_satellite_routes, init_legacy_routes
from routes.weather_routes import weather_bp, init_weather_routes
from routes.feature_routes import features_bp, init_feature_routes
from routes.raster_routes import create_raster_routes
from routes.post_disaster_weather_routes import create_post_disaster_weather_routes
from routes.post_disaster_feature_engineering_routes import post_disaster_feature_engineering_bp
from routes.hazardguard_prediction_routes import hazardguard_bp
from routes.weatherwise_prediction_routes import weatherwise_bp
from routes.geovision_fusion_routes import geovision_bp
from routes.auth_routes import auth_bp, init_auth_routes
from routes.layers_routes import layers_bp
from routes.credits_routes import credits_bp, init_credits_routes
from routes.api_keys_routes import api_keys_bp
from utils import setup_logging, create_error_response, create_success_response
print(f"[STARTUP] All modules imported.  ({_elapsed()})")

def create_app(config_name: str = None) -> Flask:
    """
    Application factory for creating Flask app with MVC architecture
    
    Args:
        config_name: Configuration name to use
        
    Returns:
        Configured Flask application
    """
    # Create Flask app
    app = Flask(__name__)
    
    # Load configuration
    if config_name:
        os.environ['FLASK_ENV'] = config_name
    
    config_class = get_config()
    app.config.from_object(config_class)
    
    # Setup logging
    setup_logging(
        log_level=config_class.LOG_LEVEL,
        log_file=config_class.LOG_FILE
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting {config_class.APP_NAME} v{config_class.APP_VERSION}")
    
    # Validate configuration — log warnings but do NOT crash.
    # Missing keys (e.g. GEMINI_API_KEY) just disable the relevant feature;
    # the server still starts so other endpoints remain available.
    config_errors = config_class.validate()
    if config_errors:
        for error in config_errors:
            logger.warning(f"Configuration warning: {error} — related features will be unavailable")
    
    # Setup CORS
    CORS(app, origins=config_class.ALLOWED_ORIGINS, 
         methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
            allow_headers=['Content-Type', 'Authorization', 'X-Demo-Local-Credits', 'x-demo-local-credits'])
    
    # Initialize services
    services = initialize_services(config_class, logger)
    
    # Initialize controllers
    controllers = initialize_controllers(services, logger)
    
    # Store controllers in app extensions for blueprint access
    if not hasattr(app, 'extensions'):
        app.extensions = {}
    app.extensions['controllers'] = controllers
    
    # Register blueprints
    register_blueprints(app, controllers, logger)
    
    # Register error handlers
    register_error_handlers(app, logger)
    
    # Add health check and info endpoints
    register_system_routes(app, config_class, services, logger)
    
    logger.info("Application initialization completed successfully")
    return app

def initialize_services(config_class: Config, logger: logging.Logger) -> dict:
    """Initialize all services"""
    services = {}
    
    # Initialize Google Earth Engine service
    logger.info("Initializing Google Earth Engine service...")
    _t = time.time()
    gee_service = GEEService(
        project_id=config_class.GEE_PROJECT_ID,
        service_account_key=config_class.GEE_SERVICE_ACCOUNT_KEY
    )
    
    if gee_service.initialize():
        services['gee'] = gee_service
        logger.info(f"GEE service initialized successfully  ({time.time()-_t:.1f}s)")
    else:
        logger.error(f"GEE service initialization failed  ({time.time()-_t:.1f}s)")
        services['gee'] = gee_service  # Still add it for status reporting
    
    # Initialize AI service
    logger.info("Initializing AI service...")
    ai_service = AIService(config_class.GEMINI_API_KEY)
    
    if ai_service.initialize():
        services['ai'] = ai_service
        logger.info("AI service initialized successfully")
    else:
        logger.error("AI service initialization failed")
        services['ai'] = ai_service  # Still add it for status reporting
    
    # Initialize Weather service
    logger.info("Initializing NASA POWER weather service...")
    weather_service = NASAPowerService()
    services['weather'] = weather_service
    logger.info("Weather service initialized successfully")
    
    # Initialize Feature Engineering service
    logger.info("Initializing feature engineering service...")
    feature_service = FeatureEngineeringService()
    services['features'] = feature_service
    logger.info("Feature engineering service initialized successfully")
    
    # Initialize Raster Data service
    logger.info("Initializing raster data service...")
    raster_config = get_raster_config()
    raster_service = RasterDataService(raster_config.get_config())
    services['raster'] = raster_service
    logger.info("Raster data service initialized successfully")
    
    # Initialize Post-Disaster Weather service
    logger.info("Initializing post-disaster weather service...")
    post_disaster_weather_service = PostDisasterWeatherService(
        days_after_disaster=60,
        max_workers=1,
        retry_limit=5,
        retry_delay=15,
        rate_limit_pause=900,
        request_delay=0.5
    )
    services['post_disaster_weather'] = post_disaster_weather_service
    logger.info("Post-disaster weather service initialized successfully")
    
    # Initialize Post-Disaster Feature Engineering service
    logger.info("Initializing post-disaster feature engineering service...")
    post_disaster_feature_service = PostDisasterFeatureEngineeringService()
    services['post_disaster_features'] = post_disaster_feature_service
    logger.info("Post-disaster feature engineering service initialized successfully")
    
    # Initialize HazardGuard Prediction service
    logger.info("Initializing HazardGuard prediction service...")
    hazardguard_service = HazardGuardPredictionService(
        weather_service=services['weather'],
        feature_service=services['features'],
        raster_service=services['raster']
    )
    # Initialize the HazardGuard service (load model)
    hazard_success, hazard_message = hazardguard_service.initialize_service()
    if hazard_success:
        logger.info("HazardGuard service initialized and model loaded successfully")
    else:
        logger.warning(f"HazardGuard service initialization warning: {hazard_message}")
    
    services['hazardguard'] = hazardguard_service
    logger.info("HazardGuard prediction service setup completed")
    
    # Initialize WeatherWise Prediction service
    logger.info("Initializing WeatherWise prediction service...")
    weatherwise_service = WeatherWisePredictionService(
        weather_service=services['weather'],
        feature_service=services['features']
    )
    services['weatherwise'] = weatherwise_service

    # Initialize GeoVision Fusion service
    logger.info("Initializing GeoVision Fusion prediction service...")
    geovision_service = GeoVisionFusionService(
        weather_service=services['weather'],
        feature_service=services['features'],
        raster_service=services['raster'],
        gee_service=services.get('gee')
    )
    services['geovision'] = geovision_service

    # ── Background warm-up ──────────────────────────────────────────────────
    # TensorFlow takes ~90 s to import.  We load it in a daemon thread so
    # gunicorn can bind its port (and pass Render's health check) immediately.
    # Requests that arrive before warm-up is complete receive a 503 response.
    import threading

    def _background_warmup():
        import time as _time
        _t0 = _time.time()
        logger.info("[WARMUP] Background thread started — loading TF models...")
        try:
            ww_success, ww_msg = weatherwise_service.initialize_service()
            if ww_success:
                logger.info(f"[WARMUP] WeatherWise ready  ({_time.time()-_t0:.1f}s)")
            else:
                logger.warning(f"[WARMUP] WeatherWise warning: {ww_msg}  ({_time.time()-_t0:.1f}s)")
        except Exception as exc:
            logger.error(f"[WARMUP] WeatherWise init error: {exc}")
        try:
            gv_success, gv_msg = geovision_service.initialize_service()
            if gv_success:
                logger.info(f"[WARMUP] GeoVision Fusion ready  ({_time.time()-_t0:.1f}s)")
            else:
                logger.warning(f"[WARMUP] GeoVision Fusion warning: {gv_msg}  ({_time.time()-_t0:.1f}s)")
        except Exception as exc:
            logger.error(f"[WARMUP] GeoVision Fusion init error: {exc}")
        logger.info(f"[WARMUP] Background warm-up complete  ({_time.time()-_t0:.1f}s total)")

    _warmup_thread = threading.Thread(target=_background_warmup, name="tf-warmup", daemon=True)
    _warmup_thread.start()
    logger.info("[WARMUP] TF model loading started in background thread — port will bind immediately")
    
    # Initialize Auth service (Supabase)
    supabase_url = config_class.SUPABASE_URL
    supabase_key = config_class.SUPABASE_SERVICE_ROLE_KEY
    if supabase_url and supabase_key and supabase_key != 'YOUR_SERVICE_ROLE_KEY_HERE':
        logger.info("Initializing Supabase auth service...")
        try:
            from services.auth_service import AuthService
            auth_service = AuthService(supabase_url, supabase_key)
            services['auth'] = auth_service
            logger.info("Auth service initialized successfully")
        except Exception as e:
            logger.error(f"Auth service initialization failed: {e}")
            services['auth'] = None
    else:
        logger.warning("Supabase credentials not configured -- auth endpoints will be unavailable")
        services['auth'] = None
    
    return services

def initialize_controllers(services: dict, logger: logging.Logger) -> dict:
    """Initialize all controllers"""
    logger.info("Initializing controllers...")
    
    controllers = {
        'chat': ChatController(services['ai'], services['gee']),
        'satellite': SatelliteController(services['gee']),
        'weather': WeatherController(services['weather']),
        'features': FeatureEngineeringController(services['features']),
        'raster': RasterDataController(get_raster_config().get_config()),
        'post_disaster_weather': PostDisasterWeatherController(
            days_after_disaster=60,
            max_workers=1,
            retry_limit=5,
            retry_delay=15,
            rate_limit_pause=900,
            request_delay=0.5
        ),
        'post_disaster_features': PostDisasterFeatureEngineeringController(),
        'hazardguard': HazardGuardPredictionController(services['hazardguard']),
        'weatherwise': WeatherWisePredictionController(services['weatherwise']),
        'geovision': GeoVisionFusionController(services['geovision'])
    }
    
    # Auth controller (optional – only if Supabase is configured)
    if services.get('auth'):
        controllers['auth'] = AuthController(services['auth'])
    else:
        controllers['auth'] = None
    
    logger.info("Controllers initialized successfully")
    return controllers

def register_blueprints(app: Flask, controllers: dict, logger: logging.Logger):
    """Register all API blueprints"""
    logger.info("Registering API blueprints...")
    
    # Initialize route handlers with controllers
    init_chat_routes(controllers['chat'])
    init_satellite_routes(controllers['satellite'])
    init_legacy_routes(controllers['satellite'])  # Legacy routes use satellite controller
    init_weather_routes(controllers['weather'])  # Initialize weather routes
    init_feature_routes(controllers['features'])  # Initialize feature engineering routes
    
    # Initialize auth routes (if configured)
    if controllers.get('auth'):
        init_auth_routes(controllers['auth'])
        init_credits_routes(controllers['auth'])
        logger.info("Auth routes initialized")
    else:
        logger.warning("Auth controller not available -- skipping auth routes")
    
    # Create raster routes blueprint
    raster_bp = create_raster_routes(get_raster_config().get_config())
    
    # Create post-disaster weather routes blueprint
    post_disaster_weather_bp = create_post_disaster_weather_routes({
        'days_after_disaster': 60,
        'max_workers': 1,
        'retry_limit': 5,
        'retry_delay': 15,
        'rate_limit_pause': 900,
        'request_delay': 0.5
    })
    
    # Register blueprints
    app.register_blueprint(chat_bp)
    app.register_blueprint(satellite_bp)
    app.register_blueprint(legacy_bp)  # Register legacy routes for backwards compatibility
    app.register_blueprint(weather_bp)  # Register weather routes
    app.register_blueprint(features_bp)  # Register feature engineering routes
    app.register_blueprint(raster_bp)  # Register raster data routes
    app.register_blueprint(post_disaster_weather_bp)  # Register post-disaster weather routes
    app.register_blueprint(post_disaster_feature_engineering_bp, url_prefix='/api/post-disaster-features')  # Register post-disaster feature engineering routes
    app.register_blueprint(hazardguard_bp, url_prefix='/api/hazardguard')  # Register HazardGuard prediction routes
    app.register_blueprint(weatherwise_bp, url_prefix='/api/weatherwise')  # Register WeatherWise prediction routes
    app.register_blueprint(geovision_bp, url_prefix='/api/geovision')  # Register GeoVision Fusion prediction routes
    app.register_blueprint(auth_bp, url_prefix='/api')  # Register auth routes at /api/auth/*
    app.register_blueprint(layers_bp, url_prefix='/api')  # Register dashboard state/goa layer routes
    app.register_blueprint(credits_bp, url_prefix='/api')  # Register dashboard credits routes
    app.register_blueprint(api_keys_bp, url_prefix='/api')  # Register dashboard API keys routes
    
    logger.info("Blueprints registered successfully")
    
    # Log all registered routes for debugging
    logger.info("=== REGISTERED ROUTES ===")
    for rule in app.url_map.iter_rules():
        methods = ','.join(rule.methods)
        logger.info(f"{rule.rule} | {methods}")
    logger.info("=== END ROUTES ===")

def register_error_handlers(app: Flask, logger: logging.Logger):
    """Register global error handlers"""
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify(create_error_response(
            "Endpoint not found",
            {"path": error.description}
        )), 404
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify(create_error_response(
            "Method not allowed",
            {"allowed_methods": error.description}
        )), 405
    
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify(create_error_response(
            "Bad request",
            {"description": error.description}
        )), 400
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {str(error)}")
        return jsonify(create_error_response(
            "Internal server error"
        )), 500

def register_system_routes(app: Flask, config_class: Config, services: dict, logger: logging.Logger):
    """Register system-level routes"""
    
    @app.route('/', methods=['GET'])
    def root():
        """Root endpoint"""
        return jsonify(create_success_response({
            'message': f'Welcome to {config_class.APP_NAME}',
            'version': config_class.APP_VERSION,
            'status': 'running',
            'endpoints': {
                'health': '/health',
                'info': '/info',
                'chat': '/api/chat/*',
                'satellite': '/api/satellite/*',
                'weather': '/api/weather/*',
                'features': '/api/features/*',
                'raster': '/api/raster/*',
                'post_disaster_weather': '/api/post-disaster-weather/*',
                'post_disaster_features': '/api/post-disaster-features/*',
                'hazardguard': '/api/hazardguard/*',
                'geovision': '/api/geovision/*'
            }
        }))
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        try:
            # Check service health
            gee_healthy = services['gee'].initialized
            ai_healthy = services['ai'].initialized
            weather_healthy = services['weather'].get_service_status().get('initialized', True)
            features_healthy = services['features'].get_service_status().get('initialized', True)
            raster_healthy = services['raster'].get_processing_statistics().get('statistics', {}).get('service_status', 'healthy') == 'healthy'
            post_disaster_weather_healthy = services['post_disaster_weather'].get_service_status().get('status', 'healthy') in ['ready', 'healthy']
            post_disaster_features_healthy = services['post_disaster_features'].get_service_health().get('service_status', 'healthy') == 'healthy'
            hazardguard_healthy = services['hazardguard'].get_service_status().get('service_status', 'healthy') in ['ready', 'healthy']
            
            overall_health = 'healthy' if (gee_healthy and ai_healthy and weather_healthy and features_healthy and raster_healthy and post_disaster_weather_healthy and post_disaster_features_healthy and hazardguard_healthy) else 'degraded'
            
            health_data = {
                'status': overall_health,
                'services': {
                    'gee': 'healthy' if gee_healthy else 'unhealthy',
                    'ai': 'healthy' if ai_healthy else 'unhealthy',
                    'weather': 'healthy' if weather_healthy else 'unhealthy',
                    'features': 'healthy' if features_healthy else 'unhealthy',
                    'raster': 'healthy' if raster_healthy else 'unhealthy',
                    'post_disaster_weather': 'healthy' if post_disaster_weather_healthy else 'unhealthy',
                    'post_disaster_features': 'healthy' if post_disaster_features_healthy else 'unhealthy',
                    'hazardguard': 'healthy' if hazardguard_healthy else 'unhealthy'
                },
                'version': config_class.APP_VERSION,
                'environment': config_class.FLASK_ENV
            }
            
            status_code = 200 if overall_health == 'healthy' else 503
            return jsonify(create_success_response(health_data)), status_code
            
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            return jsonify(create_error_response(
                "Health check failed",
                {"error": str(e)}
            )), 500
    
    @app.route('/info', methods=['GET'])
    def app_info():
        """Application information endpoint"""
        return jsonify(create_success_response({
            'name': config_class.APP_NAME,
            'version': config_class.APP_VERSION,
            'author': config_class.APP_USER,
            'environment': config_class.FLASK_ENV,
            'debug': config_class.FLASK_DEBUG,
            'gee_project': config_class.GEE_PROJECT_ID,
            'cors_origins': config_class.ALLOWED_ORIGINS,
            'api_endpoints': {
                'chat_message': 'POST /api/chat/message',
                'chat_analyze': 'POST /api/chat/analyze',
                'chat_disaster_info': 'GET /api/chat/disaster/<type>',
                'satellite_point': 'GET|POST /api/satellite/point',
                'satellite_region': 'POST /api/satellite/region',
                'satellite_availability': 'GET|POST /api/satellite/availability',
                'satellite_status': 'GET /api/satellite/status',
                'satellite_collections': 'GET /api/satellite/collections',
                'weather_data': 'GET|POST /api/weather/data',
                'weather_time_series': 'GET|POST /api/weather/time-series',
                'weather_batch': 'POST /api/weather/batch',
                'weather_summary': 'GET|POST /api/weather/summary',
                'weather_fields': 'GET /api/weather/fields',
                'weather_status': 'GET /api/weather/status',
                'weather_test': 'GET /api/weather/test',
                'features_process': 'POST /api/features/process',
                'features_batch': 'POST /api/features/batch',
                'features_dataframe': 'POST /api/features/dataframe',
                'features_validate': 'POST /api/features/validate',
                'features_export': 'POST /api/features/export',
                'features_info': 'GET /api/features/info',
                'features_status': 'GET /api/features/status',
                'features_test': 'GET|POST /api/features/test',
                'raster_process': 'POST /api/raster/process',
                'raster_batch': 'POST /api/raster/batch',
                'raster_dataframe': 'POST /api/raster/dataframe',
                'raster_export': 'POST /api/raster/export',
                'raster_validate': 'POST /api/raster/validate',
                'raster_features': 'GET /api/raster/features',
                'raster_info': 'GET /api/raster/info',
                'raster_status': 'GET /api/raster/status',
                'raster_test': 'GET /api/raster/test',
                'raster_health': 'GET /api/raster/health',
                'post_disaster_features_process': 'POST /api/post-disaster-features/process',
                'post_disaster_features_batch': 'POST /api/post-disaster-features/batch',
                'post_disaster_features_export_csv': 'POST /api/post-disaster-features/export/csv',
                'post_disaster_features_validate_coordinates': 'POST /api/post-disaster-features/validate/coordinates',
                'post_disaster_features_validate_weather': 'POST /api/post-disaster-features/validate/weather',
                'post_disaster_features_info': 'GET /api/post-disaster-features/features/info',
                'post_disaster_features_health': 'GET /api/post-disaster-features/health',
                'post_disaster_features_reset_stats': 'POST /api/post-disaster-features/statistics/reset',
                'post_disaster_features_ping': 'GET /api/post-disaster-features/ping',
                'hazardguard_predict': 'POST /api/hazardguard/predict',
                'hazardguard_predict_batch': 'POST /api/hazardguard/predict/batch',
                'hazardguard_capabilities': 'GET /api/hazardguard/capabilities',
                'hazardguard_validate_coordinates': 'POST /api/hazardguard/validate/coordinates',
                'hazardguard_health': 'GET /api/hazardguard/health',
                'hazardguard_initialize': 'POST /api/hazardguard/initialize',
                'hazardguard_reset_stats': 'POST /api/hazardguard/statistics/reset',
                'hazardguard_ping': 'GET /api/hazardguard/ping'
            }
        }))

# ── Module-level app for WSGI servers (gunicorn, etc.) ──
# gunicorn will import main:app directly, bypassing main()
app = create_app()

def main():
    """Main entry point for local development"""
    try:
        # Get configuration
        config_class = get_config()
        
        # Run application
        print(f"\n[START] Starting {config_class.APP_NAME} v{config_class.APP_VERSION}")
        print(f"[ENV] Environment: {config_class.FLASK_ENV}")
        print(f"[SERVER] Server: http://{config_class.FLASK_HOST}:{config_class.FLASK_PORT}")
        print(f"[HEALTH] Health Check: http://{config_class.FLASK_HOST}:{config_class.FLASK_PORT}/health")
        print(f"[INFO] API Info: http://{config_class.FLASK_HOST}:{config_class.FLASK_PORT}/info\n")
        
        app.run(
            host=config_class.FLASK_HOST,
            port=config_class.FLASK_PORT,
            debug=config_class.FLASK_DEBUG,
            use_reloader=False  # Disable reloader to prevent double initialization of heavy services (GEE, Gemini, models)
        )
        
    except KeyboardInterrupt:
        print("\n[STOP] Server stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Server startup failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()