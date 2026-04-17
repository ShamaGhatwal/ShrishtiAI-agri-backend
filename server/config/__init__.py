"""
Configuration management for GEO VISION Backend
Loads environment variables and provides configuration settings
"""
import os
from dotenv import load_dotenv

from .paths import SERVER_DIR, get_local_model_root, get_model_repo_id

# Load environment variables from the server-local .env file regardless of cwd.
load_dotenv(SERVER_DIR / '.env')

class Config:
    """Base configuration class"""
    
    # Google Earth Engine
    GEE_PROJECT_ID = os.getenv('GEE_PROJECT_ID', 'geovision-final')
    GEE_SERVICE_ACCOUNT_KEY = os.getenv('GEE_SERVICE_ACCOUNT_KEY', '')
    
    # Gemini AI
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    # Flask
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    FLASK_HOST = os.getenv('FLASK_HOST', '127.0.0.1')
    FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
    
    # Application
    APP_NAME = os.getenv('APP_NAME', 'Geo Vision Backend')
    APP_VERSION = os.getenv('APP_VERSION', '1.0.0')
    APP_USER = os.getenv('APP_USER', 'ShrishtiAI')
    
    # CORS
    ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000').split(',')
    
    # Supabase
    SUPABASE_URL = os.getenv('SUPABASE_URL', '')
    SUPABASE_SERVICE_ROLE_KEY = os.getenv('SUPABASE_SERVICE_ROLE_KEY', '')

    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'geovision.log')

    # Model storage
    MODEL_ROOT_PATH = str(get_local_model_root())
    MODEL_REPO_ID = get_model_repo_id()
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        errors = []
        
        if not cls.GEE_PROJECT_ID:
            errors.append("GEE_PROJECT_ID is required")
            
        if not cls.GEMINI_API_KEY:
            errors.append("GEMINI_API_KEY is required")
            
        return errors

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get configuration based on environment"""
    env = os.getenv('FLASK_ENV', 'development')
    return config.get(env, config['default'])