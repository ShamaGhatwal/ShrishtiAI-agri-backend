"""
Views Package  
API routes and request handlers
"""
from .chat_routes import chat_bp, init_chat_routes
from .satellite_routes import satellite_bp, init_satellite_routes
from .legacy_routes import legacy_bp, init_legacy_routes

__all__ = ['chat_bp', 'satellite_bp', 'legacy_bp', 'init_chat_routes', 'init_satellite_routes', 'init_legacy_routes']