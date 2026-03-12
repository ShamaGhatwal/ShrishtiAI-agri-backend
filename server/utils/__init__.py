"""
Utility Functions
Common utility functions for the application
"""
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
from functools import wraps
import time

def setup_logging(log_level: str = 'INFO', log_file: str = None) -> logging.Logger:
    """
    Set up logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    # Convert string level to logging constant
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger

def validate_coordinates(latitude: float, longitude: float) -> Tuple[bool, Optional[str]]:
    """
    Validate latitude and longitude coordinates
    
    Args:
        latitude: Latitude value
        longitude: Longitude value
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        lat = float(latitude)
        lon = float(longitude)
        
        if not (-90 <= lat <= 90):
            return False, f"Latitude must be between -90 and 90, got {lat}"
        
        if not (-180 <= lon <= 180):
            return False, f"Longitude must be between -180 and 180, got {lon}"
        
        return True, None
        
    except (ValueError, TypeError):
        return False, "Coordinates must be valid numbers"

def validate_date_string(date_string: str) -> Tuple[bool, Optional[str]]:
    """
    Validate date string in YYYY-MM-DD format
    
    Args:
        date_string: Date string to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True, None
    except ValueError:
        return False, f"Invalid date format: {date_string}. Expected YYYY-MM-DD"

def validate_date_range(start_date: str, end_date: str) -> Tuple[bool, Optional[str]]:
    """
    Validate date range
    
    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Validate individual dates
    start_valid, start_error = validate_date_string(start_date)
    if not start_valid:
        return False, f"Start date error: {start_error}"
    
    end_valid, end_error = validate_date_string(end_date)
    if not end_valid:
        return False, f"End date error: {end_error}"
    
    # Check date order
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        if start_dt >= end_dt:
            return False, "Start date must be before end date"
        
        # Check if date range is reasonable (not more than 1 year)
        if (end_dt - start_dt).days > 365:
            return False, "Date range cannot exceed 1 year"
        
        return True, None
        
    except ValueError as e:
        return False, f"Date parsing error: {str(e)}"

def get_default_date_range(days_back: int = 30) -> Tuple[str, str]:
    """
    Get default date range (end_date = today, start_date = days_back ago)
    
    Args:
        days_back: Number of days to go back
        
    Returns:
        Tuple of (start_date, end_date) in YYYY-MM-DD format
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file operations
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace dangerous characters
    dangerous_chars = '<>:"/\\|?*'
    for char in dangerous_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Ensure it's not empty
    if not filename:
        filename = 'unnamed'
    
    return filename

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def retry_on_exception(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator for retrying functions on exception
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier for delay
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        break
            
            raise last_exception
        return wrapper
    return decorator

def safe_json_response(data: Any, status_code: int = 200) -> Dict[str, Any]:
    """
    Create a safe JSON response with proper error handling
    
    Args:
        data: Data to include in response
        status_code: HTTP status code
        
    Returns:
        Response dictionary
    """
    try:
        if isinstance(data, dict):
            return data
        elif hasattr(data, 'to_dict'):
            return data.to_dict()
        else:
            return {'data': data, 'status': 'success'}
    except Exception as e:
        return {
            'error': f'Failed to serialize response: {str(e)}',
            'status': 'error'
        }

def calculate_bounds_center(bounds: List[List[float]]) -> Tuple[float, float]:
    """
    Calculate the center point of a bounds polygon
    
    Args:
        bounds: List of [longitude, latitude] coordinate pairs
        
    Returns:
        Tuple of (latitude, longitude) for center point
    """
    if not bounds or len(bounds) == 0:
        raise ValueError("Bounds cannot be empty")
    
    total_lat = sum(coord[1] for coord in bounds)
    total_lon = sum(coord[0] for coord in bounds)
    
    center_lat = total_lat / len(bounds)
    center_lon = total_lon / len(bounds)
    
    return center_lat, center_lon

def is_valid_cloud_filter(cloud_filter: Any) -> bool:
    """
    Validate cloud filter value
    
    Args:
        cloud_filter: Cloud filter value to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        value = int(cloud_filter)
        return 0 <= value <= 100
    except (ValueError, TypeError):
        return False

def create_error_response(error_message: str, details: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create standardized error response
    
    Args:
        error_message: Error message
        details: Optional error details
        
    Returns:
        Error response dictionary
    """
    response = {
        'error': error_message,
        'status': 'error',
        'timestamp': datetime.now().isoformat()
    }
    
    if details:
        response['details'] = details
    
    return response

def create_success_response(data: Any, message: str = None) -> Dict[str, Any]:
    """
    Create standardized success response
    
    Args:
        data: Response data
        message: Optional success message
        
    Returns:
        Success response dictionary
    """
    response = {
        'status': 'success',
        'timestamp': datetime.now().isoformat()
    }
    
    if message:
        response['message'] = message
    
    if data is not None:
        if isinstance(data, dict):
            response.update(data)
        else:
            response['data'] = data
    
    return response