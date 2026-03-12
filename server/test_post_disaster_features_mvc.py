"""
Comprehensive test script for Post-Disaster Feature Engineering MVC System
Tests all components: Model, Service, Controller, Routes, and Integration
"""

import sys
import os
import logging
import json
import unittest
from datetime import datetime, timedelta
import numpy as np

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import components to test
from models.post_disaster_feature_engineering_model import PostDisasterFeatureEngineeringModel
from services.post_disaster_feature_engineering_service import PostDisasterFeatureEngineeringService
from controllers.post_disaster_feature_engineering_controller import PostDisasterFeatureEngineeringController
from routes.post_disaster_feature_engineering_routes import post_disaster_feature_engineering_bp

# Test data generator
class TestDataGenerator:
    """Generate realistic test data for post-disaster feature engineering"""
    
    @staticmethod
    def generate_random_weather_data(days: int = 60, coordinate: tuple = (12.9716, 77.5946)) -> dict:
        """Generate realistic weather data for testing - Bangalore, India as default"""
        np.random.seed(42)  # For reproducible tests
        
        # Base values for Bangalore climate
        base_temp = 25.0
        temp_variation = 8.0
        
        weather_data = {}
        
        # Generate 60 days of weather data with some seasonal variation and random noise
        for i in range(days):
            # Seasonal variation (simulated)
            seasonal_factor = np.sin(i * 2 * np.pi / 365) * 0.3
            
            # Daily temperature with variation
            temp = base_temp + seasonal_factor * temp_variation + np.random.normal(0, 2)
            temp_max = temp + np.random.uniform(3, 8)
            temp_min = temp - np.random.uniform(2, 6)
            
            # Humidity (higher during monsoon season)
            humidity = np.clip(60 + seasonal_factor * 20 + np.random.normal(0, 10), 20, 95)
            
            # Initialize daily values if first iteration
            if i == 0:
                weather_data = {
                    'POST_temperature_C': [],
                    'POST_humidity_%': [],
                    'POST_wind_speed_mps': [],
                    'POST_precipitation_mm': [],
                    'POST_surface_pressure_hPa': [],
                    'POST_solar_radiation_wm2': [],
                    'POST_temperature_max_C': [],
                    'POST_temperature_min_C': [],
                    'POST_specific_humidity_g_kg': [],
                    'POST_dew_point_C': [],
                    'POST_wind_speed_10m_mps': [],
                    'POST_cloud_amount_%': [],
                    'POST_sea_level_pressure_hPa': [],
                    'POST_surface_soil_wetness_%': [],
                    'POST_wind_direction_10m_degrees': [],
                    'POST_evapotranspiration_wm2': [],
                    'POST_root_zone_soil_moisture_%': []
                }
            
            # Append daily weather values
            weather_data['POST_temperature_C'].append(round(temp, 2))
            weather_data['POST_temperature_max_C'].append(round(temp_max, 2))
            weather_data['POST_temperature_min_C'].append(round(temp_min, 2))
            weather_data['POST_humidity_%'].append(round(humidity, 1))
            weather_data['POST_specific_humidity_g_kg'].append(round(humidity * 0.15, 2))
            weather_data['POST_dew_point_C'].append(round(temp - (100 - humidity) / 5, 1))
            weather_data['POST_wind_speed_mps'].append(round(np.random.uniform(1, 8), 1))
            weather_data['POST_wind_speed_10m_mps'].append(round(np.random.uniform(1, 10), 1))
            weather_data['POST_wind_direction_10m_degrees'].append(round(np.random.uniform(0, 360), 0))
            weather_data['POST_precipitation_mm'].append(round(max(0, np.random.exponential(3) if np.random.random() > 0.7 else 0), 1))
            weather_data['POST_surface_pressure_hPa'].append(round(1013 + np.random.normal(0, 8), 1))
            weather_data['POST_sea_level_pressure_hPa'].append(round(1013 + np.random.normal(0, 6), 1))
            weather_data['POST_solar_radiation_wm2'].append(round(np.random.uniform(150, 300), 1))
            weather_data['POST_cloud_amount_%'].append(round(np.random.uniform(10, 80), 1))
            weather_data['POST_surface_soil_wetness_%'].append(round(np.random.uniform(20, 60), 1))
            weather_data['POST_root_zone_soil_moisture_%'].append(round(np.random.uniform(25, 55), 1))
            weather_data['POST_evapotranspiration_wm2'].append(round(np.random.uniform(80, 150), 1))
        
        return weather_data

def test_model_functionality():
    """Test the PostDisasterFeatureEngineeringModel"""
    print("\\n=== Testing Model Functionality ===")
    
    # Initialize model
    model = PostDisasterFeatureEngineeringModel(days_count=60)
    
    # Generate test data
    test_generator = TestDataGenerator()
    weather_data = test_generator.generate_random_weather_data()
    
    print(f"✓ Model initialized with {len(model.POST_WEATHER_VARIABLES)} input variables and {len(model.POST_FEATURE_VARIABLES)} output features")
    
    # Test single coordinate processing
    result = model.engineer_single_coordinate_features(weather_data)
    
    if result['success']:
        features = result['features']
        print(f"✓ Single coordinate feature engineering successful")
        print(f"  - Features created: {len(features)}")
        print(f"  - Days processed: {len(next(iter(features.values()), []))}")
        
        # Check for expected features
        expected_features = ['POST_temp_normalized', 'POST_heat_index', 'POST_drought_indicator']
        for feature in expected_features:
            if feature in features:
                values = features[feature]
                nan_count = sum(1 for v in values if np.isnan(v))
                print(f"  - {feature}: {len(values)} values, {nan_count} NaN values")
        
        return True
    else:
        print(f"✗ Model test failed: {result.get('error', 'Unknown error')}")
        return False

def test_service_functionality():
    """Test the PostDisasterFeatureEngineeringService"""
    print("\\n=== Testing Service Functionality ===")
    
    # Initialize service
    service = PostDisasterFeatureEngineeringService()
    
    # Generate test data
    test_generator = TestDataGenerator()
    weather_data = test_generator.generate_random_weather_data()
    coordinate = [12.9716, 77.5946]  # Bangalore
    
    print("✓ Service initialized")
    
    # Test coordinate validation
    is_valid, message, parsed_coords = service.validate_coordinates([coordinate])
    if is_valid:
        print(f"✓ Coordinate validation passed: {message}")
    else:
        print(f"✗ Coordinate validation failed: {message}")
        return False
    
    # Test weather data validation  
    is_valid, message, parsed_weather = service.validate_weather_data(weather_data)
    if is_valid:
        print(f"✓ Weather data validation passed: {message}")
    else:
        print(f"✗ Weather data validation failed: {message}")
        return False
    
    # Test single coordinate processing
    result = service.process_single_coordinate_features(weather_data, coordinate)
    
    if result['success']:
        print(f"✓ Service single coordinate processing successful")
        print(f"  - Processing time: {result['processing_time_seconds']:.3f}s")
        print(f"  - Features created: {len(result['features']) if result['features'] else 0}")
        
        # Test batch processing with multiple coordinates
        coordinates = [[12.9716, 77.5946], [28.6139, 77.2090]]  # Bangalore, Delhi
        weather_datasets = [weather_data, test_generator.generate_random_weather_data()]
        
        batch_result = service.process_batch_features(weather_datasets, coordinates)
        
        if batch_result['success']:
            print(f"✓ Service batch processing successful")
            print(f"  - Coordinates processed: {batch_result['total_coordinates']}")
            print(f"  - Success rate: {batch_result['success_rate_percent']:.1f}%")
            print(f"  - Processing time: {batch_result['processing_time_seconds']:.3f}s")
            
            return True
        else:
            print(f"✗ Service batch processing failed: {batch_result.get('error', 'Unknown error')}")
            return False
    else:
        print(f"✗ Service test failed: {result.get('error', 'Unknown error')}")  
        return False

def test_controller_functionality():
    """Test the PostDisasterFeatureEngineeringController"""
    print("\\n=== Testing Controller Functionality ===")
    
    # Initialize controller
    controller = PostDisasterFeatureEngineeringController()
    
    # Generate test data
    test_generator = TestDataGenerator()
    weather_data = test_generator.generate_random_weather_data()
    coordinate = [12.9716, 77.5946]
    
    print("✓ Controller initialized")
    
    # Test coordinate validation
    coord_result = controller.validate_coordinates({'coordinates': [coordinate]})
    if coord_result['success']:
        print(f"✓ Controller coordinate validation passed: {coord_result['message']}")
    else:
        print(f"✗ Controller coordinate validation failed: {coord_result.get('error', 'Unknown error')}")
        return False
    
    # Test weather data validation
    weather_result = controller.validate_weather_input({'weather_data': weather_data})
    if weather_result['success']:
        print(f"✓ Controller weather validation passed: {weather_result['message']}")
    else:
        print(f"✗ Controller weather validation failed: {weather_result.get('error', 'Unknown error')}")
        return False
    
    # Test feature processing
    process_request = {
        'weather_data': weather_data,
        'coordinate': coordinate
    }
    
    process_result = controller.process_single_coordinate_features(process_request)
    
    if process_result['success']:
        print(f"✓ Controller processing successful: {process_result['message']}")
        print(f"  - Processing time: {process_result['processing_info']['processing_time_seconds']:.3f}s")
        print(f"  - Features count: {process_result['processing_info']['features_count']}")
        
        # Test feature info
        info_result = controller.get_feature_info()
        if info_result['success']:
            print(f"✓ Controller feature info retrieved")
            print(f"  - Input variables: {info_result['data']['input_variables']['count']}")
            print(f"  - Output features: {info_result['data']['output_features']['count']}")
            
            # Test health check
            health_result = controller.get_service_health()
            if health_result['success']:
                print(f"✓ Controller health check passed")
                print(f"  - Service status: {health_result['data']['service_status']}")
                return True
            else:
                print(f"✗ Controller health check failed: {health_result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"✗ Controller feature info failed: {info_result.get('error', 'Unknown error')}")
            return False
    else:
        print(f"✗ Controller processing failed: {process_result.get('error', 'Unknown error')}")
        return False

def test_routes_functionality():
    """Test the blueprint functionality"""
    print("\\n=== Testing Routes Functionality ===")
    
    # Import Flask for testing
    from flask import Flask
    
    # Create test app
    app = Flask(__name__)
    app.register_blueprint(post_disaster_feature_engineering_bp, url_prefix='/api/post-disaster-features')
    
    print("✓ Blueprint registered successfully")
    
    # Test route registration
    routes = []
    for rule in app.url_map.iter_rules():
        if rule.rule.startswith('/api/post-disaster-features'):
            routes.append(f"{rule.rule} | {','.join(rule.methods)}")
    
    expected_routes = [
        '/process', '/batch', '/export/csv', '/validate/coordinates', 
        '/validate/weather', '/features/info', '/health', 
        '/statistics/reset', '/ping'
    ]
    
    registered_routes = [route.split(' | ')[0].replace('/api/post-disaster-features', '') for route in routes]
    
    all_routes_found = all(route in registered_routes for route in expected_routes)
    
    if all_routes_found:
        print(f"✓ All expected routes registered ({len(expected_routes)} routes)")
        for route in routes:
            print(f"  - {route}")
        return True
    else:
        missing_routes = [route for route in expected_routes if route not in registered_routes]
        print(f"✗ Missing routes: {missing_routes}")
        return False

def test_integration():
    """Test full integration"""
    print("\\n=== Testing Full Integration ===")
    
    try:
        # Test imports
        from main import initialize_services, initialize_controllers
        from config import get_config
        print("✓ All imports successful")
        
        # Check if new service is in the service initialization
        with open('main.py', 'r', encoding='utf-8', errors='ignore') as f:
            main_content = f.read()
        
        integration_checks = [
            'PostDisasterFeatureEngineeringService' in main_content,
            'PostDisasterFeatureEngineeringController' in main_content,
            'post_disaster_feature_engineering_bp' in main_content,
            'post_disaster_features' in main_content,
            '/api/post-disaster-features/' in main_content
        ]
        
        if all(integration_checks):
            print("✓ Integration successful - all components found in main.py")
            print("  - Service import and initialization ✓")
            print("  - Controller import and initialization ✓") 
            print("  - Blueprint import and registration ✓")
            print("  - URL prefix configuration ✓")
            print("  - Health check integration ✓")
            return True
        else:
            print("✗ Integration incomplete - missing components in main.py")
            return False
            
    except Exception as e:
        print(f"✗ Integration test failed: {str(e)}")
        return False

def run_comprehensive_test():
    """Run all tests"""
    print("🚀 Starting Comprehensive Post-Disaster Feature Engineering MVC Tests")
    print("=" * 80)
    
    test_results = {
        'model': False,
        'service': False, 
        'controller': False,
        'routes': False,
        'integration': False
    }
    
    # Run tests
    test_results['model'] = test_model_functionality()
    test_results['service'] = test_service_functionality()
    test_results['controller'] = test_controller_functionality()
    test_results['routes'] = test_routes_functionality()
    test_results['integration'] = test_integration()
    
    # Summary
    print("\\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for component, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{component.upper():>15}: {status}")
    
    print("-" * 40)
    print(f"Total: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("\\n🎉 ALL TESTS PASSED! Post-Disaster Feature Engineering MVC system is ready!")
        print("\\n📋 System Features:")
        print("  • 17 input weather variables")
        print("  • 19 engineered features")
        print("  • 60 days post-disaster analysis")
        print("  • Batch processing support")
        print("  • CSV export capabilities")
        print("  • Comprehensive validation")
        print("  • Health monitoring")
        print("  • Statistics tracking")
        print("\\n🌐 Available endpoints:")
        endpoints = [
            "POST /api/post-disaster-features/process",
            "POST /api/post-disaster-features/batch", 
            "POST /api/post-disaster-features/export/csv",
            "POST /api/post-disaster-features/validate/coordinates",
            "POST /api/post-disaster-features/validate/weather",
            "GET  /api/post-disaster-features/features/info",
            "GET  /api/post-disaster-features/health",
            "POST /api/post-disaster-features/statistics/reset",
            "GET  /api/post-disaster-features/ping"
        ]
        for endpoint in endpoints:
            print(f"  • {endpoint}")
        
        return True
    else:
        print(f"\\n❌ {total_tests - passed_tests} TEST(S) FAILED! Please review failed components.")
        return False

if __name__ == '__main__':
    try:
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\\n💥 Test runner crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)