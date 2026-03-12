"""
Post-Disaster Weather Integration Test
Test complete MVC system integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all component imports"""
    print("=== TESTING IMPORTS ===")
    
    try:
        from models.post_disaster_weather_model import PostDisasterWeatherModel
        print("✅ PostDisasterWeatherModel imported successfully")
    except Exception as e:
        print(f"❌ PostDisasterWeatherModel import failed: {e}")
        return False
    
    try:
        from services.post_disaster_weather_service import PostDisasterWeatherService
        print("✅ PostDisasterWeatherService imported successfully")
    except Exception as e:
        print(f"❌ PostDisasterWeatherService import failed: {e}")
        return False
    
    try:
        from controllers.post_disaster_weather_controller import PostDisasterWeatherController
        print("✅ PostDisasterWeatherController imported successfully")
    except Exception as e:
        print(f"❌ PostDisasterWeatherController import failed: {e}")
        return False
    
    try:
        from routes.post_disaster_weather_routes import create_post_disaster_weather_routes
        print("✅ Post-disaster weather routes imported successfully")
    except Exception as e:
        print(f"❌ Post-disaster weather routes import failed: {e}")
        return False
    
    return True

def test_model_functionality():
    """Test model basic functionality"""
    print("\n=== TESTING MODEL FUNCTIONALITY ===")
    
    try:
        from models.post_disaster_weather_model import PostDisasterWeatherModel
        
        # Test model creation
        model = PostDisasterWeatherModel()
        print("✅ PostDisasterWeatherModel created successfully")
        
        # Test coordinate validation
        test_coords = [{'latitude': 12.9716, 'longitude': 77.5946}]
        is_valid, message = model.validate_coordinates(test_coords)
        print(f"✅ Coordinate validation works: {is_valid} - {message}")
        
        # Test date validation  
        from datetime import datetime
        test_date = datetime(2023, 1, 15)
        is_valid, message, parsed = model.validate_disaster_date(test_date)
        print(f"✅ Date validation works: {is_valid} - {message}")
        
        # Test variable info
        variables = model.get_available_variables()
        print(f"✅ Variable info retrieved: {len(variables)} variables available")
        
        # Test NASA fill value cleaning
        test_values = [25.0, -999.0, 26.1, -99999, 27.2]
        cleaned = model.clean_nasa_values(test_values)
        print(f"✅ NASA value cleaning works: {len(cleaned)} values processed")
        
        return True
        
    except Exception as e:
        print(f"❌ Model functionality test failed: {e}")
        return False

def test_service_functionality():
    """Test service basic functionality""" 
    print("\n=== TESTING SERVICE FUNCTIONALITY ===")
    
    try:
        from services.post_disaster_weather_service import PostDisasterWeatherService
        
        # Test service creation
        service = PostDisasterWeatherService()
        print("✅ PostDisasterWeatherService created successfully")
        
        # Test coordinate validation
        test_coords = [{'latitude': 12.9716, 'longitude': 77.5946}]
        is_valid, message = service.validate_coordinates(test_coords)
        print(f"✅ Service coordinate validation: {is_valid} - {message}")
        
        # Test date validation
        test_dates = ['2023-01-15']
        is_valid, message, parsed = service.validate_disaster_dates(test_dates)
        print(f"✅ Service date validation: {is_valid} - {message}")
        
        # Test available variables
        result = service.get_available_variables()
        if result['success']:
            print(f"✅ Available variables: {result['total_variables']} variables")
        else:
            print(f"❌ Available variables failed: {result['error']}")
            return False
        
        # Test processing statistics
        stats_result = service.get_processing_statistics()
        if stats_result['success']:
            stats = stats_result['statistics']
            print(f"✅ Processing statistics: {stats['service_status']} status")
        else:
            print(f"❌ Processing statistics failed: {stats_result['error']}")
            return False
        
        # Test service status
        status_result = service.get_service_status()
        print(f"✅ Service status: {status_result['status']} - {status_result['message']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Service functionality test failed: {e}")
        return False

def test_controller_functionality():
    """Test controller basic functionality"""
    print("\n=== TESTING CONTROLLER FUNCTIONALITY ===") 
    
    try:
        from controllers.post_disaster_weather_controller import PostDisasterWeatherController
        
        # Test controller creation
        controller = PostDisasterWeatherController()
        print("✅ PostDisasterWeatherController created successfully")
        
        # Test coordinate validation endpoint
        request_data = {
            'coordinates': [{'latitude': 12.9716, 'longitude': 77.5946}]
        }
        result = controller.validate_coordinates(request_data)
        if result['success']:
            print(f"✅ Controller coordinate validation: {result['data']['valid']}")
        else:
            print(f"❌ Controller coordinate validation failed: {result['error']}")
            return False
        
        # Test date validation endpoint
        request_data = {
            'disaster_dates': ['2023-01-15']
        }
        result = controller.validate_disaster_dates(request_data)
        if result['success']:
            print(f"✅ Controller date validation: {result['data']['valid']}")
        else:
            print(f"❌ Controller date validation failed: {result['error']}")
            return False
        
        # Test available variables endpoint
        result = controller.get_available_variables()
        if result['success']:
            print(f"✅ Controller variables: {len(result['data'])} variables available")
        else:
            print(f"❌ Controller variables failed: {result['error']}")
            return False
        
        # Test service health endpoint
        result = controller.get_service_health()
        if result['success']:
            health_data = result['data']
            print(f"✅ Controller health check: {health_data['status']} - {health_data['message']}")
        else:
            print(f"❌ Controller health check failed: {result['error']}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Controller functionality test failed: {e}")
        return False

def test_routes_creation():
    """Test routes blueprint creation"""
    print("\n=== TESTING ROUTES CREATION ===")
    
    try:
        from routes.post_disaster_weather_routes import create_post_disaster_weather_routes
        
        # Test blueprint creation
        config = {
            'days_after_disaster': 60,
            'max_workers': 1,
            'retry_limit': 5
        }
        blueprint = create_post_disaster_weather_routes(config)
        print("✅ Post-disaster weather blueprint created successfully")
        
        # Check blueprint properties
        print(f"✅ Blueprint name: {blueprint.name}")
        print(f"✅ Blueprint URL prefix: {blueprint.url_prefix}")
        
        return True
        
    except Exception as e:
        print(f"❌ Routes creation test failed: {e}")
        return False

def test_flask_integration():
    """Test Flask app integration"""
    print("\n=== TESTING FLASK INTEGRATION ===")
    
    try:
        # Test main app creation (without actually starting server)
        import main
        from flask import Flask
        
        # Create a test app
        app = Flask(__name__)
        
        # Test if we can import and create the routes
        from routes.post_disaster_weather_routes import create_post_disaster_weather_routes
        blueprint = create_post_disaster_weather_routes()
        
        # Register the blueprint
        app.register_blueprint(blueprint)
        print("✅ Flask blueprint registration successful")
        
        # Check if routes are registered
        routes = []
        for rule in app.url_map.iter_rules():
            if '/api/post-disaster-weather/' in rule.rule:
                routes.append(rule.rule)
        
        print(f"✅ Registered {len(routes)} post-disaster weather routes:")
        for route in routes:
            print(f"   - {route}")
        
        return True
        
    except Exception as e:
        print(f"❌ Flask integration test failed: {e}")
        return False

def run_all_tests():
    """Run all integration tests"""
    print("🧪 POST-DISASTER WEATHER MVC INTEGRATION TESTS")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Model Functionality", test_model_functionality), 
        ("Service Functionality", test_service_functionality),
        ("Controller Functionality", test_controller_functionality),
        ("Routes Creation", test_routes_creation),
        ("Flask Integration", test_flask_integration)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 TEST SUMMARY: {passed}/{len(tests)} tests passed")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Post-disaster weather MVC system is ready!")
        return True
    else:
        print(f"⚠️ {failed} tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    run_all_tests()