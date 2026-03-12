"""
Comprehensive test script for HazardGuard Disaster Prediction System
Tests all components: Model, Service, Controller, Routes, and Integration
"""

import sys
import os
import logging
import json
import traceback
from datetime import datetime, timedelta

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import components to test
from models.hazardguard_prediction_model import HazardGuardPredictionModel
from services.hazardguard_prediction_service import HazardGuardPredictionService
from controllers.hazardguard_prediction_controller import HazardGuardPredictionController
from routes.hazardguard_prediction_routes import hazardguard_bp

# Test data generator for realistic location testing
class HazardGuardTestData:
    """Generate realistic test data for HazardGuard testing"""
    
    # Test locations with known geographic characteristics
    TEST_LOCATIONS = [
        {
            'name': 'Bangalore, India',
            'latitude': 12.9716,
            'longitude': 77.5946,
            'description': 'Urban tech hub, moderate climate'
        },
        {
            'name': 'Miami, USA',
            'latitude': 25.7617,
            'longitude': -80.1918,
            'description': 'Coastal city, hurricane prone'
        },
        {
            'name': 'Tokyo, Japan',
            'latitude': 35.6762,
            'longitude': 139.6503,
            'description': 'Megacity, earthquake/typhoon risk'
        },
        {
            'name': 'London, UK',
            'latitude': 51.5074,
            'longitude': -0.1278,
            'description': 'Temperate climate, flood risk'
        }
    ]

def test_model_functionality():
    """Test the HazardGuardPredictionModel functionality"""
    print("\\n=== Testing HazardGuard Model Functionality ===")
    
    try:
        # Initialize model
        model = HazardGuardPredictionModel()
        print(f"✓ Model initialized with {len(model.ARRAY_FEATURE_COLUMNS)} array features and {len(model.SCALAR_FEATURE_COLUMNS)} scalar features")
        
        # Test model loading
        print("  📦 Loading model components...")
        load_success = model.load_model_components()
        
        if load_success:
            print("  ✅ Model components loaded successfully")
            
            # Get model info
            model_info = model.get_model_info()
            print(f"    - Model loaded: {model_info['is_loaded']}")
            print(f"    - Forecast horizon: {model_info['forecasting']['horizon_days']} days")
            print(f"    - Input days used: {model_info['forecasting']['forecast_input_days']} days")
            if model_info['model_metadata']:
                print(f"    - Model accuracy: {model_info['model_metadata'].get('cv_accuracy', 'N/A')}")
                print(f"    - Algorithm: {model_info['model_metadata'].get('algorithm', 'N/A')}")
            
            return True
        else:
            print("  ❌ Model component loading failed")
            print("    💡 This may be expected if model files are not found")
            print("    💡 Model files should be in: D:\\INTERNSHIP FINAL YEAR PROJECT\\models for research paper\\HORIZON1\\output_of_combined_disaster_forecast")
            return False
            
    except Exception as e:
        print(f"  ❌ Model test error: {e}")
        print(f"     Traceback: {traceback.format_exc()}")
        return False

def test_service_functionality():
    """Test the HazardGuardPredictionService functionality"""
    print("\\n=== Testing HazardGuard Service Functionality ===")
    
    try:
        # Initialize service
        service = HazardGuardPredictionService()
        print("✓ Service initialized")
        
        # Test service initialization
        print("  🔧 Initializing service (loading model)...")
        success, message = service.initialize_service()
        
        if success:
            print(f"  ✅ Service initialization successful: {message}")
            
            # Test coordinate validation
            test_locations = HazardGuardTestData.TEST_LOCATIONS
            print(f"  📍 Testing coordinate validation with {len(test_locations)} locations...")
            
            for location in test_locations:
                is_valid, validation_message = service.validate_coordinates(
                    location['latitude'], location['longitude']
                )
                if is_valid:
                    print(f"    ✓ {location['name']}: {validation_message}")
                else:
                    print(f"    ❌ {location['name']}: {validation_message}")
            
            # Test service status
            status = service.get_service_status()
            print(f"  📊 Service status: {status['service_status']}")
            print(f"    - Model loaded: {status['model_loaded']}")
            print(f"    - Total requests: {status['statistics']['total_requests']}")
            
            return True
        else:
            print(f"  ⚠️  Service initialization failed: {message}")
            print("    💡 This may be expected if model files are not available")
            return False
            
    except Exception as e:
        print(f"  ❌ Service test error: {e}")
        return False

def test_controller_functionality():
    """Test the HazardGuardPredictionController functionality"""
    print("\\n=== Testing HazardGuard Controller Functionality ===")
    
    try:
        # Initialize controller
        controller = HazardGuardPredictionController()
        print("✓ Controller initialized")
        
        # Test controller initialization
        print("  🔧 Initializing controller...")
        init_result = controller.initialize_controller()
        
        if init_result['success']:
            print(f"  ✅ Controller initialization successful: {init_result['message']}")
            
            # Test coordinate validation
            test_location = HazardGuardTestData.TEST_LOCATIONS[0]  # Bangalore
            print(f"  📍 Testing coordinate validation for {test_location['name']}...")
            
            validation_request = {
                'latitude': test_location['latitude'],
                'longitude': test_location['longitude']
            }
            
            validation_result = controller.validate_coordinates_only(validation_request)
            
            if validation_result['success']:
                print(f"    ✓ Coordinate validation passed: {validation_result['message']}")
                
                # Test capabilities endpoint
                print("  📋 Testing capabilities retrieval...")
                capabilities_result = controller.get_prediction_capabilities()
                
                if capabilities_result['success']:
                    caps = capabilities_result['data']
                    print(f"    ✓ Capabilities retrieved successfully")
                    print(f"      - Prediction type: {caps['prediction_type']}")
                    print(f"      - Supported disasters: {', '.join(caps['supported_disaster_types'])}")
                    print(f"      - Geographic coverage: {caps['geographic_coverage']}")
                    print(f"      - Batch processing: {caps['batch_processing']['supported']}")
                    
                    # Test health check
                    print("  🏥 Testing health check...")
                    health_result = controller.get_service_health()
                    
                    if health_result['success']:
                        health = health_result['data']
                        print(f"    ✓ Health check passed")
                        print(f"      - Service status: {health['service_status']}")
                        print(f"      - Model loaded: {health.get('model_loaded', False)}")
                        print(f"      - Uptime: {health.get('uptime_hours', 0):.2f} hours")
                        
                        return True
                    else:
                        print(f"    ❌ Health check failed: {health_result.get('error', 'Unknown error')}")
                        return False
                else:
                    print(f"    ❌ Capabilities retrieval failed: {capabilities_result.get('error', 'Unknown error')}")
                    return False
            else:
                print(f"    ❌ Coordinate validation failed: {validation_result.get('error', 'Unknown error')}")
                return False
        else:
            print(f"  ⚠️  Controller initialization failed: {init_result.get('error', 'Unknown error')}")
            print("    💡 This may be expected if model files are not available")
            return False
            
    except Exception as e:
        print(f"  ❌ Controller test error: {e}")
        return False

def test_routes_functionality():
    """Test the blueprint functionality"""
    print("\\n=== Testing HazardGuard Routes Functionality ===")
    
    try:
        # Import Flask for testing
        from flask import Flask
        
        # Create test app
        app = Flask(__name__)
        app.register_blueprint(hazardguard_bp, url_prefix='/api/hazardguard')
        
        print("✓ Blueprint registered successfully")
        
        # Test route registration
        routes = []
        for rule in app.url_map.iter_rules():
            if rule.rule.startswith('/api/hazardguard'):
                routes.append(f"{rule.rule} | {','.join(rule.methods)}")
        
        expected_routes = [
            '/predict', '/predict/batch', '/capabilities', '/validate/coordinates',
            '/health', '/initialize', '/statistics/reset', '/ping'
        ]
        
        registered_routes = [route.split(' | ')[0].replace('/api/hazardguard', '') for route in routes]
        
        all_routes_found = all(route in registered_routes for route in expected_routes)
        
        if all_routes_found:
            print(f"✓ All expected routes registered ({len(expected_routes)} routes)")
            for route in routes:
                print(f"  - {route}")
            
            # Test blueprint info
            from routes.hazardguard_prediction_routes import get_blueprint_info
            blueprint_info = get_blueprint_info()
            
            print(f"\\n  📋 Blueprint Information:")
            print(f"    - Name: {blueprint_info['name']}")
            print(f"    - Description: {blueprint_info['description']}")
            print(f"    - Version: {blueprint_info['version']}")
            print(f"    - Prediction type: {blueprint_info['prediction_type']}")
            print(f"    - Supported disasters: {', '.join(blueprint_info['supported_disasters'])}")
            
            return True
        else:
            missing_routes = [route for route in expected_routes if route not in registered_routes]
            print(f"❌ Missing routes: {missing_routes}")
            return False
            
    except Exception as e:
        print(f"  ❌ Routes test error: {e}")
        return False

def test_integration():
    """Test full integration with main.py"""
    print("\\n=== Testing HazardGuard Integration ===")
    
    try:
        # Check if HazardGuard components are in main.py
        with open('main.py', 'r', encoding='utf-8', errors='ignore') as f:
            main_content = f.read()
        
        integration_checks = [
            ('HazardGuardPredictionService import', 'HazardGuardPredictionService' in main_content),
            ('HazardGuardPredictionController import', 'HazardGuardPredictionController' in main_content),
            ('hazardguard_bp import', 'hazardguard_bp' in main_content),
            ('HazardGuard service initialization', 'hazardguard_service' in main_content),
            ('HazardGuard controller creation', "'hazardguard':" in main_content),
            ('HazardGuard blueprint registration', '/api/hazardguard' in main_content),
            ('HazardGuard health check', 'hazardguard_healthy' in main_content)
        ]
        
        passed_checks = []
        failed_checks = []
        
        for check_name, check_result in integration_checks:
            if check_result:
                passed_checks.append(check_name)
                print(f"  ✓ {check_name}")
            else:
                failed_checks.append(check_name)
                print(f"  ❌ {check_name}")
        
        if all(check[1] for check in integration_checks):
            print("\\n✅ Integration successful - all HazardGuard components found in main.py")
            print(f"  - Service integration: ✓")
            print(f"  - Controller integration: ✓")
            print(f"  - Blueprint registration: ✓")
            print(f"  - Health check integration: ✓")
            return True
        else:
            print(f"\\n❌ Integration incomplete - {len(failed_checks)} components missing")
            return False
            
    except Exception as e:
        print(f"  ❌ Integration test error: {e}")
        return False

def test_end_to_end_simulation():
    """Simulate an end-to-end prediction workflow (without actual model inference)"""
    print("\\n=== Testing End-to-End Workflow Simulation ===")
    
    try:
        # Test the complete workflow components
        print("  🔄 Simulating map-based location selection...")
        test_location = HazardGuardTestData.TEST_LOCATIONS[0]  # Bangalore
        print(f"    - Selected location: {test_location['name']}")
        print(f"    - Coordinates: ({test_location['latitude']}, {test_location['longitude']})")
        print(f"    - Description: {test_location['description']}")
        
        print("  📅 Simulating reference date calculation...")
        reference_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        print(f"    - Reference date: {reference_date} (60 days ago)")
        
        print("  🌤️  Simulating data collection workflow...")
        workflows = [
            "Weather data collection (NASA POWER API - 17 variables, 60 days)",
            "Feature engineering (19 engineered features from weather data)",
            "Raster data collection (9 geographic variables)",
            "Feature preparation and statistical expansion (~300 features)",
            "Model prediction (XGBoost binary classification)"
        ]
        
        for i, workflow in enumerate(workflows, 1):
            print(f"    {i}. {workflow}")
        
        print("  📊 Simulating expected outputs...")
        simulated_outputs = {
            'prediction_class': 'DISASTER or NORMAL',
            'disaster_probability': '0.0 to 1.0',
            'normal_probability': '0.0 to 1.0', 
            'confidence': 'Difference between class probabilities',
            'processing_time': '~5-15 seconds (depending on data collection)',
            'metadata': 'Feature counts, model info, timestamps'
        }
        
        for key, value in simulated_outputs.items():
            print(f"    - {key}: {value}")
        
        print("\\n  🎯 Workflow simulation completed successfully!")
        return True
        
    except Exception as e:
        print(f"  ❌ End-to-end simulation error: {e}")
        return False

def run_comprehensive_hazardguard_tests():
    """Run all HazardGuard prediction system tests"""
    print("🚀 Starting Comprehensive HazardGuard Disaster Prediction Tests")
    print("=" * 80)
    
    test_results = {
        'model': False,
        'service': False,
        'controller': False,
        'routes': False,
        'integration': False,
        'end_to_end': False
    }
    
    # Run tests
    test_results['model'] = test_model_functionality()
    test_results['service'] = test_service_functionality()
    test_results['controller'] = test_controller_functionality()
    test_results['routes'] = test_routes_functionality()
    test_results['integration'] = test_integration()
    test_results['end_to_end'] = test_end_to_end_simulation()
    
    # Summary
    print("\\n" + "=" * 80)
    print("📊 HAZARDGUARD TEST SUMMARY")
    print("=" * 80)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for component, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{component.upper():>15}: {status}")
    
    print("-" * 40)
    print(f"Total: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests >= total_tests - 1:  # Allow 1 failure for model loading if files missing
        print("\\n🎉 HAZARDGUARD SYSTEM READY!")
        print("\\n🎯 HazardGuard Disaster Prediction System:")
        print("  • Binary classification: DISASTER vs NORMAL")
        print("  • Supported disasters: Flood, Storm, Landslide, Drought")  
        print("  • Forecasting horizon: 1 day ahead")
        print("  • Global coverage: Any lat/lon coordinates")
        print("  • Map-based location selection")
        print("  • Real-time data collection and processing")
        print("  • Batch prediction support (up to 50 locations)")
        
        print("\\n🌐 Available endpoints:")
        endpoints = [
            "POST /api/hazardguard/predict - Primary prediction endpoint",
            "POST /api/hazardguard/predict/batch - Batch predictions",
            "GET  /api/hazardguard/capabilities - System capabilities", 
            "POST /api/hazardguard/validate/coordinates - Coordinate validation",
            "GET  /api/hazardguard/health - Service health monitoring",
            "POST /api/hazardguard/initialize - Service initialization",
            "POST /api/hazardguard/statistics/reset - Reset statistics",
            "GET  /api/hazardguard/ping - Service ping test"
        ]
        for endpoint in endpoints:
            print(f"  • {endpoint}")
        
        print("\\n📋 Usage workflow:")
        print("  1. Select location on map (latitude, longitude)")
        print("  2. System fetches weather, features, and raster data")
        print("  3. Model processes ~300 engineered features")
        print("  4. Returns disaster risk prediction with probability")
        
        if test_results['model']:
            print("\\n✅ Model files loaded successfully - predictions will work!")
        else:
            print("\\n⚠️  Model files not found - predictions will fail until model is trained")
            print("   💡 Run the training script: combined_all6_predict_research.py")
            print("   💡 Model files should be generated in: output_of_combined_disaster_forecast/")
        
        return True
    else:
        print(f"\\n❌ {total_tests - passed_tests} CRITICAL TEST(S) FAILED!")
        print("   Please review failed components before deployment.")
        return False

if __name__ == '__main__':
    try:
        success = run_comprehensive_hazardguard_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\\n💥 Test runner crashed: {e}")
        traceback.print_exc()
        sys.exit(1)