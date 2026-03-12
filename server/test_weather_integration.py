"""
Quick test script to verify weather service integration
"""
import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_weather_imports():
    """Test if all weather service components can be imported"""
    try:
        print("Testing weather service imports...")
        
        # Test model imports
        from models.weather_model import WeatherRequest, WeatherDataModel
        print("✅ Weather model imports successful")
        
        # Test service imports
        from services.weather_service import NASAPowerService
        print("✅ Weather service imports successful")
        
        # Test controller imports
        from controllers.weather_controller import WeatherController
        print("✅ Weather controller imports successful")
        
        # Test route imports
        from routes.weather_routes import weather_bp, init_weather_routes
        print("✅ Weather routes imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

def test_weather_service_creation():
    """Test weather service creation and basic functionality"""
    try:
        print("\nTesting weather service creation...")
        
        from services.weather_service import NASAPowerService
        from controllers.weather_controller import WeatherController
        from models.weather_model import WeatherRequest
        
        # Create service
        weather_service = NASAPowerService()
        print("✅ Weather service created successfully")
        
        # Create controller
        weather_controller = WeatherController(weather_service)
        print("✅ Weather controller created successfully")
        
        # Test weather request creation
        test_request = WeatherRequest(
            latitude=19.076,
            longitude=72.8777,
            disaster_date="2024-01-15",
            days_before=7
        )
        print("✅ Weather request created successfully")
        
        # Test validation
        validation = test_request.validate()
        if validation['valid']:
            print("✅ Weather request validation successful")
        else:
            print(f"❌ Weather request validation failed: {validation['errors']}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Weather service creation error: {str(e)}")
        return False

def test_weather_data_model():
    """Test weather data model functionality"""
    try:
        print("\nTesting weather data model...")
        
        from models.weather_model import WeatherDataModel
        
        # Create sample weather data (simulating NASA API response format)
        sample_raw_data = {
            'T2M': {'20240109': 25.5, '20240110': 26.0, '20240111': 24.8, 
                    '20240112': 27.2, '20240113': -999, '20240114': 25.1, '20240115': 26.3},
            'RH2M': {'20240109': 65.2, '20240110': 67.8, '20240111': 70.1, 
                     '20240112': 62.5, '20240113': 68.9, '20240114': -999, '20240115': 66.4},
            'PRECTOT': {'20240109': 0.0, '20240110': 2.3, '20240111': 5.1, 
                        '20240112': 0.0, '20240113': 1.2, '20240114': 0.8, '20240115': 3.4}
        }
        
        # Test raw data processing
        processed_data = WeatherDataModel.process_raw_data(sample_raw_data, 7)
        print("✅ Weather raw data processing successful")
        
        # Verify processed data structure
        expected_fields = ['temperature_C', 'humidity_perc', 'precipitation_mm']
        for field in expected_fields:
            if field in processed_data and len(processed_data[field]) == 7:
                print(f"✅ Field {field}: {len(processed_data[field])} values")
            else:
                print(f"❌ Field {field}: missing or wrong length")
        
        # Test time series creation
        try:
            import pandas as pd
            df = WeatherDataModel.create_time_series_dataframe(
                processed_data, "2024-01-15", 7
            )
            print("✅ Time series DataFrame creation successful")
            print(f"    DataFrame shape: {df.shape}")
            
            # Verify date range
            start_date = df['date'].iloc[0]
            end_date = df['date'].iloc[-1]
            print(f"    Date range: {start_date} to {end_date}")
            
        except ImportError:
            print("⚠️  Pandas not available - time series test skipped")
        
        return True
        
    except Exception as e:
        print(f"❌ Weather data model error: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("=== Weather Service Integration Test ===\n")
    
    tests = [
        ("Import Tests", test_weather_imports),
        ("Service Creation Tests", test_weather_service_creation),
        ("Data Model Tests", test_weather_data_model)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        success = test_func()
        results.append((test_name, success))
    
    print("\n=== Test Results Summary ===")
    all_passed = True
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Weather service integration is ready.")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
    
    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)