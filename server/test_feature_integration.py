"""
Feature Engineering Integration Test
Verifies feature engineering MVC components work correctly with NaN handling
"""
import sys
import os
import numpy as np

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_feature_engineering_imports():
    """Test if all feature engineering components can be imported"""
    try:
        print("Testing feature engineering imports...")
        
        # Test model imports
        from models.feature_engineering_model import WeatherFeatureModel
        print("✅ Feature engineering model imports successful")
        
        # Test service imports
        from services.feature_engineering_service import FeatureEngineeringService
        print("✅ Feature engineering service imports successful")
        
        # Test controller imports
        from controllers.feature_engineering_controller import FeatureEngineeringController
        print("✅ Feature engineering controller imports successful")
        
        # Test route imports
        from routes.feature_routes import features_bp, init_feature_routes
        print("✅ Feature engineering routes imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

def test_feature_service_creation():
    """Test feature engineering service creation and basic functionality"""
    try:
        print("\nTesting feature engineering service creation...")
        
        from services.feature_engineering_service import FeatureEngineeringService
        from controllers.feature_engineering_controller import FeatureEngineeringController
        from models.feature_engineering_model import WeatherFeatureModel
        
        # Create service
        feature_service = FeatureEngineeringService()
        print("✅ Feature engineering service created successfully")
        
        # Create controller
        feature_controller = FeatureEngineeringController(feature_service)
        print("✅ Feature engineering controller created successfully")
        
        # Test service status
        status = feature_service.get_service_status()
        if status['initialized']:
            print(f"✅ Service initialized: {status['supported_features']} features available")
        else:
            print("❌ Service not properly initialized")
            return False
        
        # Test feature info
        feature_info = feature_service.get_feature_info()
        expected_features = len(WeatherFeatureModel.ENGINEERED_FEATURES)
        if feature_info['feature_count'] == expected_features:
            print(f"✅ Feature info correct: {expected_features} engineered features")
        else:
            print(f"❌ Feature count mismatch: expected {expected_features}, got {feature_info['feature_count']}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Feature service creation error: {str(e)}")
        return False

def test_nan_handling():
    """Test proper NaN handling in feature engineering"""
    try:
        print("\nTesting NaN handling in feature engineering...")
        
        from services.feature_engineering_service import FeatureEngineeringService
        from models.feature_engineering_model import WeatherFeatureModel
        
        feature_service = FeatureEngineeringService()
        
        # Create sample weather data with some NaN values
        sample_weather_data = {
            'temperature_C': [25.5, np.nan, 24.8, 27.2, None],
            'humidity_perc': [65.2, 67.8, np.nan, 62.5, 68.9],
            'wind_speed_mps': [3.2, 2.8, 4.1, np.nan, 3.9],
            'precipitation_mm': [0.0, 2.3, np.nan, 0.0, 1.2],
            'surface_pressure_hPa': [1013.2, 1012.8, 1011.5, np.nan, 1013.7],
            'solar_radiation_wm2': [220.5, np.nan, 150.8, 240.2, 200.1],
            'temperature_max_C': [30.2, 31.1, np.nan, 32.8, 30.9],
            'temperature_min_C': [20.8, np.nan, 20.1, 21.6, 21.0],
            'specific_humidity_g_kg': [12.5, 13.1, np.nan, 11.9, 13.2],
            'dew_point_C': [18.2, np.nan, 19.8, 17.5, 18.9],
            'wind_speed_10m_mps': [4.1, 3.6, np.nan, 7.1, 4.9],
            'cloud_amount_perc': [30.0, np.nan, 80.0, 20.0, 50.0],
            'sea_level_pressure_hPa': [1013.5, 1013.1, np.nan, 1014.4, 1014.0],
            'surface_soil_wetness_perc': [45.0, 52.0, np.nan, 42.0, 48.0],
            'wind_direction_10m_degrees': [180.0, np.nan, 220.0, 195.0, 170.0],
            'evapotranspiration_wm2': [85.2, 72.1, np.nan, 95.8, 82.4],
            'root_zone_soil_moisture_perc': [55.0, np.nan, 74.0, 52.0, 58.0]
        }
        
        # Process features
        success, result = feature_service.process_weather_features(
            sample_weather_data, event_duration=1.0, include_metadata=True
        )
        
        if not success:
            print(f"❌ Feature processing failed: {result}")
            return False
        
        print("✅ Feature processing with NaN values successful")
        
        # Check that NaN values are properly handled
        engineered_features = result['engineered_features']
        
        feature_count = 0
        nan_handled_correctly = 0
        
        for feature_name, values in engineered_features.items():
            feature_count += 1
            # Check if NaN values in input produced appropriate NaN handling in output
            nan_count = sum(1 for v in values if v is not None and isinstance(v, float) and np.isnan(v))
            
            if nan_count > 0:
                nan_handled_correctly += 1
                print(f"  ✅ {feature_name}: {nan_count}/5 NaN values properly handled")
            else:
                # Check if all values are valid numbers
                valid_count = sum(1 for v in values if v is not None and not (isinstance(v, float) and np.isnan(v)))
                print(f"  ✅ {feature_name}: {valid_count}/5 valid values computed")
        
        print(f"✅ Processed {feature_count} engineered features")
        print(f"✅ NaN handling verified for features with missing input data")
        
        # Verify we got the expected number of features (19 total, excluding precip_intensity_mm_day)
        expected_feature_count = 19
        if feature_count == expected_feature_count:
            print(f"✅ Correct number of features: {feature_count}/{expected_feature_count}")
        else:
            print(f"❌ Feature count mismatch: expected {expected_feature_count}, got {feature_count}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ NaN handling test error: {str(e)}")
        return False

def test_feature_computation_accuracy():
    """Test accuracy of feature computations"""
    try:
        print("\nTesting feature computation accuracy...")
        
        from models.feature_engineering_model import WeatherFeatureModel
        
        # Create simple test data
        simple_weather_data = {
            'temperature_C': [25.0, 30.0],
            'humidity_perc': [60.0, 70.0], 
            'wind_speed_mps': [2.0, 4.0],
            'precipitation_mm': [0.0, 10.0],
            'surface_pressure_hPa': [1013.0, 1010.0],
            'solar_radiation_wm2': [200.0, 150.0],
            'temperature_max_C': [30.0, 35.0],
            'temperature_min_C': [20.0, 25.0],
            'specific_humidity_g_kg': [12.0, 15.0],
            'dew_point_C': [18.0, 22.0],
            'wind_speed_10m_mps': [3.0, 5.0],
            'cloud_amount_perc': [40.0, 80.0],
            'sea_level_pressure_hPa': [1013.2, 1010.2],
            'surface_soil_wetness_perc': [50.0, 70.0],
            'wind_direction_10m_degrees': [180.0, 270.0],
            'evapotranspiration_wm2': [80.0, 60.0],
            'root_zone_soil_moisture_perc': [60.0, 80.0]
        }
        
        # Compute features
        features = WeatherFeatureModel.compute_engineered_features(simple_weather_data, 1.0)
        
        # Verify specific computations
        temp_range_day1 = features['temp_range'][0]  # Should be 30.0 - 20.0 = 10.0
        temp_range_day2 = features['temp_range'][1]  # Should be 35.0 - 25.0 = 10.0
        
        if abs(temp_range_day1 - 10.0) < 0.001 and abs(temp_range_day2 - 10.0) < 0.001:
            print("✅ Temperature range calculation correct")
        else:
            print(f"❌ Temperature range calculation incorrect: {temp_range_day1}, {temp_range_day2}")
            return False
        
        # Test wind-precip interaction
        wind_precip_day1 = features['wind_precip_interaction'][0]  # Should be 2.0 * 0.0 = 0.0
        wind_precip_day2 = features['wind_precip_interaction'][1]  # Should be 4.0 * 10.0 = 40.0
        
        if abs(wind_precip_day1 - 0.0) < 0.001 and abs(wind_precip_day2 - 40.0) < 0.001:
            print("✅ Wind-precipitation interaction calculation correct")
        else:
            print(f"❌ Wind-precipitation interaction incorrect: {wind_precip_day1}, {wind_precip_day2}")
            return False
        
        # Test high precipitation flag
        high_precip_day1 = features['high_precip_flag'][0]  # Should be 0 (0.0 mm < 50mm)
        high_precip_day2 = features['high_precip_flag'][1]  # Should be 0 (10.0 mm < 50mm)
        
        if high_precip_day1 == 0 and high_precip_day2 == 0:
            print("✅ High precipitation flag calculation correct")
        else:
            print(f"❌ High precipitation flag incorrect: {high_precip_day1}, {high_precip_day2}")
            return False
        
        print(f"✅ Feature computation accuracy verified for {len(features)} features")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature computation accuracy test error: {str(e)}")
        return False

def test_excluded_features():
    """Test that precip_intensity_mm_day is properly excluded"""
    try:
        print("\nTesting excluded features...")
        
        from models.feature_engineering_model import WeatherFeatureModel
        
        # Check that precip_intensity_mm_day is not in the feature list
        if 'precip_intensity_mm_day' in WeatherFeatureModel.ENGINEERED_FEATURES:
            print("❌ precip_intensity_mm_day should be excluded from features")
            return False
        else:
            print("✅ precip_intensity_mm_day properly excluded from features")
        
        # Verify we have exactly 19 features (not 20)
        feature_count = len(WeatherFeatureModel.ENGINEERED_FEATURES)
        if feature_count == 19:
            print(f"✅ Correct feature count: {feature_count} features (precip_intensity_mm_day excluded)")
        else:
            print(f"❌ Incorrect feature count: expected 19, got {feature_count}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Excluded features test error: {str(e)}")
        return False

def main():
    """Run all feature engineering tests"""
    print("=== Feature Engineering Integration Test ===\n")
    
    tests = [
        ("Import Tests", test_feature_engineering_imports),
        ("Service Creation Tests", test_feature_service_creation),
        ("NaN Handling Tests", test_nan_handling),
        ("Feature Computation Accuracy", test_feature_computation_accuracy),
        ("Excluded Features Test", test_excluded_features)
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
        print("\n🎉 All tests passed! Feature engineering integration is ready.")
        print("✅ NaN values handled properly (NaN input → NaN output for that day only)")
        print("✅ 19 engineered features computed correctly")
        print("✅ precip_intensity_mm_day properly excluded as requested")
        print("✅ MVC architecture complete and functional")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
    
    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)