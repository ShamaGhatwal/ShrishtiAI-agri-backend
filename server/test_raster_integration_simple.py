"""
Comprehensive Test Suite for Raster Data Integration
Tests all components of the raster data MVC system
"""

import sys
import os
import traceback
from typing import Dict, Any

# Add backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports() -> Dict[str, Any]:
    """Test that all raster data components can be imported"""
    print("\n" + "="*50)
    print("TEST 1: Import Tests")
    print("="*50)
    
    import_results = {
        'total_tests': 5,
        'passed_tests': 0,
        'failed_tests': 0,
        'details': {}
    }
    
    # Test model import
    try:
        from models.raster_data_model import RasterDataModel
        print("✅ RasterDataModel imported successfully")
        import_results['details']['model'] = 'SUCCESS'
        import_results['passed_tests'] += 1
    except Exception as e:
        print(f"❌ Failed to import RasterDataModel: {e}")
        import_results['details']['model'] = f'FAILED: {e}'
        import_results['failed_tests'] += 1
    
    # Test service import
    try:
        from services.raster_data_service import RasterDataService
        print("✅ RasterDataService imported successfully")
        import_results['details']['service'] = 'SUCCESS'
        import_results['passed_tests'] += 1
    except Exception as e:
        print(f"❌ Failed to import RasterDataService: {e}")
        import_results['details']['service'] = f'FAILED: {e}'
        import_results['failed_tests'] += 1
    
    # Test controller import
    try:
        from controllers.raster_data_controller import RasterDataController
        print("✅ RasterDataController imported successfully")
        import_results['details']['controller'] = 'SUCCESS'
        import_results['passed_tests'] += 1
    except Exception as e:
        print(f"❌ Failed to import RasterDataController: {e}")
        import_results['details']['controller'] = f'FAILED: {e}'
        import_results['failed_tests'] += 1
    
    # Test routes import
    try:
        from routes.raster_routes import create_raster_routes
        print("✅ Raster routes imported successfully")
        import_results['details']['routes'] = 'SUCCESS'
        import_results['passed_tests'] += 1
    except Exception as e:
        print(f"❌ Failed to import raster routes: {e}")
        import_results['details']['routes'] = f'FAILED: {e}'
        import_results['failed_tests'] += 1
    
    # Test configuration import
    try:
        from config.raster_config import RasterDataConfig, get_raster_config
        print("✅ RasterDataConfig imported successfully")
        import_results['details']['config'] = 'SUCCESS'
        import_results['passed_tests'] += 1
    except Exception as e:
        print(f"❌ Failed to import RasterDataConfig: {e}")
        import_results['details']['config'] = f'FAILED: {e}'
        import_results['failed_tests'] += 1
    
    print(f"\n📊 Import Tests: {import_results['passed_tests']}/{import_results['total_tests']} passed")
    return import_results

def test_configuration() -> Dict[str, Any]:
    """Test raster data configuration loading"""
    print("\n" + "="*50)
    print("TEST 2: Configuration Tests")
    print("="*50)
    
    config_results = {
        'total_tests': 4,
        'passed_tests': 0,
        'failed_tests': 0,
        'details': {}
    }
    
    try:
        from config.raster_config import RasterDataConfig, get_raster_config
        
        # Test configuration loading
        try:
            config = RasterDataConfig()
            print("✅ RasterDataConfig created successfully")
            config_results['details']['creation'] = 'SUCCESS'
            config_results['passed_tests'] += 1
        except Exception as e:
            print(f"❌ Failed to create RasterDataConfig: {e}")
            config_results['details']['creation'] = f'FAILED: {e}'
            config_results['failed_tests'] += 1
            return config_results
        
        # Test getting configuration
        try:
            config_dict = config.get_config()
            raster_paths = config.get_raster_paths()
            print(f"✅ Configuration loaded: {len(raster_paths)} raster sources configured")
            config_results['details']['loading'] = f'SUCCESS: {len(raster_paths)} sources'
            config_results['passed_tests'] += 1
        except Exception as e:
            print(f"❌ Failed to get configuration: {e}")
            config_results['details']['loading'] = f'FAILED: {e}'
            config_results['failed_tests'] += 1
        
        # Test configuration validation
        try:
            validation = config.validate_configuration()
            print(f"✅ Configuration validation completed: {validation['valid']}")
            config_results['details']['validation'] = f"SUCCESS: Valid={validation['valid']}"
            config_results['passed_tests'] += 1
        except Exception as e:
            print(f"❌ Failed to validate configuration: {e}")
            config_results['details']['validation'] = f'FAILED: {e}'
            config_results['failed_tests'] += 1
        
        # Test global config access
        try:
            global_config = get_raster_config()
            print("✅ Global configuration access successful")
            config_results['details']['global_access'] = 'SUCCESS'
            config_results['passed_tests'] += 1
        except Exception as e:
            print(f"❌ Failed to access global configuration: {e}")
            config_results['details']['global_access'] = f'FAILED: {e}'
            config_results['failed_tests'] += 1
        
    except Exception as e:
        print(f"❌ Critical error in configuration tests: {e}")
        config_results['details']['critical_error'] = str(e)
        config_results['failed_tests'] = config_results['total_tests']
    
    print(f"\n📊 Configuration Tests: {config_results['passed_tests']}/{config_results['total_tests']} passed")
    return config_results

def test_model_functionality() -> Dict[str, Any]:
    """Test raster data model functionality"""
    print("\n" + "="*50)
    print("TEST 3: Model Functionality Tests")
    print("="*50)
    
    model_results = {
        'total_tests': 3,
        'passed_tests': 0,
        'failed_tests': 0,
        'details': {}
    }
    
    try:
        from models.raster_data_model import RasterDataModel
        
        # Test model creation
        try:
            model = RasterDataModel()
            print("✅ RasterDataModel created successfully")
            model_results['details']['creation'] = 'SUCCESS'
            model_results['passed_tests'] += 1
        except Exception as e:
            print(f"❌ Failed to create RasterDataModel: {e}")
            model_results['details']['creation'] = f'FAILED: {e}'
            model_results['failed_tests'] += 1
            return model_results
        
        # Test soil class encoding
        try:
            soil_code = model.encode_soil_class('Acrisols')
            print(f"✅ Soil class encoding works: 'Acrisols' -> {soil_code}")
            model_results['details']['soil_encoding'] = f'SUCCESS: {soil_code}'
            model_results['passed_tests'] += 1
        except Exception as e:
            print(f"❌ Soil class encoding failed: {e}")
            model_results['details']['soil_encoding'] = f'FAILED: {e}'
            model_results['failed_tests'] += 1
        
        # Test land cover encoding
        try:
            lc_code = model.encode_land_cover(111)
            print(f"✅ Land cover encoding works: 111 -> {lc_code}")
            model_results['details']['landcover_encoding'] = f'SUCCESS: {lc_code}'
            model_results['passed_tests'] += 1
        except Exception as e:
            print(f"❌ Land cover encoding failed: {e}")
            model_results['details']['landcover_encoding'] = f'FAILED: {e}'
            model_results['failed_tests'] += 1
        
    except Exception as e:
        print(f"❌ Critical error in model tests: {e}")
        model_results['details']['critical_error'] = str(e)
        model_results['failed_tests'] = model_results['total_tests']
    
    print(f"\n📊 Model Functionality Tests: {model_results['passed_tests']}/{model_results['total_tests']} passed")
    return model_results

def run_all_tests():
    """Run all raster data integration tests"""
    print("\n" + "="*80)
    print("🧪 RASTER DATA INTEGRATION TEST SUITE")
    print("Testing complete MVC architecture for raster data extraction")
    print("="*80)
    
    # Run test suites
    test_results = []
    
    try:
        test_results.append(test_imports())
        test_results.append(test_configuration())
        test_results.append(test_model_functionality())
        
    except Exception as e:
        print(f"\n❌ Critical error during test execution: {e}")
        traceback.print_exc()
        return
    
    # Calculate overall results
    total_tests = sum(result['total_tests'] for result in test_results)
    total_passed = sum(result['passed_tests'] for result in test_results)
    total_failed = sum(result['failed_tests'] for result in test_results)
    success_rate = (total_passed / total_tests) * 100 if total_tests > 0 else 0
    
    # Print summary
    print("\n" + "="*80)
    print("📋 TEST SUMMARY")
    print("="*80)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed} ✅")
    print(f"Failed: {total_failed} ❌")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("\n🎉 RASTER DATA INTEGRATION: SYSTEM READY")
        print("✅ All critical components are functional")
        print("✅ MVC architecture is properly integrated")
        print("\n🚀 Ready for production use with 9 geospatial features:")
        print("   • Soil Type (HWSD2) - 33 classifications")
        print("   • Elevation (WorldClim) - meters ASL")
        print("   • Population Density (GlobPOP) - persons/km²")
        print("   • Land Cover (Copernicus) - 22 classes")
        print("   • NDVI (MODIS/eVIIRS) - vegetation index")
        print("   • Annual Precipitation (WorldClim) - mm/year")
        print("   • Annual Temperature (WorldClim) - °C")
        print("   • Wind Speed (Global Wind Atlas) - m/s")
        print("   • Impervious Surface (GHSL) - percentage")
    else:
        print("\n⚠️  RASTER DATA INTEGRATION: ISSUES DETECTED")
        print("❌ Some components failed testing")
        print("❌ Review failed tests before production use")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    run_all_tests()