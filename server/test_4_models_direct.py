"""
Direct verification of the 4 disaster type classifier models (NO_X vs X).
Bypasses the full API pipeline — loads each model directly and runs predictions
with synthetic feature data to confirm they load, transform, and predict correctly.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_BASE_DIR = r"D:\INTERNSHIP FINAL YEAR PROJECT\models for research paper\HORIZON1"

MODEL_CONFIGS = {
    'Storm': os.path.join(MODEL_BASE_DIR, 'output_of_NOstorm_vs_storm_forecast', 'binary_NOstorm_storm_pipeline.joblib'),
    'Flood': os.path.join(MODEL_BASE_DIR, 'output_of_NOflood_vs_flood_forecast', 'binary_NOflood_flood_pipeline.joblib'),
    'Drought': os.path.join(MODEL_BASE_DIR, 'output_of_NOdrought_vs_drought_forecast', 'binary_NOdrought_drought_pipeline.joblib'),
    'Landslide': os.path.join(MODEL_BASE_DIR, 'output_of_NOmassmovement_vs_massmovement_forecast', 'binary_NOmassmovement_massmovement_pipeline.joblib'),
}

# 36 array features × 8 stats = 288
ARRAY_FEATURE_COLUMNS = [
    'temperature_C', 'humidity_%', 'wind_speed_mps', 'precipitation_mm',
    'surface_pressure_hPa', 'solar_radiation_wm2', 'temperature_max_C', 'temperature_min_C',
    'specific_humidity_g_kg', 'dew_point_C', 'wind_speed_10m_mps', 'cloud_amount_%',
    'sea_level_pressure_hPa', 'surface_soil_wetness_%', 'wind_direction_10m_degrees',
    'evapotranspiration_wm2', 'root_zone_soil_moisture_%',
    'temp_normalized', 'temp_range', 'discomfort_index', 'heat_index',
    'wind_precip_interaction', 'solar_temp_ratio', 'pressure_anomaly',
    'high_precip_flag', 'adjusted_humidity', 'wind_chill',
    'solar_radiation_anomaly', 'weather_severity_score',
    'moisture_stress_index', 'evaporation_deficit', 'soil_saturation_index',
    'atmospheric_instability', 'drought_indicator', 'flood_risk_score', 'storm_intensity_index'
]
STATS = ['mean', 'min', 'max', 'std', 'median', 'q25', 'q75', 'skew']

SCALAR_FEATURE_COLUMNS = [
    'soil_type', 'elevation_m', 'pop_density_persqkm', 'land_cover_class',
    'ndvi', 'annual_precip_mm', 'annual_mean_temp_c', 'mean_wind_speed_ms',
    'impervious_surface_pct'
]


def build_synthetic_features(seed: int = 42) -> pd.DataFrame:
    """Build a 297-feature DataFrame with realistic-ish random values."""
    rng = np.random.RandomState(seed)
    row = {}
    for col in ARRAY_FEATURE_COLUMNS:
        for stat in STATS:
            row[f"{col}_{stat}"] = rng.uniform(-2, 2)
    for col in SCALAR_FEATURE_COLUMNS:
        row[col] = rng.uniform(0, 100)
    return pd.DataFrame([row])


def test_single_model(name: str, path: str, features: pd.DataFrame) -> dict:
    """Load a pipeline, run the feature transform + prediction, and return results."""
    result = {'name': name, 'loaded': False, 'predicted': False, 'error': None}

    # 1. Load
    if not os.path.exists(path):
        result['error'] = f"File not found: {path}"
        return result

    data = joblib.load(path)
    if not isinstance(data, dict) or 'model' not in data:
        result['error'] = f"Unexpected pipeline format: keys={list(data.keys()) if isinstance(data, dict) else type(data)}"
        return result

    result['loaded'] = True
    result['pipeline_keys'] = list(data.keys())
    result['target_disaster'] = data.get('target_disaster', '?')
    result['negative_label'] = data.get('negative_label', '?')

    selector = data['selector']
    scaler   = data['scaler']
    model    = data['model']

    # Check selected_features if present
    selected_features = data.get('selected_features', [])
    result['n_selected_features'] = len(selected_features)

    # 2. Transform & predict
    try:
        X_sel = selector.transform(features)
        X_sc  = scaler.transform(X_sel)
        pred  = model.predict(X_sc)[0]
        proba = model.predict_proba(X_sc)[0]

        result['predicted'] = True
        result['prediction_raw'] = int(pred)
        result['probabilities'] = {f"class_{i}": round(float(p), 4) for i, p in enumerate(proba)}
        result['disaster_prob'] = round(float(proba[1]) if len(proba) > 1 else float(proba[0]), 4)
        result['input_features'] = features.shape[1]
        result['features_after_select'] = X_sel.shape[1]
        result['features_after_scale'] = X_sc.shape[1]
    except Exception as e:
        result['error'] = str(e)

    return result


def main():
    print("=" * 72)
    print("DISASTER TYPE CLASSIFIER — DIRECT MODEL VERIFICATION")
    print("=" * 72)

    features = build_synthetic_features()
    print(f"\nSynthetic feature vector: {features.shape[1]} columns  (expected 297)\n")

    all_ok = True
    for name, path in MODEL_CONFIGS.items():
        print(f"--- {name} ---")
        r = test_single_model(name, path, features.copy())

        if not r['loaded']:
            print(f"  LOAD FAILED: {r['error']}")
            all_ok = False
            continue

        print(f"  Pipeline keys : {r['pipeline_keys']}")
        print(f"  Target class  : {r['target_disaster']}  |  Negative class: {r['negative_label']}")
        print(f"  Selected feats: {r['n_selected_features']}")

        if not r['predicted']:
            print(f"  PREDICT FAILED: {r['error']}")
            all_ok = False
            continue

        print(f"  Input → select → scale: {r['input_features']} → {r['features_after_select']} → {r['features_after_scale']}")
        print(f"  Raw prediction : {r['prediction_raw']}")
        print(f"  Class probas   : {r['probabilities']}")
        print(f"  Disaster prob  : {r['disaster_prob']}")
        print(f"  STATUS: OK ✓")
        print()

    print("=" * 72)
    if all_ok:
        print("ALL 4 MODELS LOADED AND PREDICTED SUCCESSFULLY ✓")
    else:
        print("SOME MODELS FAILED — see details above")
    print("=" * 72)


if __name__ == "__main__":
    main()
