"""
Quick standalone test: verify all 4 binary disaster-type classifiers
load correctly and produce predictions from synthetic feature data.
"""

import os, sys, joblib, numpy as np, pandas as pd
from pathlib import Path

MODEL_BASE = r"D:\INTERNSHIP FINAL YEAR PROJECT\models for research paper\HORIZON1"

MODELS = {
    "Storm":     os.path.join(MODEL_BASE, "output_of_NOstorm_vs_storm_forecast",           "binary_NOstorm_storm_pipeline.joblib"),
    "Flood":     os.path.join(MODEL_BASE, "output_of_NOflood_vs_flood_forecast",            "binary_NOflood_flood_pipeline.joblib"),
    "Drought":   os.path.join(MODEL_BASE, "output_of_NOdrought_vs_drought_forecast",        "binary_NOdrought_drought_pipeline.joblib"),
    "Landslide": os.path.join(MODEL_BASE, "output_of_NOmassmovement_vs_massmovement_forecast", "binary_NOmassmovement_massmovement_pipeline.joblib"),
}

# 36 array features × 8 stats = 288  +  9 scalar = 297 total
ARRAY_FEATURES = [
    'temperature_C','humidity_%','wind_speed_mps','precipitation_mm',
    'surface_pressure_hPa','solar_radiation_wm2','temperature_max_C','temperature_min_C',
    'specific_humidity_g_kg','dew_point_C','wind_speed_10m_mps','cloud_amount_%',
    'sea_level_pressure_hPa','surface_soil_wetness_%','wind_direction_10m_degrees',
    'evapotranspiration_wm2','root_zone_soil_moisture_%',
    'temp_normalized','temp_range','discomfort_index','heat_index',
    'wind_precip_interaction','solar_temp_ratio','pressure_anomaly',
    'high_precip_flag','adjusted_humidity','wind_chill',
    'solar_radiation_anomaly','weather_severity_score',
    'moisture_stress_index','evaporation_deficit','soil_saturation_index',
    'atmospheric_instability','drought_indicator','flood_risk_score','storm_intensity_index'
]
STATS = ['mean','min','max','std','median','q25','q75','skew']
SCALAR_FEATURES = [
    'soil_type','elevation_m','pop_density_persqkm','land_cover_class',
    'ndvi','annual_precip_mm','annual_mean_temp_c','mean_wind_speed_ms',
    'impervious_surface_pct'
]

def build_feature_columns():
    cols = []
    for feat in ARRAY_FEATURES:
        for stat in STATS:
            cols.append(f"{feat}_{stat}")
    cols.extend(SCALAR_FEATURES)
    return cols

def make_synthetic_row(seed=42):
    """Generate one row of 297 features with realistic-ish random values."""
    rng = np.random.RandomState(seed)
    cols = build_feature_columns()
    vals = rng.uniform(-1, 3, size=len(cols))  # broad range so selector/scaler get something
    return pd.DataFrame([vals], columns=cols)

def main():
    print("=" * 70)
    print("  DISASTER-TYPE CLASSIFIER VERIFICATION")
    print("=" * 70)

    features = make_synthetic_row()
    print(f"\nSynthetic feature vector: {features.shape[1]} columns  (expect 297)\n")

    all_ok = True
    for name, path in MODELS.items():
        tag = f"[{name:>10}]"

        # --- file exists? ---
        if not os.path.exists(path):
            print(f"{tag}  FAIL  pipeline file not found: {path}")
            all_ok = False
            continue

        # --- load ---
        try:
            pipe = joblib.load(path)
        except Exception as e:
            print(f"{tag}  FAIL  could not load pipeline: {e}")
            all_ok = False
            continue

        # --- inspect contents ---
        keys = list(pipe.keys()) if isinstance(pipe, dict) else []
        required = {'model', 'scaler', 'selector'}
        if not required.issubset(set(keys)):
            print(f"{tag}  FAIL  missing keys in pipeline (got {keys})")
            all_ok = False
            continue

        selector = pipe['selector']
        scaler   = pipe['scaler']
        model    = pipe['model']
        sel_feats = pipe.get('selected_features', [])
        target   = pipe.get('target_disaster', '?')
        neg_lab  = pipe.get('negative_label', '?')

        # --- run pipeline ---
        try:
            X_sel = selector.transform(features)
            X_sc  = scaler.transform(X_sel)
            pred  = model.predict(X_sc)[0]
            proba = model.predict_proba(X_sc)[0]
            prob_pos = proba[1] if len(proba) > 1 else proba[0]
        except Exception as e:
            print(f"{tag}  FAIL  prediction error: {e}")
            all_ok = False
            continue

        label = target if pred == 1 else neg_lab
        print(f"{tag}  OK")
        print(f"           pipeline keys  : {keys}")
        print(f"           selected feats  : {len(sel_feats)}")
        print(f"           target/neg      : {target} / {neg_lab}")
        print(f"           prediction      : {pred}  ({label})")
        print(f"           P(disaster)     : {prob_pos:.4f}")
        print(f"           full proba      : {[round(float(p),4) for p in proba]}")
        print()

    print("=" * 70)
    if all_ok:
        print("  RESULT:  ALL 4 CLASSIFIERS LOADED & PREDICTED SUCCESSFULLY")
    else:
        print("  RESULT:  SOME CLASSIFIERS FAILED — see above")
    print("=" * 70)

if __name__ == "__main__":
    main()
