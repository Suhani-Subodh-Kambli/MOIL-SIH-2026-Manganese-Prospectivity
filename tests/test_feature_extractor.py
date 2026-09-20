"""
tests/test_feature_extractor.py

Tests ProspectivityFeatureExtractor enrichment, preprocessor compatibility,
and end-to-end model inference.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.feature_extractor import ProspectivityFeatureExtractor


def test_feature_extractor():
    print("Testing ProspectivityFeatureExtractor...")

    extractor = ProspectivityFeatureExtractor()

    # Load Balaghat grid sample (which has raw satellite/terrain bands)
    grid_file = PROJECT_ROOT / "data" / "MOIL_Balaghat_Phase5_Final_Feature_Grid.csv"
    if not grid_file.exists():
        raise FileNotFoundError(f"Missing grid file: {grid_file}")

    sample_df = pd.read_csv(grid_file, nrows=5)

    # 1. Enrich
    enriched = extractor.enrich_dataframe(sample_df)
    print(f"Enriched columns: {len(enriched.columns)}")

    # 2. Get 51 raw model features
    X_raw = extractor.get_model_features_df(sample_df)
    print(f"Extracted raw model features shape: {X_raw.shape}")
    assert X_raw.shape[1] == 51, f"Expected 51 raw features, got {X_raw.shape[1]}"

    # 3. Transform with saved Phase 4B preprocessor
    preprocessor_file = PROJECT_ROOT / "models" / "phase4b_preprocessor.joblib"
    preprocessor = joblib.load(preprocessor_file)

    feature_cols_file = PROJECT_ROOT / "models" / "phase4b_feature_columns.json"
    with open(feature_cols_file, "r") as f:
        expected_feature_cols = json.load(f)

    X_transformed = preprocessor.transform(X_raw)
    print(f"Transformed features shape: {X_transformed.shape}")
    assert X_transformed.shape[1] == len(expected_feature_cols), (
        f"Transformed feature count {X_transformed.shape[1]} != {len(expected_feature_cols)}"
    )

    # 4. Predict with Phase 4B XGBoost model
    model_file = PROJECT_ROOT / "models" / "moil_manganese_prospectivity_xgb_phase4b.joblib"
    model = joblib.load(model_file)

    X_transformed_df = pd.DataFrame(X_transformed, columns=expected_feature_cols)
    probs = model.predict_proba(X_transformed_df)[:, 1]
    print(f"Predicted probabilities on sample: {probs.round(4)}")
    assert len(probs) == 5
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    print("\nAll ProspectivityFeatureExtractor tests passed successfully!")


if __name__ == "__main__":
    test_feature_extractor()
