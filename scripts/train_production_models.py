"""
scripts/train_production_models.py

Trains and evaluates machine learning models for MOIL Production Shortfall Prediction:
1. Shortfall Percentage Regressor (RandomForestRegressor)
2. Shortfall Risk Tier Classifier (RandomForestClassifier)

Inputs:
- data/production/moil_mine_operations_history.csv

Outputs:
- models/moil_production_shortfall_regressor.joblib
- models/moil_production_shortfall_classifier.joblib
- models/production_feature_schema.json
- outputs/production_shortfall_metrics.csv
"""

import sys
from pathlib import Path
from typing import Tuple

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
    classification_report
)

warnings.filterwarnings("ignore")

SEED = 42
DATA_FILE = BASE_DIR / "data" / "production" / "moil_mine_operations_history.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

REGRESSOR_FILE = MODEL_DIR / "moil_production_shortfall_regressor.joblib"
CLASSIFIER_FILE = MODEL_DIR / "moil_production_shortfall_classifier.joblib"
SCHEMA_FILE = MODEL_DIR / "production_feature_schema.json"
METRICS_FILE = OUTPUT_DIR / "production_shortfall_metrics.csv"

FEATURE_COLUMNS = [
    "monthly_target_tonnes",
    "rainfall_mm",
    "fleet_availability_pct",
    "fleet_utilization_pct",
    "unplanned_downtime_hrs",
    "stripping_ratio_deficit",
    "shaft_utilization_pct",
    "power_outage_hrs",
    "is_opencast",
    "is_underground",
    "is_mixed"
]


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = df.copy()
    df["is_opencast"] = (df["mine_type"] == "Opencast").astype(int)
    df["is_underground"] = (df["mine_type"] == "Underground").astype(int)
    df["is_mixed"] = (df["mine_type"] == "Mixed").astype(int)

    X = df[FEATURE_COLUMNS].copy()
    y_reg = df["shortfall_pct"].to_numpy(dtype=float)
    y_clf = df["risk_tier"].to_numpy()
    return X, y_reg, y_clf


def main():
    print("=" * 75)
    print("TRAINING MOIL PRODUCTION SHORTFALL MODELS")
    print("=" * 75)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} records from {DATA_FILE}")

    X, y_reg, y_clf = prepare_features(df)

    # 1. 5-Fold Cross Validation for Regressor
    print("\n--- 5-Fold Cross-Validation: Shortfall Regressor ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    r2_scores = []
    mae_scores = []
    rmse_scores = []

    for tr_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y_reg[tr_idx], y_reg[val_idx]

        rf_reg = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=SEED)
        rf_reg.fit(X_tr, y_tr)
        preds = rf_reg.predict(X_val)

        r2_scores.append(r2_score(y_val, preds))
        mae_scores.append(mean_absolute_error(y_val, preds))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))

    print(f"  Regressor R² Score : {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f"  Regressor MAE (%)  : {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
    print(f"  Regressor RMSE (%) : {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")

    # 2. 5-Fold Cross Validation for Classifier
    print("\n--- 5-Fold Cross-Validation: Risk Tier Classifier ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    acc_scores = []
    f1_scores = []

    for tr_idx, val_idx in skf.split(X, y_clf):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y_clf[tr_idx], y_clf[val_idx]

        rf_clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=SEED)
        rf_clf.fit(X_tr, y_tr)
        preds = rf_clf.predict(X_val)

        acc_scores.append(accuracy_score(y_val, preds))
        f1_scores.append(f1_score(y_val, preds, average="weighted"))

    print(f"  Classifier Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
    print(f"  Classifier F1 (wt) : {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

    # 3. Train Final Models on Full Data
    print("\nTraining final production models on full historical dataset...")
    final_reg = RandomForestRegressor(n_estimators=120, max_depth=6, random_state=SEED)
    final_reg.fit(X, y_reg)
    joblib.dump(final_reg, REGRESSOR_FILE)
    print(f"Saved Shortfall Regressor: {REGRESSOR_FILE}")

    final_clf = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=SEED)
    final_clf.fit(X, y_clf)
    joblib.dump(final_clf, CLASSIFIER_FILE)
    print(f"Saved Risk Classifier: {CLASSIFIER_FILE}")

    with open(SCHEMA_FILE, "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)
    print(f"Saved Feature Schema: {SCHEMA_FILE}")

    # Feature Importances
    fi = sorted(zip(FEATURE_COLUMNS, final_reg.feature_importances_), key=lambda x: x[1], reverse=True)
    print("\nTop Operational Drivers of Mining Shortfalls:")
    for f_name, f_val in fi[:6]:
        print(f"  {f_name:25s}: {f_val:.4f}")

    # Save metrics table
    metrics_df = pd.DataFrame([
        {"model": "Shortfall_Regressor", "metric": "R2", "mean": np.mean(r2_scores), "std": np.std(r2_scores)},
        {"model": "Shortfall_Regressor", "metric": "MAE_pct", "mean": np.mean(mae_scores), "std": np.std(mae_scores)},
        {"model": "Shortfall_Regressor", "metric": "RMSE_pct", "mean": np.mean(rmse_scores), "std": np.std(rmse_scores)},
        {"model": "Risk_Classifier", "metric": "Accuracy", "mean": np.mean(acc_scores), "std": np.std(acc_scores)},
        {"model": "Risk_Classifier", "metric": "F1_weighted", "mean": np.mean(f1_scores), "std": np.std(f1_scores)}
    ])
    metrics_df.to_csv(METRICS_FILE, index=False)
    print(f"\nSaved metrics summary to: {METRICS_FILE}")


if __name__ == "__main__":
    main()
