"""
scripts/train_phase4b_nb.py

Trains and evaluates a Naive Bayes (GaussianNB) Manganese Prospectivity Model
using the exact same 5-fold StratifiedGroupKFold spatial cross-validation
and 117-feature preprocessing pipeline as Phase 4B XGBoost.

Outputs:
- models/moil_manganese_prospectivity_nb_phase4b.joblib
- outputs/phase4b_nb_spatial_metrics.csv
- outputs/phase4b_model_comparison.csv
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

SEED = 42

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "MOIL_SIH_Phase4B_Balanced_Features.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_FILE = MODEL_DIR / "moil_manganese_prospectivity_nb_phase4b.joblib"
XGB_MODEL_FILE = MODEL_DIR / "moil_manganese_prospectivity_xgb_phase4b.joblib"
PREPROCESSOR_FILE = MODEL_DIR / "phase4b_preprocessor.joblib"
FEATURE_COLUMNS_FILE = MODEL_DIR / "phase4b_feature_columns.json"

NB_METRICS_FILE = OUTPUT_DIR / "phase4b_nb_spatial_metrics.csv"
COMPARISON_FILE = OUTPUT_DIR / "phase4b_model_comparison.csv"
XGB_METRICS_FILE = OUTPUT_DIR / "phase4b_spatial_metrics.csv"


DROP_COLUMNS = [
    "label",
    "Distance_Mn",
    "latitude",
    "longitude",
    "system:index",
    ".geo",
]

SPATIAL_HELPER_COLUMNS = [
    "block",
    "block_lat",
    "block_lon",
    "nearest_positive_km",
]


def precision_at_fraction(y_true, y_prob, fraction):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    k = max(1, int(np.ceil(len(y_true) * fraction)))
    order = np.argsort(-y_prob)
    top_k = order[:k]
    return float(np.mean(y_true[top_k]))


def main():
    print("=" * 75)
    print("PHASE 4B — NAIVE BAYES MANGANESE PROSPECTIVITY MODEL")
    print("5-FOLD SPATIAL CROSS-VALIDATION")
    print("=" * 75)

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    print(f"Loaded balanced samples: {len(df)} (Positives: {(df['label'] == 1).sum()}, Background: {(df['label'] == 0).sum()})")

    # Spatial grouping
    df["spatial_block"] = (
        "block_"
        + (df["latitude"] // 0.05 * 0.05).round(4).astype(str)
        + "_"
        + (df["longitude"] // 0.05 * 0.05).round(4).astype(str)
    )
    groups = df["spatial_block"].to_numpy()

    # Features
    feature_drop = DROP_COLUMNS + SPATIAL_HELPER_COLUMNS + ["spatial_block"]
    feature_drop = [c for c in feature_drop if c in df.columns]
    X_raw = df.drop(columns=feature_drop).copy()
    y = df["label"].to_numpy(dtype=int)

    # Load saved preprocessor and feature columns
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    with open(FEATURE_COLUMNS_FILE, "r") as f:
        feature_names = json.load(f)

    # 5-fold StratifiedGroupKFold
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_raw, y, groups=groups), start=1):
        X_train_raw = X_raw.iloc[train_idx]
        X_val_raw = X_raw.iloc[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        # Fit preprocessor on train fold
        preprocessor_fold = joblib.load(PREPROCESSOR_FILE)
        X_train_trans = preprocessor_fold.fit_transform(X_train_raw)
        X_val_trans = preprocessor_fold.transform(X_val_raw)

        # Naive Bayes model with gentle variance smoothing for high-dimensional stability
        nb = GaussianNB(var_smoothing=1e-2)
        nb.fit(X_train_trans, y_train)

        y_prob = nb.predict_proba(X_val_trans)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else np.nan
        pr_auc = average_precision_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else np.nan

        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        acc = accuracy_score(y_val, y_pred)

        p_top5 = precision_at_fraction(y_val, y_prob, 0.05)
        p_top10 = precision_at_fraction(y_val, y_prob, 0.10)
        p_top20 = precision_at_fraction(y_val, y_prob, 0.20)

        fold_metrics.append({
            "fold": fold,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "accuracy": acc,
            "precision_at_5pct": p_top5,
            "precision_at_10pct": p_top10,
            "precision_at_20pct": p_top20
        })

    metrics_df = pd.DataFrame(fold_metrics)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(NB_METRICS_FILE, index=False)

    print("\n" + "=" * 75)
    print("NAIVE BAYES 5-FOLD SPATIAL CROSS-VALIDATION RESULTS")
    print("=" * 75)
    for col in ["roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy", "precision_at_5pct", "precision_at_10pct", "precision_at_20pct"]:
        mean = metrics_df[col].mean()
        std = metrics_df[col].std()
        print(f"  {col:22s}: {mean:.4f} ± {std:.4f}")

    # Train final Naive Bayes model on entire dataset
    print("\nTraining final Naive Bayes model on full Phase 4B feature set...")
    preprocessor_final = joblib.load(PREPROCESSOR_FILE)
    X_full_trans = preprocessor_final.transform(X_raw)
    final_nb = GaussianNB(var_smoothing=1e-2)
    final_nb.fit(X_full_trans, y)

    joblib.dump(final_nb, MODEL_FILE)
    print(f"Saved Naive Bayes model: {MODEL_FILE}")

    # Model comparison table
    comparison_rows = []
    if XGB_METRICS_FILE.exists():
        xgb_df = pd.read_csv(XGB_METRICS_FILE)
        for metric in ["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "precision_at_5pct", "precision_at_10pct", "precision_at_20pct"]:
            comparison_rows.append({
                "metric": metric,
                "XGBoost_mean": xgb_df[metric].mean(),
                "XGBoost_std": xgb_df[metric].std(),
                "NaiveBayes_mean": metrics_df[metric].mean(),
                "NaiveBayes_std": metrics_df[metric].std()
            })
        comp_df = pd.DataFrame(comparison_rows)
        comp_df.to_csv(COMPARISON_FILE, index=False)

        print("\n" + "=" * 75)
        print("MODEL BENCHMARK: XGBOOST vs NAIVE BAYES (5-Fold Spatial CV)")
        print("=" * 75)
        print(f"{'Metric':<22} | {'XGBoost (Phase 4B)':<22} | {'Naive Bayes (GaussianNB)':<22}")
        print("-" * 75)
        for _, r in comp_df.iterrows():
            xgb_str = f"{r['XGBoost_mean']:.4f} ± {r['XGBoost_std']:.4f}"
            nb_str = f"{r['NaiveBayes_mean']:.4f} ± {r['NaiveBayes_std']:.4f}"
            print(f"{r['metric']:<22} | {xgb_str:<22} | {nb_str:<22}")
        print("=" * 75)


if __name__ == "__main__":
    main()

