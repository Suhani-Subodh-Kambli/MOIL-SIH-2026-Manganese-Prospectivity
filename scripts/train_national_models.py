"""
scripts/train_national_models.py

Trains and validates India-Wide Multi-Belt Manganese Prospectivity Models:
- National XGBoost Classifier
- National Gaussian Naive Bayes Classifier
- National 50/50 Probabilistic Ensemble

Rigorous Validation:
1. 5-Fold Spatial Cross-Validation (StratifiedGroupKFold on spatial_block)
2. Out-of-Craton Transferability Cross-Validation (GroupKFold on craton_domain)

Outputs:
- models/national_preprocessor.joblib
- models/national_feature_columns.json
- models/moil_manganese_prospectivity_xgb_national.joblib
- models/moil_manganese_prospectivity_nb_national.joblib
- outputs/national_model_comparison.csv
- outputs/national_spatial_cv_metrics.csv
- outputs/national_craton_cv_metrics.csv
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import json
import warnings
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
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

DATA_FILE = BASE_DIR / "data" / "MOIL_National_MultiBelt_Dataset.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

PREPROCESSOR_FILE = MODEL_DIR / "national_preprocessor.joblib"
FEATURE_COLS_FILE = MODEL_DIR / "national_feature_columns.json"
XGB_MODEL_FILE = MODEL_DIR / "moil_manganese_prospectivity_xgb_national.joblib"
NB_MODEL_FILE = MODEL_DIR / "moil_manganese_prospectivity_nb_national.joblib"

COMPARISON_FILE = OUTPUT_DIR / "national_model_comparison.csv"
SPATIAL_METRICS_FILE = OUTPUT_DIR / "national_spatial_cv_metrics.csv"
CRATON_METRICS_FILE = OUTPUT_DIR / "national_craton_cv_metrics.csv"

DROP_COLUMNS = [
    "label",
    "name",
    "system:index",
    "latitude",
    "longitude",
    "Distance_Mn",
    "spatial_block",
    "craton_domain",
    "state",
    "district",
    "metallogenic_belt",
    "host_rock",
    "formation",
    "source",
    "deposit_type",
    "reserve",
    "grade",
    "metallogenic_zone_no",
    "occurrence_id",
    ".geo",
    "geo_age",
    "geo_supergroup",
    "geo_group",
    "geo_formation",
    "geo_lithology",
    "geo_intrusive",
    "geo_stratigraphy",
]

NUMERIC_COLUMNS = [
    "Aspect", "B11", "B12", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A",
    "Elevation", "NDVI", "NIR_Red_Ratio", "Red_Green_Ratio", "SWIR_NIR_Ratio",
    "SWIR_Ratio", "Slope", "VH", "VV", "VV_VH_Difference", "NDMI", "NBR", "NDRE",
    "Iron_Oxide_Index", "Clay_Alteration_Index", "Ferrous_Index", "SWIR_Red_Ratio",
    "SWIR_Green_Ratio", "NIR_SWIR2_Ratio", "BSI", "B11_B12_NormDiff", "B8_B12_NormDiff",
    "B4_B2_NormDiff", "Sausar_Group_Proxy", "Metamorphic_Host_Proxy", "Geology_Unknown",
    "Geo_Boundary_Distance_km", "Geo_Boundary_Density_1km", "Geo_Boundary_Density_3km",
    "Lithology_Diversity_3km", "Formation_Diversity_3km", "Aspect_Sin", "Aspect_Cos",
    "metallogenic_host_affinity"
]

CATEGORICAL_COLUMNS = [
    "macro_lithology"
]


def precision_at_top_k(y_true: np.ndarray, y_prob: np.ndarray, k_pct: float) -> float:
    """Calculates precision among top k% scored candidates."""
    n = len(y_prob)
    k = max(1, int(n * k_pct))
    top_indices = np.argsort(y_prob)[::-1][:k]
    return float(np.mean(y_true[top_indices]))


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision_at_5pct": precision_at_top_k(y_true, y_prob, 0.05),
        "precision_at_10pct": precision_at_top_k(y_true, y_prob, 0.10),
        "precision_at_20pct": precision_at_top_k(y_true, y_prob, 0.20),
    }


def main():
    print("=" * 75)
    print("PHASE 3: TRAINING NATIONAL MULTI-BELT PROSPECTIVITY MODELS")
    print("=" * 75)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Dataset
    print(f"Loading national dataset: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"Total samples: {len(df)} | Positives: {(df['label'] == 1).sum()} | Background: {(df['label'] == 0).sum()}")

    # Ensure required columns
    for c in NUMERIC_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    for c in CATEGORICAL_COLUMNS:
        if c not in df.columns:
            df[c] = "OTHER_UNDIVIDED"
        else:
            df[c] = df[c].fillna("OTHER_UNDIVIDED").astype(str)

    feature_cols = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
    X_raw = df[feature_cols].copy()
    y = df["label"].to_numpy()
    spatial_blocks = df["spatial_block"].to_numpy()
    craton_domains = df["craton_domain"].to_numpy()

    # Build Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC_COLUMNS),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), CATEGORICAL_COLUMNS)
        ]
    )

    # Fit preprocessor on full data to establish fixed column schema
    preprocessor.fit(X_raw)
    ohe_cols = preprocessor.named_transformers_["categorical"]["onehot"].get_feature_names_out(CATEGORICAL_COLUMNS)
    all_feature_names = [f"numeric__{c}" for c in NUMERIC_COLUMNS] + [f"categorical__{c}" for c in ohe_cols]
    print(f"Total Transformed Model Features: {len(all_feature_names)} ({len(NUMERIC_COLUMNS)} numeric + {len(ohe_cols)} one-hot)")

    joblib.dump(preprocessor, PREPROCESSOR_FILE)
    with open(FEATURE_COLS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_feature_names, f, indent=2)
    print(f"Saved preprocessor: {PREPROCESSOR_FILE}")
    print(f"Saved feature schema: {FEATURE_COLS_FILE}")

    # ============================================================
    # 2. EVALUATION 1: 5-FOLD SPATIAL CROSS-VALIDATION
    # ============================================================
    print("\n" + "=" * 75)
    print("EVALUATION 1: 5-FOLD SPATIAL CROSS-VALIDATION (StratifiedGroupKFold)")
    print("=" * 75)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    xgb_spatial_results = []
    nb_spatial_results = []
    ens_spatial_results = []

    fold = 0
    for train_idx, val_idx in sgkf.split(X_raw, y, groups=spatial_blocks):
        fold += 1
        X_tr_raw, y_tr = X_raw.iloc[train_idx], y[train_idx]
        X_va_raw, y_va = X_raw.iloc[val_idx], y[val_idx]

        # Fit preprocessor strictly on train fold
        fold_preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC_COLUMNS),
                ("categorical", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                ]), CATEGORICAL_COLUMNS)
            ]
        )
        X_tr = fold_preprocessor.fit_transform(X_tr_raw)
        X_va = fold_preprocessor.transform(X_va_raw)

        # Align columns with global schema
        fold_ohe = fold_preprocessor.named_transformers_["categorical"]["onehot"].get_feature_names_out(CATEGORICAL_COLUMNS)
        fold_cols = [f"numeric__{c}" for c in NUMERIC_COLUMNS] + [f"categorical__{c}" for c in fold_ohe]
        X_tr_df = pd.DataFrame(X_tr, columns=fold_cols).reindex(columns=all_feature_names, fill_value=0.0)
        X_va_df = pd.DataFrame(X_va, columns=fold_cols).reindex(columns=all_feature_names, fill_value=0.0)

        # Train XGBoost
        scale_pos = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())
        xgb = XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.80,
            scale_pos_weight=scale_pos,
            random_state=SEED,
            eval_metric="logloss"
        )
        xgb.fit(X_tr_df, y_tr)
        p_xgb = xgb.predict_proba(X_va_df)[:, 1]

        # Train Naive Bayes
        nb = GaussianNB(var_smoothing=1e-2)
        nb.fit(X_tr_df.values, y_tr)
        p_nb = nb.predict_proba(X_va_df.values)[:, 1]

        # Ensemble (50-50 blend)
        p_ens = 0.5 * p_xgb + 0.5 * p_nb

        res_xgb = evaluate_predictions(y_va, p_xgb)
        res_nb = evaluate_predictions(y_va, p_nb)
        res_ens = evaluate_predictions(y_va, p_ens)

        res_xgb["fold"] = fold
        res_nb["fold"] = fold
        res_ens["fold"] = fold

        xgb_spatial_results.append(res_xgb)
        nb_spatial_results.append(res_nb)
        ens_spatial_results.append(res_ens)

        print(f"Fold {fold} | Val Samples: {len(y_va)} (Pos: {(y_va==1).sum()}) | "
              f"XGB PR-AUC: {res_xgb['pr_auc']:.4f} | NB PR-AUC: {res_nb['pr_auc']:.4f} | Ens PR-AUC: {res_ens['pr_auc']:.4f}")

    xgb_sp_df = pd.DataFrame(xgb_spatial_results)
    nb_sp_df = pd.DataFrame(nb_spatial_results)
    ens_sp_df = pd.DataFrame(ens_spatial_results)

    # Save spatial metrics
    spatial_summary = pd.DataFrame([
        {"model": "XGBoost", **{f"{k}_mean": xgb_sp_df[k].mean() for k in xgb_sp_df.columns if k != "fold"},
         **{f"{k}_std": xgb_sp_df[k].std() for k in xgb_sp_df.columns if k != "fold"}},
        {"model": "Naive Bayes", **{f"{k}_mean": nb_sp_df[k].mean() for k in nb_sp_df.columns if k != "fold"},
         **{f"{k}_std": nb_sp_df[k].std() for k in nb_sp_df.columns if k != "fold"}},
        {"model": "Ensemble", **{f"{k}_mean": ens_sp_df[k].mean() for k in ens_sp_df.columns if k != "fold"},
         **{f"{k}_std": ens_sp_df[k].std() for k in ens_sp_df.columns if k != "fold"}}
    ])
    spatial_summary.to_csv(SPATIAL_METRICS_FILE, index=False)

    # ============================================================
    # 3. EVALUATION 2: OUT-OF-CRATON CROSS-VALIDATION
    # ============================================================
    print("\n" + "=" * 75)
    print("EVALUATION 2: OUT-OF-CRATON TRANSFERABILITY (GroupKFold on craton_domain)")
    print("=" * 75)

    gkf = GroupKFold(n_splits=5)
    xgb_craton_results = []
    nb_craton_results = []
    ens_craton_results = []

    c_fold = 0
    for train_idx, val_idx in gkf.split(X_raw, y, groups=craton_domains):
        c_fold += 1
        held_out_cratons = list(set(craton_domains[val_idx]))
        X_tr_raw, y_tr = X_raw.iloc[train_idx], y[train_idx]
        X_va_raw, y_va = X_raw.iloc[val_idx], y[val_idx]

        if len(set(y_va)) < 2:
            continue

        fold_preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC_COLUMNS),
                ("categorical", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                ]), CATEGORICAL_COLUMNS)
            ]
        )
        X_tr = fold_preprocessor.fit_transform(X_tr_raw)
        X_va = fold_preprocessor.transform(X_va_raw)

        fold_ohe = fold_preprocessor.named_transformers_["categorical"]["onehot"].get_feature_names_out(CATEGORICAL_COLUMNS)
        fold_cols = [f"numeric__{c}" for c in NUMERIC_COLUMNS] + [f"categorical__{c}" for c in fold_ohe]
        X_tr_df = pd.DataFrame(X_tr, columns=fold_cols).reindex(columns=all_feature_names, fill_value=0.0)
        X_va_df = pd.DataFrame(X_va, columns=fold_cols).reindex(columns=all_feature_names, fill_value=0.0)

        scale_pos = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())
        xgb = XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.80,
            scale_pos_weight=scale_pos,
            random_state=SEED,
            eval_metric="logloss"
        )
        xgb.fit(X_tr_df, y_tr)
        p_xgb = xgb.predict_proba(X_va_df)[:, 1]

        nb = GaussianNB(var_smoothing=1e-2)
        nb.fit(X_tr_df.values, y_tr)
        p_nb = nb.predict_proba(X_va_df.values)[:, 1]

        p_ens = 0.5 * p_xgb + 0.5 * p_nb

        res_xgb = evaluate_predictions(y_va, p_xgb)
        res_nb = evaluate_predictions(y_va, p_nb)
        res_ens = evaluate_predictions(y_va, p_ens)

        res_xgb["held_out"] = ", ".join(held_out_cratons)
        res_nb["held_out"] = ", ".join(held_out_cratons)
        res_ens["held_out"] = ", ".join(held_out_cratons)

        xgb_craton_results.append(res_xgb)
        nb_craton_results.append(res_nb)
        ens_craton_results.append(res_ens)

        print(f"Craton Fold {c_fold} | Held Out: {res_xgb['held_out']} | Samples: {len(y_va)} | "
              f"XGB PR-AUC: {res_xgb['pr_auc']:.4f} | NB PR-AUC: {res_nb['pr_auc']:.4f} | Ens PR-AUC: {res_ens['pr_auc']:.4f}")

    xgb_cr_df = pd.DataFrame(xgb_craton_results)
    nb_cr_df = pd.DataFrame(nb_craton_results)
    ens_cr_df = pd.DataFrame(ens_craton_results)

    craton_summary = pd.DataFrame([
        {"model": "XGBoost", **{f"{k}_mean": xgb_cr_df[k].mean() for k in xgb_cr_df.columns if k not in ["held_out"]},
         **{f"{k}_std": xgb_cr_df[k].std() for k in xgb_cr_df.columns if k not in ["held_out"]}},
        {"model": "Naive Bayes", **{f"{k}_mean": nb_cr_df[k].mean() for k in nb_cr_df.columns if k not in ["held_out"]},
         **{f"{k}_std": nb_cr_df[k].std() for k in nb_cr_df.columns if k not in ["held_out"]}},
        {"model": "Ensemble", **{f"{k}_mean": ens_cr_df[k].mean() for k in ens_cr_df.columns if k not in ["held_out"]},
         **{f"{k}_std": ens_cr_df[k].std() for k in ens_cr_df.columns if k not in ["held_out"]}}
    ])
    craton_summary.to_csv(CRATON_METRICS_FILE, index=False)

    # ============================================================
    # 4. TRAIN FINAL PRODUCTION NATIONAL MODELS
    # ============================================================
    print("\n" + "=" * 75)
    print("TRAINING FINAL PRODUCTION NATIONAL MODELS ON FULL DATASET")
    print("=" * 75)

    X_full_trans = preprocessor.transform(X_raw)
    X_full_df = pd.DataFrame(X_full_trans, columns=all_feature_names)

    # Train final XGBoost
    scale_pos_full = (y == 0).sum() / max(1, (y == 1).sum())
    final_xgb = XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.80,
        scale_pos_weight=scale_pos_full,
        random_state=SEED,
        eval_metric="logloss"
    )
    final_xgb.fit(X_full_df, y)
    joblib.dump(final_xgb, XGB_MODEL_FILE)
    print(f"Saved National XGBoost Model: {XGB_MODEL_FILE}")

    # Train final Naive Bayes
    final_nb = GaussianNB(var_smoothing=1e-2)
    final_nb.fit(X_full_df.values, y)
    joblib.dump(final_nb, NB_MODEL_FILE)
    print(f"Saved National Naive Bayes Model: {NB_MODEL_FILE}")

    # Top Feature Importances
    fi = sorted(zip(all_feature_names, final_xgb.feature_importances_), key=lambda x: x[1], reverse=True)
    print("\nTop 10 Drivers of National Manganese Prospectivity (XGBoost):")
    for f_name, f_val in fi[:10]:
        print(f"  {f_name:45s}: {f_val:.4f}")

    # Model Comparison Summary Table
    comp_rows = []
    metrics_to_show = ["pr_auc", "roc_auc", "accuracy", "precision", "recall", "f1", "precision_at_10pct"]
    for m in metrics_to_show:
        comp_rows.append({
            "metric": m,
            "Spatial_XGB": f"{xgb_sp_df[m].mean():.4f} ± {xgb_sp_df[m].std():.4f}",
            "Spatial_NB": f"{nb_sp_df[m].mean():.4f} ± {nb_sp_df[m].std():.4f}",
            "Spatial_Ensemble": f"{ens_sp_df[m].mean():.4f} ± {ens_sp_df[m].std():.4f}",
            "OutOfCraton_XGB": f"{xgb_cr_df[m].mean():.4f} ± {xgb_cr_df[m].std():.4f}",
            "OutOfCraton_NB": f"{nb_cr_df[m].mean():.4f} ± {nb_cr_df[m].std():.4f}",
            "OutOfCraton_Ensemble": f"{ens_cr_df[m].mean():.4f} ± {ens_cr_df[m].std():.4f}"
        })
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(COMPARISON_FILE, index=False)

    print("\n" + "=" * 90)
    print("NATIONAL PROSPECTIVITY MODEL BENCHMARK SUMMARY")
    print("=" * 90)
    print(f"{'Metric':<20} | {'Spatial XGB':<18} | {'Spatial NB':<18} | {'Spatial Ens':<18} | {'Out-Craton Ens':<18}")
    print("-" * 90)
    for _, r in comp_df.iterrows():
        print(f"{r['metric']:<20} | {r['Spatial_XGB']:<18} | {r['Spatial_NB']:<18} | {r['Spatial_Ensemble']:<18} | {r['OutOfCraton_Ensemble']:<18}")
    print("=" * 90)


if __name__ == "__main__":
    main()
