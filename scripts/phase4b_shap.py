import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib


# ============================================================
# PATHS
# ============================================================

DATA_FILE = Path(
    "data/MOIL_SIH_Phase4B_Balanced_Features.csv"
)

MODEL_FILE = Path(
    "models/moil_manganese_prospectivity_xgb_phase4b.joblib"
)

PREPROCESSOR_FILE = Path(
    "models/phase4b_preprocessor.joblib"
)

FEATURE_FILE = Path(
    "models/phase4b_feature_columns.json"
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHAP_SUMMARY = (
    OUTPUT_DIR / "phase4b_shap_summary.png"
)

SHAP_BAR = (
    OUTPUT_DIR / "phase4b_shap_bar.png"
)

SHAP_IMPORTANCE = (
    OUTPUT_DIR / "phase4b_shap_importance.csv"
)

SHAP_VALUES_FILE = (
    OUTPUT_DIR / "phase4b_shap_values.csv"
)


# ============================================================
# LOAD DATA + MODEL
# ============================================================

print("=" * 70)
print("PHASE 4B.7 — SHAP EXPLAINABILITY")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

model = joblib.load(MODEL_FILE)

preprocessor = joblib.load(PREPROCESSOR_FILE)

with open(
    FEATURE_FILE,
    "r",
    encoding="utf-8"
) as f:
    feature_columns = json.load(f)


print(f"\nInput shape: {df.shape}")
print(f"Model feature count: {len(model.feature_names_in_)}")
print(f"Saved feature count: {len(feature_columns)}")


# ============================================================
# SAFETY CHECK
# ============================================================

if len(model.feature_names_in_) != len(feature_columns):

    raise RuntimeError(
        "Mismatch between model features and saved feature columns."
    )


if list(model.feature_names_in_) != list(feature_columns):

    raise RuntimeError(
        "Feature order mismatch between model and feature_columns.json."
    )


# ============================================================
# RAW MODEL INPUT
# ============================================================

drop_columns = [
    "label",
    "Distance_Mn",
    "longitude",
    "latitude",
    "system:index",
    ".geo"
]

X_raw = df.drop(
    columns=drop_columns,
    errors="ignore"
).copy()


# ============================================================
# CLEAN CATEGORICAL COLUMNS
# ============================================================

categorical_columns = [
    "geo_age",
    "geo_supergroup",
    "geo_group",
    "geo_formation",
    "geo_lithology",
    "geo_intrusive",
    "geo_stratigraphy"
]

categorical_columns = [
    col
    for col in categorical_columns
    if col in X_raw.columns
]

for col in categorical_columns:

    X_raw[col] = (
        X_raw[col]
        .fillna("UNKNOWN")
        .astype(str)
    )


# ============================================================
# CLEAN NUMERIC COLUMNS
# ============================================================

numeric_columns = [
    col
    for col in X_raw.columns
    if col not in categorical_columns
]

for col in numeric_columns:

    X_raw[col] = pd.to_numeric(
        X_raw[col],
        errors="coerce"
    )

X_raw = X_raw.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# USE THE EXACT TRAINING PREPROCESSOR
# ============================================================

print("\nTransforming data using saved Phase 4B preprocessor...")

X_transformed = preprocessor.transform(X_raw)


# ============================================================
# CONVERT TO DATAFRAME
# ============================================================

X_transformed = np.asarray(
    X_transformed,
    dtype=float
)

print(
    f"Transformed shape: "
    f"{X_transformed.shape}"
)


if X_transformed.shape[1] != len(feature_columns):

    raise RuntimeError(
        f"Expected {len(feature_columns)} features "
        f"but preprocessing produced "
        f"{X_transformed.shape[1]}."
    )


X = pd.DataFrame(
    X_transformed,
    columns=feature_columns,
    index=df.index
)


# ============================================================
# SAMPLE FOR SHAP
# ============================================================

MAX_SHAP_SAMPLES = min(
    500,
    len(X)
)

X_shap = X.sample(
    n=MAX_SHAP_SAMPLES,
    random_state=42
)

print(
    f"SHAP samples: "
    f"{len(X_shap)}"
)


# ============================================================
# SHAP EXPLAINER
# ============================================================

print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(
    model
)

shap_values = explainer.shap_values(
    X_shap
)

shap_values = np.asarray(
    shap_values
)


# ============================================================
# HANDLE SHAP OUTPUT
# ============================================================

# Binary XGBoost normally returns:
# (samples, features)

if shap_values.ndim == 3:

    # Defensive handling for possible
    # multi-output SHAP formats.

    shap_values = shap_values[:, :, 1]


if shap_values.ndim != 2:

    raise RuntimeError(
        f"Unexpected SHAP shape: "
        f"{shap_values.shape}"
    )


if shap_values.shape[1] != len(feature_columns):

    raise RuntimeError(
        f"SHAP feature count "
        f"{shap_values.shape[1]} does not match "
        f"{len(feature_columns)}."
    )


print(
    f"SHAP matrix shape: "
    f"{shap_values.shape}"
)


# ============================================================
# MEAN ABSOLUTE SHAP IMPORTANCE
# ============================================================

mean_abs_shap = np.abs(
    shap_values
).mean(axis=0)


shap_importance = pd.DataFrame(
    {
        "feature": feature_columns,
        "mean_abs_shap": mean_abs_shap
    }
).sort_values(
    "mean_abs_shap",
    ascending=False
).reset_index(
    drop=True
)


shap_importance.to_csv(
    SHAP_IMPORTANCE,
    index=False
)


# ============================================================
# SAVE SHAP VALUES
# ============================================================

shap_values_df = pd.DataFrame(
    shap_values,
    columns=feature_columns
)

shap_values_df.to_csv(
    SHAP_VALUES_FILE,
    index=False
)


# ============================================================
# SHAP SUMMARY PLOT
# ============================================================

print("\nCreating SHAP summary plot...")

plt.figure()

shap.summary_plot(
    shap_values,
    X_shap,
    show=False,
    max_display=25
)

plt.tight_layout()

plt.savefig(
    SHAP_SUMMARY,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SHAP BAR PLOT
# ============================================================

print("Creating SHAP bar plot...")

plt.figure()

shap.summary_plot(
    shap_values,
    X_shap,
    plot_type="bar",
    show=False,
    max_display=25
)

plt.tight_layout()

plt.savefig(
    SHAP_BAR,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("PHASE 4B.7 COMPLETE")
print("=" * 70)

print(
    "\nTop 30 SHAP features:"
)

print(
    shap_importance
    .head(30)
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(SHAP_SUMMARY)
print(SHAP_BAR)
print(SHAP_IMPORTANCE)
print(SHAP_VALUES_FILE)

print("\nPHASE 4B.7 COMPLETE")