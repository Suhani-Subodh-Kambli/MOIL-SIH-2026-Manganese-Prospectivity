# ============================================================
# MOIL SIH 2026
# PHASE 4 - MANGANESE PROSPECTIVITY MODEL
# CORRECTED VERSION
#
# IMPORTANT:
# Distance_Mn is intentionally NOT used as an ML feature.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)

import shap


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = "data/MOIL_SIH_Phase3_ML_Features.csv"

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n========================================")
print("LOADING DATASET")
print("========================================")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 3. BASIC CLEANING
# ============================================================

df = df.dropna(how="all").copy()

df["label"] = pd.to_numeric(
    df["label"],
    errors="coerce"
)

df = df.dropna(
    subset=["label"]
).copy()

df["label"] = df["label"].astype(int)


# ============================================================
# 4. CLASS DISTRIBUTION
# ============================================================

print("\n========================================")
print("CLASS DISTRIBUTION")
print("========================================")

print(
    df["label"].value_counts()
)

print("\nPercentages:")

print(
    (df["label"].value_counts(normalize=True) * 100)
    .round(2)
)


# ============================================================
# 5. ML FEATURES
# ============================================================
#
# IMPORTANT:
#
# Distance_Mn is intentionally excluded.
#
# latitude and longitude are also excluded because they
# represent location rather than environmental evidence.
# ============================================================

feature_columns = [

    # Sentinel-2 spectral bands
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",

    # Spectral indices
    "NDVI",
    "NIR_Red_Ratio",
    "Red_Green_Ratio",
    "SWIR_NIR_Ratio",
    "SWIR_Ratio",

    # Sentinel-1
    "VV",
    "VH",
    "VV_VH_Difference",

    # Terrain
    "Elevation",
    "Slope",
    "Aspect"
]


print("\nNumber of ML features:", len(feature_columns))


# ============================================================
# 6. VERIFY FEATURES
# ============================================================

missing_columns = [
    col
    for col in feature_columns
    if col not in df.columns
]

if missing_columns:

    print("\nERROR: Missing columns")

    for col in missing_columns:
        print("-", col)

    raise SystemExit()


# ============================================================
# 7. PREPARE X AND Y
# ============================================================

X = df[feature_columns].copy()

y = df["label"].copy()


# Replace infinity
X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


# Fill missing values
X = X.fillna(
    X.median()
)


print(
    "\nTotal missing feature values:",
    X.isna().sum().sum()
)


# ============================================================
# 8. CREATE SPATIAL BLOCKS
# ============================================================
#
# We use geographic blocks rather than random pixels.
#
# This prevents nearby pixels from appearing in both
# training and testing data.
# ============================================================

print("\n========================================")
print("CREATING SPATIAL BLOCKS")
print("========================================")


longitude = df["longitude"].values
latitude = df["latitude"].values


# Approximately 10 km blocks
block_size = 0.10


block_x = np.floor(
    longitude / block_size
).astype(int)


block_y = np.floor(
    latitude / block_size
).astype(int)


groups = (
    block_x.astype(str)
    + "_"
    + block_y.astype(str)
)


print(
    "Number of spatial blocks:",
    len(np.unique(groups))
)


# ============================================================
# 9. STRATIFIED SPATIAL CROSS-VALIDATION
# ============================================================
#
# StratifiedGroupKFold keeps geographic blocks separate while
# attempting to preserve class balance.
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


splits = list(
    cv.split(
        X,
        y,
        groups=groups
    )
)


# Use first spatial fold as final test set
train_idx, test_idx = splits[0]


X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print(
    "\nTraining labels:"
)

print(
    y_train.value_counts()
)

print(
    "\nTesting labels:"
)

print(
    y_test.value_counts()
)


# ============================================================
# 10. TRAIN XGBOOST
# ============================================================

print("\n========================================")
print("TRAINING XGBOOST")
print("========================================")


negative_count = (
    y_train == 0
).sum()

positive_count = (
    y_train == 1
).sum()


scale_pos_weight = (
    negative_count / positive_count
)


print(
    "Scale positive weight:",
    round(scale_pos_weight, 3)
)


model = XGBClassifier(

    n_estimators=300,

    max_depth=5,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="binary:logistic",

    eval_metric="logloss",

    random_state=42,

    n_jobs=-1,

    scale_pos_weight=scale_pos_weight
)


model.fit(
    X_train,
    y_train
)


print(
    "\nTraining complete."
)


# ============================================================
# 11. PREDICTIONS
# ============================================================

probabilities = model.predict_proba(
    X_test
)[:, 1]


predictions = (
    probabilities >= 0.50
).astype(int)


# ============================================================
# 12. EVALUATION
# ============================================================

print("\n========================================")
print("MODEL EVALUATION")
print("========================================")


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


if len(np.unique(y_test)) == 2:

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

else:

    roc_auc = np.nan
    pr_auc = np.nan


print(
    f"\nROC-AUC: {roc_auc:.4f}"
)

print(
    f"PR-AUC: {pr_auc:.4f}"
)


precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)


print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall: {recall:.4f}"
)

print(
    f"F1 Score: {f1:.4f}"
)


# ============================================================
# 13. TOP-K PROSPECTIVITY
# ============================================================

def precision_at_k(
    y_true,
    probability,
    percentage
):

    y_true = np.asarray(y_true)
    probability = np.asarray(probability)

    k = max(
        1,
        int(len(probability) * percentage)
    )

    order = np.argsort(
        probability
    )[::-1]

    top_k = order[:k]

    return np.mean(
        y_true[top_k]
    )


print("\n========================================")
print("TOP-K PROSPECTIVITY")
print("========================================")


top_k_results = {}


for percentage in [
    0.05,
    0.10,
    0.20
]:

    score = precision_at_k(
        y_test.values,
        probabilities,
        percentage
    )

    top_k_results[
        f"Precision@Top{int(percentage*100)}"
    ] = score

    print(
        f"Top {int(percentage*100)}%: "
        f"{score:.4f}"
    )


# ============================================================
# 14. FEATURE IMPORTANCE
# ============================================================

print("\n========================================")
print("FEATURE IMPORTANCE")
print("========================================")


importance = pd.DataFrame({

    "feature": feature_columns,

    "importance":
        model.feature_importances_

})


importance = importance.sort_values(
    "importance",
    ascending=False
)


print(
    importance.to_string(
        index=False
    )
)


importance.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "feature_importance.csv"
    ),
    index=False
)


# ============================================================
# 15. FEATURE IMPORTANCE PLOT
# ============================================================

top = importance.head(15)


plt.figure(
    figsize=(10, 8)
)


plt.barh(
    top["feature"][::-1],
    top["importance"][::-1]
)


plt.xlabel(
    "XGBoost Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top Features - Manganese Prospectivity"
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "feature_importance.png"
    ),
    dpi=200
)


plt.close()


# ============================================================
# 16. SHAP
# ============================================================

print(
    "\nCalculating SHAP explanations..."
)


sample_size = min(
    500,
    len(X_test)
)


X_shap = X_test.iloc[
    :sample_size
]


explainer = shap.TreeExplainer(
    model
)


shap_values = explainer(
    X_shap
)


plt.figure()


shap.plots.beeswarm(
    shap_values,
    max_display=15,
    show=False
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "shap_summary.png"
    ),
    dpi=200,
    bbox_inches="tight"
)


plt.close()


print(
    "SHAP plot saved."
)


# ============================================================
# 17. SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "moil_manganese_prospectivity_xgb.joblib"
)


joblib.dump(
    model,
    model_path
)


print(
    "\nModel saved:"
)

print(
    model_path
)


# ============================================================
# 18. SAVE TEST PREDICTIONS
# ============================================================

test_results = df.iloc[
    test_idx
].copy()


test_results[
    "prospectivity_probability"
] = probabilities


test_results[
    "prospectivity_score"
] = (
    probabilities * 100
)


test_results[
    "predicted_class"
] = predictions


test_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test_predictions.csv"
    ),
    index=False
)


# ============================================================
# 19. SAVE METRICS
# ============================================================

metrics = {

    "ROC_AUC": roc_auc,

    "PR_AUC": pr_auc,

    "Precision": precision,

    "Recall": recall,

    "F1": f1

}


metrics.update(
    top_k_results
)


pd.DataFrame(
    [metrics]
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_metrics.csv"
    ),
    index=False
)


# ============================================================
# 20. FINAL
# ============================================================

print("\n========================================")
print("PHASE 4 COMPLETE")
print("========================================")

print(
    "\nGenerated files:"
)

print(
    "models/moil_manganese_prospectivity_xgb.joblib"
)

print(
    "outputs/feature_importance.csv"
)

print(
    "outputs/feature_importance.png"
)

print(
    "outputs/shap_summary.png"
)

print(
    "outputs/test_predictions.csv"
)

print(
    "outputs/model_metrics.csv"
)