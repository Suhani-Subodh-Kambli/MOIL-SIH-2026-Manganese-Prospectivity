# scripts/train_phase4b.py

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

SEED = 42

INPUT_FILE = Path(
    "data/MOIL_SIH_Phase4B_Balanced_Features.csv"
)

MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

MODEL_FILE = (
    MODEL_DIR
    / "moil_manganese_prospectivity_xgb_phase4b.joblib"
)

PREPROCESSOR_FILE = (
    MODEL_DIR
    / "phase4b_preprocessor.joblib"
)

FEATURE_COLUMNS_FILE = (
    MODEL_DIR
    / "phase4b_feature_columns.json"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "phase4b_spatial_metrics.csv"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "phase4b_fold_predictions.csv"
)

FEATURE_IMPORTANCE_FILE = (
    OUTPUT_DIR
    / "phase4b_feature_importance.csv"
)

FEATURE_IMPORTANCE_PLOT = (
    OUTPUT_DIR
    / "phase4b_feature_importance.png"
)

BLOCK_AUDIT_FILE = (
    OUTPUT_DIR
    / "phase4b_spatial_block_audit.csv"
)


# ============================================================
# COLUMNS TO EXCLUDE FROM MODEL
# ============================================================

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

CATEGORICAL_COLUMNS = [
    "geo_age",
    "geo_supergroup",
    "geo_group",
    "geo_formation",
    "geo_lithology",
    "geo_intrusive",
    "geo_stratigraphy",
]


# ============================================================
# XGBOOST MODEL
# ============================================================

def make_model(scale_pos_weight):

    return XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.80,
        min_child_weight=3,
        reg_lambda=1.0,
        reg_alpha=0.05,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
    )


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor(
    numeric_columns,
    categorical_columns
):
    """
    Build preprocessing pipeline.

    Numeric:
        median imputation

    Categorical:
        most-frequent imputation
        one-hot encoding

    The preprocessor is fitted separately inside every
    spatial validation fold to avoid preprocessing leakage.
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            ),
        ],
        remainder="drop"
    )

    return preprocessor


# ============================================================
# PRECISION @ TOP FRACTION
# ============================================================

def precision_at_fraction(
    y_true,
    y_probability,
    fraction
):
    """
    Calculate precision among the highest-scoring
    fraction of validation samples.
    """

    y_true = np.asarray(y_true)
    y_probability = np.asarray(y_probability)

    n = len(y_true)

    if n == 0:
        return np.nan

    k = max(
        1,
        int(np.ceil(n * fraction))
    )

    order = np.argsort(
        -y_probability
    )

    top_indices = order[:k]

    return float(
        np.mean(
            y_true[top_indices]
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PHASE 4B.5 + 4B.6")
    print("GEOLOGY + SPECTRAL XGBOOST")
    print("5-FOLD SPATIAL VALIDATION")
    print("=" * 70)

    # ========================================================
    # CHECK FILES
    # ========================================================

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    df = pd.read_csv(
        INPUT_FILE
    )

    print()
    print(
        f"Input shape: {df.shape}"
    )

    # ========================================================
    # CHECK REQUIRED COLUMNS
    # ========================================================

    required_columns = {
        "label",
        "latitude",
        "longitude"
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Missing required columns: "
            + str(sorted(missing))
        )

    # ========================================================
    # BASIC CLEANING
    # ========================================================

    df = df.copy()

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "label",
            "latitude",
            "longitude"
        ]
    ).copy()

    df["label"] = (
        df["label"]
        .astype(int)
    )

    # ========================================================
    # DUPLICATE COORDINATE SAFETY
    # ========================================================

    duplicate_coordinates = int(
        df.duplicated(
            subset=[
                "latitude",
                "longitude"
            ]
        ).sum()
    )

    print()
    print(
        "Duplicate coordinates found: "
        f"{duplicate_coordinates}"
    )

    if duplicate_coordinates > 0:

        df = df.drop_duplicates(
            subset=[
                "latitude",
                "longitude"
            ],
            keep="first"
        ).copy()

        print(
            "After duplicate removal: "
            f"{len(df)} records"
        )

    # ========================================================
    # LABEL DISTRIBUTION
    # ========================================================

    y = df["label"].to_numpy(
        dtype=int
    )

    label_counts = (
        pd.Series(y)
        .value_counts()
        .sort_index()
    )

    print()
    print("Label distribution:")
    print(
        label_counts.to_string()
    )

    positive_count = int(
        np.sum(y == 1)
    )

    negative_count = int(
        np.sum(y == 0)
    )

    if positive_count == 0:
        raise ValueError(
            "No positive samples found."
        )

    if negative_count == 0:
        raise ValueError(
            "No negative samples found."
        )

    # ========================================================
    # SPATIAL BLOCKS
    #
    # 0.10 degree is approximately 10-11 km around
    # the Balaghat region.
    # ========================================================

    BLOCK_SIZE = 0.10

    df["spatial_block"] = (
        np.floor(
            df["latitude"]
            / BLOCK_SIZE
        )
        .astype(int)
        .astype(str)
        + "_"
        +
        np.floor(
            df["longitude"]
            / BLOCK_SIZE
        )
        .astype(int)
        .astype(str)
    )

    groups = (
        df["spatial_block"]
        .to_numpy()
    )

    number_of_blocks = (
        df["spatial_block"]
        .nunique()
    )

    print()
    print(
        f"Spatial blocks: "
        f"{number_of_blocks}"
    )

    # ========================================================
    # SPATIAL BLOCK AUDIT
    # ========================================================

    block_audit = (
        df.groupby(
            "spatial_block"
        )
        .agg(
            samples=("label", "size"),
            positives=("label", "sum")
        )
        .reset_index()
    )

    block_audit["negatives"] = (
        block_audit["samples"]
        - block_audit["positives"]
    )

    block_audit["positive_rate"] = (
        block_audit["positives"]
        / block_audit["samples"]
    )

    block_audit.to_csv(
        BLOCK_AUDIT_FILE,
        index=False
    )

    # ========================================================
    # BUILD RAW FEATURE TABLE
    # ========================================================

    feature_drop = (
        DROP_COLUMNS
        + SPATIAL_HELPER_COLUMNS
        + ["spatial_block"]
    )

    feature_drop = [
        column
        for column in feature_drop
        if column in df.columns
    ]

    X_raw = df.drop(
        columns=feature_drop
    ).copy()

    # ========================================================
    # DETECT CATEGORICAL FEATURES
    # ========================================================

    categorical_columns = [
        column
        for column in CATEGORICAL_COLUMNS
        if column in X_raw.columns
    ]

    numeric_columns = [
        column
        for column in X_raw.columns
        if column not in categorical_columns
    ]

    print()
    print(
        f"Numeric features: "
        f"{len(numeric_columns)}"
    )

    print(
        f"Categorical features: "
        f"{len(categorical_columns)}"
    )

    print()
    print(
        "Categorical columns:"
    )

    for column in categorical_columns:
        print(
            f"  - {column}"
        )

    # ========================================================
    # CLEAN CATEGORICAL VALUES
    # ========================================================

    for column in categorical_columns:

        X_raw[column] = (
            X_raw[column]
            .fillna("UNKNOWN")
            .astype(str)
        )

    # ========================================================
    # CLEAN NUMERIC VALUES
    # ========================================================

    for column in numeric_columns:

        X_raw[column] = pd.to_numeric(
            X_raw[column],
            errors="coerce"
        )

        X_raw[column] = (
            X_raw[column]
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
        )

    # ========================================================
    # CHECK FOR ACCIDENTAL LEAKAGE COLUMNS
    # ========================================================

    forbidden_keywords = [
        "distance_mn",
        "label",
    ]

    suspicious_features = [
        column
        for column in X_raw.columns
        if any(
            keyword in column.lower()
            for keyword in forbidden_keywords
        )
    ]

    if suspicious_features:

        raise RuntimeError(
            "Potential leakage columns remain in "
            f"model features: {suspicious_features}"
        )

    # ========================================================
    # 5-FOLD STRATIFIED GROUP CV
    # ========================================================

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    fold_metrics = []
    fold_predictions = []

    # ========================================================
    # CROSS-VALIDATION LOOP
    # ========================================================

    for fold, (
        train_idx,
        val_idx
    ) in enumerate(
        cv.split(
            X_raw,
            y,
            groups=groups
        ),
        start=1
    ):

        print()
        print("=" * 70)
        print(
            f"FOLD {fold}"
        )
        print("=" * 70)

        X_train_raw = (
            X_raw
            .iloc[train_idx]
            .copy()
        )

        X_val_raw = (
            X_raw
            .iloc[val_idx]
            .copy()
        )

        y_train = y[
            train_idx
        ]

        y_val = y[
            val_idx
        ]

        train_groups = (
            set(
                groups[train_idx]
            )
        )

        validation_groups = (
            set(
                groups[val_idx]
            )
        )

        overlap = (
            train_groups
            & validation_groups
        )

        if overlap:

            raise RuntimeError(
                "Spatial leakage detected. "
                f"Overlapping blocks: {overlap}"
            )

        print(
            f"Train size: "
            f"{len(train_idx)}"
        )

        print(
            f"Validation size: "
            f"{len(val_idx)}"
        )

        print(
            f"Train positives: "
            f"{int(y_train.sum())}"
        )

        print(
            f"Validation positives: "
            f"{int(y_val.sum())}"
        )

        print(
            f"Train spatial blocks: "
            f"{len(train_groups)}"
        )

        print(
            f"Validation spatial blocks: "
            f"{len(validation_groups)}"
        )

        # ====================================================
        # FOLD-SPECIFIC PREPROCESSING
        # ====================================================

        preprocessor = build_preprocessor(
            numeric_columns,
            categorical_columns
        )

        X_train = (
            preprocessor
            .fit_transform(
                X_train_raw
            )
        )

        X_val = (
            preprocessor
            .transform(
                X_val_raw
            )
        )

        # ====================================================
        # CLASS WEIGHT
        # ====================================================

        train_positive = int(
            np.sum(y_train == 1)
        )

        train_negative = int(
            np.sum(y_train == 0)
        )

        scale_pos_weight = (
            train_negative
            / max(
                train_positive,
                1
            )
        )

        # ====================================================
        # CREATE MODEL
        # ====================================================

        model = make_model(
            scale_pos_weight
        )

        # ====================================================
        # TRAIN
        # ====================================================

        model.fit(
            X_train,
            y_train
        )

        # ====================================================
        # PREDICT
        # ====================================================

        probability = (
            model
            .predict_proba(
                X_val
            )[:, 1]
        )

        prediction = (
            probability >= 0.5
        ).astype(int)

        # ====================================================
        # METRICS
        # ====================================================

        if len(
            np.unique(y_val)
        ) == 2:

            roc_auc = roc_auc_score(
                y_val,
                probability
            )

            pr_auc = (
                average_precision_score(
                    y_val,
                    probability
                )
            )

        else:

            roc_auc = np.nan
            pr_auc = np.nan

        precision = precision_score(
            y_val,
            prediction,
            zero_division=0
        )

        recall = recall_score(
            y_val,
            prediction,
            zero_division=0
        )

        f1 = f1_score(
            y_val,
            prediction,
            zero_division=0
        )

        accuracy = accuracy_score(
            y_val,
            prediction
        )

        precision_5 = (
            precision_at_fraction(
                y_val,
                probability,
                0.05
            )
        )

        precision_10 = (
            precision_at_fraction(
                y_val,
                probability,
                0.10
            )
        )

        precision_20 = (
            precision_at_fraction(
                y_val,
                probability,
                0.20
            )
        )

        cm = confusion_matrix(
            y_val,
            prediction,
            labels=[0, 1]
        )

        # ====================================================
        # PRINT FOLD RESULTS
        # ====================================================

        print()
        print(
            "Confusion matrix:"
        )

        print(cm)

        print()
        print(
            f"ROC-AUC: "
            f"{roc_auc:.4f}"
        )

        print(
            f"PR-AUC: "
            f"{pr_auc:.4f}"
        )

        print(
            f"Precision: "
            f"{precision:.4f}"
        )

        print(
            f"Recall: "
            f"{recall:.4f}"
        )

        print(
            f"F1: "
            f"{f1:.4f}"
        )

        print(
            f"Accuracy: "
            f"{accuracy:.4f}"
        )

        print(
            f"Precision@5%: "
            f"{precision_5:.4f}"
        )

        print(
            f"Precision@10%: "
            f"{precision_10:.4f}"
        )

        print(
            f"Precision@20%: "
            f"{precision_20:.4f}"
        )

        # ====================================================
        # STORE METRICS
        # ====================================================

        fold_metrics.append(
            {
                "fold": fold,
                "train_size": len(train_idx),
                "validation_size": len(val_idx),
                "train_positive": train_positive,
                "validation_positive": int(
                    y_val.sum()
                ),
                "train_spatial_blocks": len(
                    train_groups
                ),
                "validation_spatial_blocks": len(
                    validation_groups
                ),
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
                "precision_at_5pct": precision_5,
                "precision_at_10pct": precision_10,
                "precision_at_20pct": precision_20,
            }
        )

        # ====================================================
        # STORE FOLD PREDICTIONS
        # ====================================================

        fold_result = (
            df.iloc[val_idx]
            [
                [
                    "latitude",
                    "longitude",
                    "label"
                ]
            ]
            .copy()
        )

        fold_result["fold"] = fold

        fold_result["probability"] = (
            probability
        )

        fold_result["prediction"] = (
            prediction
        )

        fold_predictions.append(
            fold_result
        )

    # ========================================================
    # SAVE CROSS-VALIDATION RESULTS
    # ========================================================

    metrics_df = pd.DataFrame(
        fold_metrics
    )

    predictions_df = pd.concat(
        fold_predictions,
        ignore_index=True
    )

    metrics_df.to_csv(
        METRICS_FILE,
        index=False
    )

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    # ========================================================
    # CROSS-VALIDATION SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print(
        "5-FOLD SPATIAL VALIDATION SUMMARY"
    )
    print("=" * 70)

    summary_metrics = [
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "precision_at_5pct",
        "precision_at_10pct",
        "precision_at_20pct",
    ]

    for metric in summary_metrics:

        mean_value = (
            metrics_df[metric]
            .mean()
        )

        std_value = (
            metrics_df[metric]
            .std()
        )

        print(
            f"{metric:25s}: "
            f"{mean_value:.4f} "
            f"± {std_value:.4f}"
        )

    # ========================================================
    # TRAIN FINAL PREPROCESSOR
    #
    # This is now allowed to use the complete balanced
    # dataset because the CV evaluation has already been
    # completed.
    # ========================================================

    print()
    print("=" * 70)
    print(
        "TRAINING FINAL MODEL"
    )
    print("=" * 70)

    final_preprocessor = (
        build_preprocessor(
            numeric_columns,
            categorical_columns
        )
    )

    X_final = (
        final_preprocessor
        .fit_transform(
            X_raw
        )
    )

    # ========================================================
    # FINAL FEATURE NAMES
    # ========================================================

    feature_names = (
        final_preprocessor
        .get_feature_names_out()
        .tolist()
    )

    print(
        f"Final transformed feature count: "
        f"{len(feature_names)}"
    )

    # ========================================================
    # CONVERT TRANSFORMED MATRIX TO DATAFRAME
    #
    # IMPORTANT:
    #
    # Passing a DataFrame to XGBoost makes sklearn/XGBoost
    # preserve feature names automatically.
    #
    # We DO NOT manually assign feature_names_in_.
    # ========================================================

    X_final_df = pd.DataFrame(
        X_final,
        columns=feature_names
    )

    # ========================================================
    # FINAL CLASS WEIGHT
    # ========================================================

    final_positive = int(
        np.sum(y == 1)
    )

    final_negative = int(
        np.sum(y == 0)
    )

    final_scale_pos_weight = (
        final_negative
        / max(
            final_positive,
            1
        )
    )

    print(
        f"Final scale_pos_weight: "
        f"{final_scale_pos_weight:.4f}"
    )

    # ========================================================
    # FINAL XGBOOST MODEL
    # ========================================================

    final_model = make_model(
        final_scale_pos_weight
    )

    final_model.fit(
        X_final_df,
        y
    )

    # ========================================================
    # VERIFY FEATURE NAMES
    # ========================================================

    if not hasattr(
        final_model,
        "feature_names_in_"
    ):

        raise RuntimeError(
            "XGBoost did not preserve feature names."
        )

    saved_feature_names = (
        final_model
        .feature_names_in_
        .tolist()
    )

    if saved_feature_names != feature_names:

        raise RuntimeError(
            "Feature-name mismatch between "
            "preprocessor and XGBoost model."
        )

    print(
        f"Final model feature count: "
        f"{len(final_model.feature_names_in_)}"
    )

    # ========================================================
    # SAVE FINAL MODEL
    # ========================================================

    joblib.dump(
        final_model,
        MODEL_FILE
    )

    # ========================================================
    # SAVE PREPROCESSOR
    # ========================================================

    joblib.dump(
        final_preprocessor,
        PREPROCESSOR_FILE
    )

    # ========================================================
    # SAVE FEATURE NAMES
    # ========================================================

    with open(
        FEATURE_COLUMNS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            feature_names,
            file,
            indent=2
        )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    importance = (
        final_model
        .feature_importances_
    )

    feature_importance_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importance
            }
        )
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    feature_importance_df.to_csv(
        FEATURE_IMPORTANCE_FILE,
        index=False
    )

    # ========================================================
    # FEATURE IMPORTANCE PLOT
    # ========================================================

    top_features = (
        feature_importance_df
        .head(25)
        .sort_values(
            "importance"
        )
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        top_features["feature"],
        top_features["importance"]
    )

    plt.xlabel(
        "XGBoost Feature Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Phase 4B - Top 25 XGBoost Features"
    )

    plt.tight_layout()

    plt.savefig(
        FEATURE_IMPORTANCE_PLOT,
        dpi=200
    )

    plt.close()

    # ========================================================
    # PRINT TOP FEATURES
    # ========================================================

    print()
    print(
        "Top 20 features:"
    )

    print(
        feature_importance_df
        .head(20)
        .to_string(
            index=False
        )
    )

    # ========================================================
    # FINAL FILE SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print(
        "FILES SAVED"
    )
    print("=" * 70)

    print(
        MODEL_FILE
    )

    print(
        PREPROCESSOR_FILE
    )

    print(
        FEATURE_COLUMNS_FILE
    )

    print(
        METRICS_FILE
    )

    print(
        PREDICTIONS_FILE
    )

    print(
        FEATURE_IMPORTANCE_FILE
    )

    print(
        FEATURE_IMPORTANCE_PLOT
    )

    print(
        BLOCK_AUDIT_FILE
    )

    print()
    print(
        "PHASE 4B.5 COMPLETE"
    )

    print(
        "PHASE 4B.6 COMPLETE"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()