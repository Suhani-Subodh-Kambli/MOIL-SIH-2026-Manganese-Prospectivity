# ============================================================
# MOIL SIH 2026
# PHASE 4B - LEAKAGE AUDIT
#
# Purpose:
#   Audit the Phase 4B dataset/model before retraining.
#
# Checks:
#   1. Distance_Mn leakage
#   2. Coordinate leakage
#   3. Duplicate samples
#   4. Near-duplicate spatial samples
#   5. Positive/negative spatial overlap
#   6. Spatial-fold separation
#   7. Feature-label correlation
#   8. Suspiciously predictive individual features
#   9. Geological category overlap
#
# DOES NOT retrain the model.
# ============================================================

import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(ROOT, "outputs")
MODEL_DIR = os.path.join(ROOT, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FIND DATASET
# ============================================================

candidate_files = [
    os.path.join(OUTPUT_DIR, "phase4b_features.csv"),
    os.path.join(OUTPUT_DIR, "phase4b_feature_dataset.csv"),
    os.path.join(OUTPUT_DIR, "MOIL_Phase4B_Features.csv"),
    os.path.join(OUTPUT_DIR, "phase4b_training_dataset.csv"),
    os.path.join(ROOT, "data", "MOIL_SIH_Phase4B_Features.csv"),
    os.path.join(ROOT, "data", "MOIL_SIH_Phase3_ML_Features.csv"),
    os.path.join(ROOT, "MOIL_SIH_Phase3_ML_Features.csv"),
]


def find_dataset():

    for path in candidate_files:
        if os.path.exists(path):
            return path

    # Search recursively for likely CSVs
    matches = []

    for base in [ROOT]:
        for root, dirs, files in os.walk(base):

            # Skip very large / irrelevant folders
            dirs[:] = [
                d for d in dirs
                if d not in [
                    ".git",
                    "__pycache__",
                    "node_modules",
                    ".venv",
                    "venv"
                ]
            ]

            for file in files:

                if not file.lower().endswith(".csv"):
                    continue

                lower = file.lower()

                if (
                    "phase4b" in lower
                    or "phase4" in lower
                    or "phase3" in lower
                    or "feature" in lower
                ):
                    matches.append(os.path.join(root, file))

    if matches:
        # Prefer phase4b
        phase4b = [
            x for x in matches
            if "phase4b" in os.path.basename(x).lower()
        ]

        if phase4b:
            return phase4b[0]

        return matches[0]

    return None


DATASET = find_dataset()


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("MOIL SIH 2026 - PHASE 4B LEAKAGE AUDIT")
print("=" * 75)


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 75)
print("STEP 1 - LOCATING DATASET")
print("=" * 75)

if DATASET is None:

    print("\nERROR: Could not automatically find the Phase 4B CSV.")

    print("\nExpected one of:")
    for x in candidate_files:
        print(" ", x)

    print(
        "\nIf your Phase 4B CSV has a different name/path, "
        "change DATASET manually near the top of this script."
    )

    raise SystemExit


print("Dataset:")
print(DATASET)


df = pd.read_csv(DATASET)

print("\nDataset shape:", df.shape)

print("\nColumns:")
for c in df.columns:
    print(" ", c)


# ============================================================
# BASIC LABEL AUDIT
# ============================================================

print("\n" + "=" * 75)
print("STEP 2 - LABEL AUDIT")
print("=" * 75)

if "label" not in df.columns:

    print("ERROR: 'label' column not found.")
    raise SystemExit

print("\nLabel distribution:")
print(df["label"].value_counts(dropna=False).sort_index())

print("\nLabel percentages:")
print(
    df["label"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)


# ============================================================
# DISTANCE_MN LEAKAGE
# ============================================================

print("\n" + "=" * 75)
print("STEP 3 - Distance_Mn LEAKAGE CHECK")
print("=" * 75)

if "Distance_Mn" in df.columns:

    print("\nWARNING:")
    print("Distance_Mn exists in the dataset.")

    print(
        "\nThis feature MUST NOT be used by the model because "
        "your labels were generated from manganese occurrence proximity."
    )

    print("\nDistance_Mn statistics by label:")

    print(
        df.groupby("label")["Distance_Mn"]
        .agg(["count", "min", "mean", "median", "max"])
    )

    # How well does Distance_Mn itself predict label?
    try:

        valid = df[["Distance_Mn", "label"]].dropna()

        # Smaller distance = higher probability of positive
        score = -valid["Distance_Mn"].values

        auc = roc_auc_score(
            valid["label"].values,
            score
        )

        print(
            f"\nAUC of Distance_Mn alone "
            f"(negative distance as score): {auc:.6f}"
        )

    except Exception as e:
        print("Could not calculate AUC:", e)

else:

    print("\nPASS:")
    print("Distance_Mn is not present.")


# ============================================================
# COORDINATE LEAKAGE
# ============================================================

print("\n" + "=" * 75)
print("STEP 4 - COORDINATE LEAKAGE CHECK")
print("=" * 75)

coordinate_columns = [
    "latitude",
    "longitude",
    "lat",
    "lon",
    "x",
    "y"
]

present_coordinates = [
    c for c in coordinate_columns
    if c in df.columns
]

if present_coordinates:

    print("\nCoordinate columns found:")
    print(present_coordinates)

    print(
        "\nIMPORTANT: coordinates should NOT be supplied "
        "as XGBoost predictor features."
    )

    for c in present_coordinates:

        try:

            auc = roc_auc_score(
                df["label"],
                df[c]
            )

            auc_inverse = roc_auc_score(
                df["label"],
                -df[c]
            )

            print(
                f"{c}: AUC={auc:.4f}, "
                f"inverse AUC={auc_inverse:.4f}"
            )

        except Exception:
            pass

else:

    print("\nNo coordinate columns found.")


# ============================================================
# DUPLICATE ROW CHECK
# ============================================================

print("\n" + "=" * 75)
print("STEP 5 - EXACT DUPLICATE CHECK")
print("=" * 75)

duplicate_count = df.duplicated().sum()

print("Exact duplicate rows:", duplicate_count)

if duplicate_count > 0:

    print(
        "\nWARNING: Exact duplicate rows exist."
        "\nDuplicates can cause train/validation leakage."
    )

else:

    print("\nPASS: No exact duplicate rows.")


# ============================================================
# DUPLICATE COORDINATE CHECK
# ============================================================

print("\n" + "=" * 75)
print("STEP 6 - DUPLICATE COORDINATE CHECK")
print("=" * 75)

if "latitude" in df.columns and "longitude" in df.columns:

    duplicate_coordinates = (
        df.duplicated(
            subset=["latitude", "longitude"]
        ).sum()
    )

    unique_coordinates = (
        df[["latitude", "longitude"]]
        .drop_duplicates()
        .shape[0]
    )

    print("Rows:", len(df))
    print("Unique coordinates:", unique_coordinates)
    print("Duplicate coordinate rows:", duplicate_coordinates)

    if duplicate_coordinates > 0:

        print(
            "\nWARNING: Multiple rows have identical coordinates."
        )

    else:

        print("\nPASS: No duplicate coordinates.")

else:

    print("Latitude/longitude not available.")


# ============================================================
# SPATIAL BLOCKS
# ============================================================

print("\n" + "=" * 75)
print("STEP 7 - SPATIAL BLOCK AUDIT")
print("=" * 75)

if "latitude" in df.columns and "longitude" in df.columns:

    BLOCK_SIZE = 0.10

    df["_spatial_x"] = np.floor(
        df["longitude"] / BLOCK_SIZE
    ).astype(int)

    df["_spatial_y"] = np.floor(
        df["latitude"] / BLOCK_SIZE
    ).astype(int)

    df["_spatial_block"] = (
        df["_spatial_x"].astype(str)
        + "_"
        + df["_spatial_y"].astype(str)
    )

    print(
        "\nSpatial block size:",
        BLOCK_SIZE,
        "degrees"
    )

    print(
        "Number of spatial blocks:",
        df["_spatial_block"].nunique()
    )

    block_stats = (
        df.groupby("_spatial_block")["label"]
        .agg(
            samples="count",
            positives="sum"
        )
    )

    block_stats["negative"] = (
        block_stats["samples"]
        - block_stats["positives"]
    )

    print("\nBlocks containing positives:")
    print(
        (block_stats["positives"] > 0).sum()
    )

    print(
        "Blocks containing only negatives:",
        (
            (block_stats["positives"] == 0)
            & (block_stats["negative"] > 0)
        ).sum()
    )

    print(
        "Blocks containing both classes:",
        (
            (block_stats["positives"] > 0)
            & (block_stats["negative"] > 0)
        ).sum()
    )

else:

    print(
        "\nCannot perform spatial block audit "
        "because coordinates are missing."
    )


# ============================================================
# POSITIVE/NEGATIVE SPATIAL PROXIMITY
# ============================================================

print("\n" + "=" * 75)
print("STEP 8 - POSITIVE / NEGATIVE SPATIAL OVERLAP")
print("=" * 75)

if "latitude" in df.columns and "longitude" in df.columns:

    positive = df[df["label"] == 1][
        ["latitude", "longitude"]
    ].to_numpy()

    negative = df[df["label"] == 0][
        ["latitude", "longitude"]
    ].to_numpy()

    print("Positive samples:", len(positive))
    print("Negative samples:", len(negative))

    if len(positive) > 0 and len(negative) > 0:

        # Approximate degree -> km conversion
        # Good enough for an audit around Balaghat.
        lat_scale = 111.0

        mean_lat = df["latitude"].mean()

        lon_scale = 111.0 * np.cos(
            np.radians(mean_lat)
        )

        nearest_negative_distances = []

        # Process in chunks to avoid memory explosion
        chunk_size = 500

        for start in range(
            0,
            len(positive),
            chunk_size
        ):

            p = positive[
                start:start + chunk_size
            ]

            dlat = (
                p[:, None, 0]
                - negative[None, :, 0]
            ) * lat_scale

            dlon = (
                p[:, None, 1]
                - negative[None, :, 1]
            ) * lon_scale

            dist = np.sqrt(
                dlat ** 2
                + dlon ** 2
            )

            nearest = dist.min(axis=1)

            nearest_negative_distances.extend(
                nearest.tolist()
            )

        nearest_negative_distances = np.array(
            nearest_negative_distances
        )

        print(
            "\nNearest negative sample distance "
            "from each positive:"
        )

        print(
            "Minimum :",
            round(
                nearest_negative_distances.min(),
                4
            ),
            "km"
        )

        print(
            "Median  :",
            round(
                np.median(
                    nearest_negative_distances
                ),
                4
            ),
            "km"
        )

        print(
            "Mean    :",
            round(
                nearest_negative_distances.mean(),
                4
            ),
            "km"
        )

        print(
            "Max     :",
            round(
                nearest_negative_distances.max(),
                4
            ),
            "km"
        )

        for threshold in [0.1, 0.25, 0.5, 1, 2, 3, 5]:

            count = (
                nearest_negative_distances
                <= threshold
            ).sum()

            percent = (
                count
                / len(nearest_negative_distances)
                * 100
            )

            print(
                f"Positive samples with "
                f"negative within {threshold} km: "
                f"{count} ({percent:.2f}%)"
            )

else:

    print("Coordinates unavailable.")


# ============================================================
# SPATIAL BLOCK LABEL PURITY
# ============================================================

print("\n" + "=" * 75)
print("STEP 9 - BLOCK LABEL PURITY")
print("=" * 75)

if "_spatial_block" in df.columns:

    purity = (
        df.groupby("_spatial_block")["label"]
        .agg(["count", "mean", "sum"])
    )

    purity["positive_fraction"] = purity["mean"]

    print("\nMost positive-heavy blocks:")

    print(
        purity.sort_values(
            "positive_fraction",
            ascending=False
        ).head(15)
    )

    print("\nMost negative-heavy blocks:")

    print(
        purity.sort_values(
            "positive_fraction",
            ascending=True
        ).head(15)
    )


# ============================================================
# NUMERIC FEATURE AUC AUDIT
# ============================================================

print("\n" + "=" * 75)
print("STEP 10 - INDIVIDUAL FEATURE PREDICTIVENESS")
print("=" * 75)

exclude = {
    "label",
    "Distance_Mn",
    "latitude",
    "longitude",
    "lat",
    "lon",
    "x",
    "y",
    "id",
    "system:index",
    ".geo",
    "_spatial_x",
    "_spatial_y",
    "_spatial_block"
}

numeric_columns = df.select_dtypes(
    include=[np.number]
).columns.tolist()

numeric_features = [
    c for c in numeric_columns
    if c not in exclude
]

feature_results = []

for feature in numeric_features:

    try:

        valid = df[
            [feature, "label"]
        ].dropna()

        if valid[feature].nunique() < 2:
            continue

        auc = roc_auc_score(
            valid["label"],
            valid[feature]
        )

        auc_inverse = roc_auc_score(
            valid["label"],
            -valid[feature]
        )

        best_auc = max(
            auc,
            auc_inverse
        )

        feature_results.append(
            {
                "feature": feature,
                "auc": auc,
                "inverse_auc": auc_inverse,
                "best_auc": best_auc
            }
        )

    except Exception:
        pass


feature_results_df = pd.DataFrame(
    feature_results
)

if len(feature_results_df) > 0:

    feature_results_df = (
        feature_results_df
        .sort_values(
            "best_auc",
            ascending=False
        )
    )

    print(
        "\nTop individual numeric features:"
    )

    print(
        feature_results_df.head(20).to_string(
            index=False
        )
    )

    print(
        "\nFeatures with individual AUC >= 0.90:"
    )

    suspicious = feature_results_df[
        feature_results_df["best_auc"] >= 0.90
    ]

    if len(suspicious) == 0:

        print("None.")

    else:

        print(
            suspicious.to_string(
                index=False
            )
        )

else:

    print("No numeric features available.")


# ============================================================
# CORRELATION WITH LABEL
# ============================================================

print("\n" + "=" * 75)
print("STEP 11 - NUMERIC CORRELATION AUDIT")
print("=" * 75)

if numeric_features:

    correlations = []

    for feature in numeric_features:

        try:

            corr = df[
                [feature, "label"]
            ].corr().iloc[0, 1]

            correlations.append(
                {
                    "feature": feature,
                    "label_correlation": corr
                }
            )

        except Exception:
            pass

    corr_df = pd.DataFrame(
        correlations
    )

    corr_df["abs_correlation"] = (
        corr_df["label_correlation"]
        .abs()
    )

    corr_df = corr_df.sort_values(
        "abs_correlation",
        ascending=False
    )

    print(
        corr_df.head(20).to_string(
            index=False
        )
    )


# ============================================================
# GEOLOGICAL CATEGORY AUDIT
# ============================================================

print("\n" + "=" * 75)
print("STEP 12 - GEOLOGICAL CATEGORY AUDIT")
print("=" * 75)

geo_columns = [
    "geo_group",
    "geo_lithology",
    "geo_formation",
    "geo_age",
    "geo_intrusive",
    "geo_supergroup"
]

present_geo = [
    c for c in geo_columns
    if c in df.columns
]

if present_geo:

    print(
        "\nGeological columns found:",
        present_geo
    )

    for col in present_geo:

        print("\n" + "-" * 70)
        print(col)

        # Positive distribution
        positive_dist = (
            df[df["label"] == 1][col]
            .fillna("UNKNOWN")
            .value_counts(
                normalize=True
            )
            .head(15)
            * 100
        )

        # Overall distribution
        overall_dist = (
            df[col]
            .fillna("UNKNOWN")
            .value_counts(
                normalize=True
            )
            .head(15)
            * 100
        )

        result = pd.DataFrame(
            {
                "overall_%": overall_dist,
                "positive_%": positive_dist
            }
        )

        result["enrichment"] = (
            result["positive_%"]
            / result["overall_%"]
        )

        print(
            result.sort_values(
                "enrichment",
                ascending=False
            ).head(15).to_string()
        )

else:

    print("No geological columns found.")


# ============================================================
# MODEL AUDIT
# ============================================================

print("\n" + "=" * 75)
print("STEP 13 - SAVED MODEL AUDIT")
print("=" * 75)

model_candidates = [
    os.path.join(
        MODEL_DIR,
        "moil_manganese_prospectivity_xgb_phase4b.joblib"
    ),
    os.path.join(
        MODEL_DIR,
        "moil_manganese_prospectivity_xgb.joblib"
    )
]

model_path = None

for path in model_candidates:

    if os.path.exists(path):
        model_path = path
        break


if model_path is None:

    print(
        "\nNo saved XGBoost model found."
    )

else:

    print(
        "\nModel:",
        model_path
    )

    try:

        import joblib

        model = joblib.load(
            model_path
        )

        print(
            "Model type:",
            type(model)
        )

        if hasattr(
            model,
            "n_features_in_"
        ):

            print(
                "Model feature count:",
                model.n_features_in_
            )

        if hasattr(
            model,
            "feature_names_in_"
        ):

            model_features = list(
                model.feature_names_in_
            )

            print(
                "\nModel feature names:"
            )

            for feature in model_features:
                print(" ", feature)

            forbidden = [
                f
                for f in model_features
                if f in [
                    "Distance_Mn",
                    "latitude",
                    "longitude",
                    "lat",
                    "lon",
                    "x",
                    "y"
                ]
            ]

            print(
                "\nForbidden/leakage-prone features "
                "present in model:"
            )

            if forbidden:
                for f in forbidden:
                    print(" WARNING:", f)

            else:
                print(" PASS - none")

            # Check features against dataset
            missing = [
                f
                for f in model_features
                if f not in df.columns
            ]

            print(
                "\nModel features missing from dataset:",
                len(missing)
            )

            if missing:
                print(missing)

    except Exception as e:

        print(
            "Could not load model:",
            repr(e)
        )


# ============================================================
# TRAIN/VALIDATION FILE AUDIT
# ============================================================

print("\n" + "=" * 75)
print("STEP 14 - EXISTING VALIDATION FILES")
print("=" * 75)

validation_candidates = [
    "phase4b_spatial_metrics.csv",
    "phase4b_predictions.csv",
    "phase4b_test_predictions.csv",
    "test_predictions.csv"
]

for filename in validation_candidates:

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    if os.path.exists(path):

        print(
            "\nFound:",
            path
        )

        try:

            temp = pd.read_csv(path)

            print(
                "Shape:",
                temp.shape
            )

            print(
                "Columns:",
                list(temp.columns)
            )

        except Exception as e:

            print(
                "Could not read:",
                e
            )


# ============================================================
# SAVE FEATURE AUDIT
# ============================================================

feature_audit_path = os.path.join(
    OUTPUT_DIR,
    "phase4b_feature_leakage_audit.csv"
)

if len(feature_results_df) > 0:

    feature_results_df.to_csv(
        feature_audit_path,
        index=False
    )

    print(
        "\nSaved:",
        feature_audit_path
    )


# ============================================================
# SAVE BLOCK AUDIT
# ============================================================

if "_spatial_block" in df.columns:

    block_output = (
        df.groupby("_spatial_block")["label"]
        .agg(
            sample_count="count",
            positive_count="sum"
        )
        .reset_index()
    )

    block_output["negative_count"] = (
        block_output["sample_count"]
        - block_output["positive_count"]
    )

    block_output["positive_fraction"] = (
        block_output["positive_count"]
        / block_output["sample_count"]
    )

    block_path = os.path.join(
        OUTPUT_DIR,
        "phase4b_spatial_block_audit.csv"
    )

    block_output.to_csv(
        block_path,
        index=False
    )

    print(
        "Saved:",
        block_path
    )


# ============================================================
# FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 75)
print("FINAL LEAKAGE AUDIT CHECKLIST")
print("=" * 75)

issues = []

if "Distance_Mn" in df.columns:
    issues.append(
        "Distance_Mn exists in dataset - must remain excluded from model."
    )

if duplicate_count > 0:
    issues.append(
        f"{duplicate_count} exact duplicate rows found."
    )

if (
    "latitude" in df.columns
    and "longitude" in df.columns
    and duplicate_coordinates > 0
):
    issues.append(
        f"{duplicate_coordinates} duplicate coordinate rows found."
    )

if len(feature_results_df) > 0:

    suspicious = feature_results_df[
        feature_results_df["best_auc"] >= 0.90
    ]

    if len(suspicious) > 0:

        issues.append(
            "At least one individual feature has AUC >= 0.90."
        )


print()

if issues:

    print("POTENTIAL ISSUES FOUND:")
    print()

    for i, issue in enumerate(
        issues,
        start=1
    ):

        print(
            f"{i}. {issue}"
        )

else:

    print(
        "No obvious leakage issue detected "
        "by the automated checks."
    )


print("\n" + "=" * 75)
print("AUDIT COMPLETE")
print("=" * 75)

print(
    "\nDO NOT retrain yet."
)

print(
    "Review this output first, especially:"
)

print(
    "1. Distance_Mn"
)

print(
    "2. individual feature AUC"
)

print(
    "3. duplicate coordinates"
)

print(
    "4. positive/negative spatial distances"
)

print(
    "5. model feature names"
)

print(
    "6. spatial block statistics"
)