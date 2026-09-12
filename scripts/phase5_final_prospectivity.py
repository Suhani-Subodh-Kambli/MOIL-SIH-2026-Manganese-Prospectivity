# ============================================================
# MOIL SIH 2026
# PHASE 5 - FINAL MANGANESE PROSPECTIVITY MAPPING
#
# FINAL CORRECTED VERSION
#
# Uses the EXACT Phase 4B preprocessing pipeline and
# the NEW 117-feature Phase 4B XGBoost model.
#
# Architecture:
#
# GEE Sentinel-2/Sentinel-1/SRTM grid
#          +
#      NGDR geology
#          ↓
# Phase 4B engineered features
#          ↓
# Saved Phase 4B preprocessor
#          ↓
# 117-feature XGBoost
#          ↓
# Prospectivity score
#          ↓
# Target cells / zones
#
# IMPORTANT:
# - No Distance_Mn is used.
# - No latitude/longitude are model features.
# - No sample-neighbourhood terrain features are recreated.
# - geo_stratigraphy remains categorical.
# - The saved Phase 4B preprocessor is the source of truth.
# ============================================================


import os
import json
import warnings

import numpy as np
import pandas as pd
import joblib

from shapely.geometry import shape, Point, mapping
from shapely.strtree import STRtree
from shapely.ops import unary_union

from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS
# ============================================================

BASE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)


DATA_DIR = os.path.join(
    BASE,
    "data"
)


MODEL_DIR = os.path.join(
    BASE,
    "models"
)


OUTPUT_DIR = os.path.join(
    BASE,
    "outputs"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# Input files
# ------------------------------------------------------------

GRID_FILE = os.path.join(
    DATA_DIR,
    "MOIL_Balaghat_Phase5_Final_Feature_Grid.csv"
)


GEOLOGY_FILE = os.path.join(
    DATA_DIR,
    "geology",
    "balaghat",
    "balaghat_lithology.geojsonl"
)


# ------------------------------------------------------------
# Phase 4B model files
# ------------------------------------------------------------

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "moil_manganese_prospectivity_xgb_phase4b.joblib"
)


PREPROCESSOR_FILE = os.path.join(
    MODEL_DIR,
    "phase4b_preprocessor.joblib"
)


FEATURE_FILE = os.path.join(
    MODEL_DIR,
    "phase4b_feature_columns.json"
)


# ============================================================
# 2. OUTPUT FILES
# ============================================================

FEATURE_MATRIX_FILE = os.path.join(
    OUTPUT_DIR,
    "phase5_model_feature_matrix.csv"
)


PREDICTION_FILE = os.path.join(
    OUTPUT_DIR,
    "phase5_prospectivity_grid.csv"
)


TARGET_CELLS_FILE = os.path.join(
    OUTPUT_DIR,
    "phase5_top_target_cells.csv"
)


TARGET_ZONES_FILE = os.path.join(
    OUTPUT_DIR,
    "phase5_target_zones.csv"
)


TARGET_CELLS_GEOJSON = os.path.join(
    OUTPUT_DIR,
    "phase5_target_cells.geojson"
)


TARGET_ZONES_GEOJSON = os.path.join(
    OUTPUT_DIR,
    "phase5_target_zones.geojson"
)


MAP_FILE = os.path.join(
    OUTPUT_DIR,
    "phase5_prospectivity_map.png"
)


SCORE_DISTRIBUTION_FILE = os.path.join(
    OUTPUT_DIR,
    "phase5_score_distribution.png"
)


SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "phase5_summary.csv"
)


# ============================================================
# 3. SETTINGS
# ============================================================

# Top 10% of cells are treated as exploration targets.
TARGET_SCORE_PERCENTILE = 90


# Target-cell clustering radius.
# Approximately 1.5 km.
CLUSTER_EPS_KM = 1.5


CLUSTER_MIN_SAMPLES = 3


# ============================================================
# 4. HEADER
# ============================================================

print()
print("=" * 75)
print("MOIL SIH 2026")
print("PHASE 5 - FINAL MANGANESE PROSPECTIVITY MAPPING")
print("=" * 75)


# ============================================================
# 5. LOAD MODEL
# ============================================================

print()
print("-" * 75)
print("STEP 1 - Loading Phase 4B model")
print("-" * 75)


if not os.path.exists(
    MODEL_FILE
):

    raise FileNotFoundError(
        "\nMissing Phase 4B model:\n"
        + MODEL_FILE
    )


if not os.path.exists(
    PREPROCESSOR_FILE
):

    raise FileNotFoundError(
        "\nMissing Phase 4B preprocessor:\n"
        + PREPROCESSOR_FILE
    )


model = joblib.load(
    MODEL_FILE
)


preprocessor = joblib.load(
    PREPROCESSOR_FILE
)


print(
    "Model:",
    MODEL_FILE
)


print(
    "Preprocessor:",
    PREPROCESSOR_FILE
)


# ============================================================
# 6. MODEL FEATURE CHECK
# ============================================================

if not hasattr(
    model,
    "feature_names_in_"
):

    raise RuntimeError(
        "Saved XGBoost model does not contain "
        "feature_names_in_."
    )


EXPECTED_FEATURES = list(
    model.feature_names_in_
)


print()
print(
    "Model feature count:",
    len(EXPECTED_FEATURES)
)


if len(EXPECTED_FEATURES) != 117:

    raise RuntimeError(
        "This Phase 5 pipeline expects the new "
        "117-feature Phase 4B model.\n"
        f"Found: {len(EXPECTED_FEATURES)}"
    )


# ============================================================
# 7. LOAD FEATURE COLUMN FILE
# ============================================================

if not os.path.exists(
    FEATURE_FILE
):

    raise FileNotFoundError(
        "\nMissing:\n"
        + FEATURE_FILE
    )


with open(
    FEATURE_FILE,
    "r",
    encoding="utf-8"
) as f:

    saved_feature_columns = json.load(
        f
    )


if list(saved_feature_columns) != EXPECTED_FEATURES:

    raise RuntimeError(
        "phase4b_feature_columns.json does not "
        "match the saved XGBoost model."
    )


print(
    "Feature-column verification: PASS"
)


# ============================================================
# 8. LOAD GEE GRID
# ============================================================

print()
print("-" * 75)
print("STEP 2 - Loading GEE Phase 5 grid")
print("-" * 75)


if not os.path.exists(
    GRID_FILE
):

    raise FileNotFoundError(
        "\nMissing GEE grid:\n"
        + GRID_FILE
    )


grid = pd.read_csv(
    GRID_FILE
)


print(
    "Grid shape:",
    grid.shape
)


# ============================================================
# 9. COORDINATE VALIDATION
# ============================================================

required_coordinates = [
    "longitude",
    "latitude"
]


for col in required_coordinates:

    if col not in grid.columns:

        raise RuntimeError(
            f"Missing required column: {col}"
        )


grid["longitude"] = pd.to_numeric(
    grid["longitude"],
    errors="coerce"
)


grid["latitude"] = pd.to_numeric(
    grid["latitude"],
    errors="coerce"
)


valid = (
    grid["longitude"].notna()
    &
    grid["latitude"].notna()
)


if not valid.all():

    removed = int(
        (~valid).sum()
    )

    print(
        f"Removing {removed} rows "
        "with invalid coordinates."
    )

    grid = grid.loc[
        valid
    ].copy()


grid = grid.reset_index(
    drop=True
)


# ============================================================
# 10. LOAD NGDR GEOLOGY
# ============================================================

print()
print("-" * 75)
print("STEP 3 - Loading NGDR Balaghat geology")
print("-" * 75)


if not os.path.exists(
    GEOLOGY_FILE
):

    raise FileNotFoundError(
        "\nMissing geology file:\n"
        + GEOLOGY_FILE
    )


geometries = []
properties = []


invalid_geometries = 0


with open(
    GEOLOGY_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        try:

            obj = json.loads(
                line
            )


            if (
                "geometry"
                not in obj
            ):

                continue


            geom = shape(
                obj["geometry"]
            )


            if geom.is_empty:

                continue


            if not geom.is_valid:

                invalid_geometries += 1

                try:

                    geom = geom.buffer(
                        0
                    )

                except Exception:

                    continue


            if geom.is_empty:

                continue


            geometries.append(
                geom
            )


            properties.append(
                obj.get(
                    "properties",
                    {}
                )
            )


        except Exception:

            continue


print(
    "Geology polygons:",
    len(geometries)
)


print(
    "Invalid geometries repaired/skipped:",
    invalid_geometries
)


if len(geometries) == 0:

    raise RuntimeError(
        "No geology polygons were loaded."
    )


geo_tree = STRtree(
    geometries
)


# ============================================================
# 11. SHAPELY TREE INDEX HELPER
# ============================================================

def candidate_indices(
    tree,
    geometries_list,
    query_geometry
):

    candidates = tree.query(
        query_geometry
    )


    indices = []


    for candidate in candidates:

        if isinstance(
            candidate,
            (int, np.integer)
        ):

            indices.append(
                int(candidate)
            )

        else:

            # Compatibility fallback for
            # Shapely versions returning geometries.
            try:

                idx = geometries_list.index(
                    candidate
                )

                indices.append(
                    idx
                )

            except ValueError:

                continue


    return indices


# ============================================================
# 12. GEOLOGY LOOKUP
# ============================================================

def get_geology_properties(
    longitude,
    latitude
):

    point = Point(
        float(longitude),
        float(latitude)
    )


    candidates = candidate_indices(
        geo_tree,
        geometries,
        point
    )


    for idx in candidates:

        geom = geometries[
            idx
        ]


        if geom.intersects(
            point
        ):

            return properties[
                idx
            ]


    return None


# ============================================================
# 13. ASSIGN GEOLOGY TO GRID
# ============================================================

print()
print("-" * 75)
print("STEP 4 - Assigning NGDR geology")
print("-" * 75)


geo_rows = []


for i, row in grid.iterrows():

    if i % 1000 == 0:

        print(
            f"Processed {i:,} / {len(grid):,}"
        )


    props = get_geology_properties(
        row["longitude"],
        row["latitude"]
    )


    if props is None:

        props = {}


    geo_rows.append({

        "geo_age":
            str(
                props.get(
                    "age",
                    "UNKNOWN"
                )
            ),

        "geo_supergroup":
            str(
                props.get(
                    "supergroup",
                    "UNKNOWN"
                )
            ),

        "geo_group":
            str(
                props.get(
                    "group_name",
                    "UNKNOWN"
                )
            ),

        "geo_formation":
            str(
                props.get(
                    "formation",
                    "UNKNOWN"
                )
            ),

        "geo_lithology":
            str(
                props.get(
                    "lithologic",
                    "UNKNOWN"
                )
            ),

        "geo_intrusive":
            str(
                props.get(
                    "intrusive",
                    "UNKNOWN"
                )
            ),

        "geo_stratigraphy":
            str(
                props.get(
                    "stratigraphy_new",
                    "UNKNOWN"
                )
            )

    })


geo_df = pd.DataFrame(
    geo_rows
)


grid = pd.concat(
    [
        grid.reset_index(
            drop=True
        ),
        geo_df
    ],
    axis=1
)


# ============================================================
# 14. CLEAN GEOLOGY
# ============================================================

geo_columns = [

    "geo_age",
    "geo_supergroup",
    "geo_group",
    "geo_formation",
    "geo_lithology",
    "geo_intrusive",
    "geo_stratigraphy"

]


for col in geo_columns:

    grid[col] = (

        grid[col]

        .fillna(
            "UNKNOWN"
        )

        .astype(str)

        .replace(
            {
                "None":
                    "UNKNOWN",

                "nan":
                    "UNKNOWN",

                "":
                    "UNKNOWN"
            }
        )

    )


print(
    "\nGeology assignment complete."
)


# ============================================================
# 15. RECREATE PHASE 4B SPECTRAL FEATURES
# ============================================================

print()
print("-" * 75)
print("STEP 5 - Recreating Phase 4B spectral features")
print("-" * 75)


# ------------------------------------------------------------
# NDMI
# ------------------------------------------------------------

grid["NDMI"] = (

    (
        grid["B8"]
        -
        grid["B11"]
    )

    /

    (
        grid["B8"]
        +
        grid["B11"]
        +
        1e-6
    )

)


# ------------------------------------------------------------
# NBR
# ------------------------------------------------------------

grid["NBR"] = (

    (
        grid["B8"]
        -
        grid["B12"]
    )

    /

    (
        grid["B8"]
        +
        grid["B12"]
        +
        1e-6
    )

)


# ------------------------------------------------------------
# NDRE
# ------------------------------------------------------------

grid["NDRE"] = (

    (
        grid["B8A"]
        -
        grid["B5"]
    )

    /

    (
        grid["B8A"]
        +
        grid["B5"]
        +
        1e-6
    )

)


# ------------------------------------------------------------
# Iron oxide index
# ------------------------------------------------------------

grid[
    "Iron_Oxide_Index"
] = (

    grid["B4"]
    /

    (
        grid["B2"]
        +
        1e-6
    )

)


# ------------------------------------------------------------
# Clay alteration index
# ------------------------------------------------------------

grid[
    "Clay_Alteration_Index"
] = (

    grid["B11"]
    /

    (
        grid["B12"]
        +
        1e-6
    )

)


# ------------------------------------------------------------
# Ferrous index
# ------------------------------------------------------------

grid[
    "Ferrous_Index"
] = (

    grid["B12"]
    /

    (
        grid["B11"]
        +
        1e-6
    )

)


# ------------------------------------------------------------
# SWIR / Red
# ------------------------------------------------------------

grid[
    "SWIR_Red_Ratio"
] = (

    grid["B11"]
    /

    (
        grid["B4"]
        +
        1e-6
    )

)


# ------------------------------------------------------------
# SWIR / Green
# ------------------------------------------------------------

grid[
    "SWIR_Green_Ratio"
] = (

    grid["B11"]
    /

    (
        grid["B3"]
        +
        1e-6
    )

)


# ------------------------------------------------------------
# NIR / SWIR2
# ------------------------------------------------------------

grid[
    "NIR_SWIR2_Ratio"
] = (

    grid["B8"]
    /

    (
        grid["B12"]
        +
        1e-6
    )

)


# ------------------------------------------------------------
# BSI
# ------------------------------------------------------------

grid["BSI"] = (

    (
        grid["B11"]
        +
        grid["B4"]
        -
        grid["B8"]
        -
        grid["B2"]
    )

    /

    (
        grid["B11"]
        +
        grid["B4"]
        +
        grid["B8"]
        +
        grid["B2"]
        +
        1e-6
    )

)


# ------------------------------------------------------------
# B11/B12 normalized difference
# ------------------------------------------------------------

grid[
    "B11_B12_NormDiff"
] = (

    (
        grid["B11"]
        -
        grid["B12"]
    )

    /

    (
        grid["B11"]
        +
        grid["B12"]
        +
        1e-6
    )

)


# ------------------------------------------------------------
# B8/B12 normalized difference
# ------------------------------------------------------------

grid[
    "B8_B12_NormDiff"
] = (

    (
        grid["B8"]
        -
        grid["B12"]
    )

    /

    (
        grid["B8"]
        +
        grid["B12"]
        +
        1e-6
    )

)


# ------------------------------------------------------------
# B4/B2 normalized difference
# ------------------------------------------------------------

grid[
    "B4_B2_NormDiff"
] = (

    (
        grid["B4"]
        -
        grid["B2"]
    )

    /

    (
        grid["B4"]
        +
        grid["B2"]
        +
        1e-6
    )

)


print(
    "Spectral features recreated."
)


# ============================================================
# 16. GEOLOGICAL PROXIES
# ============================================================

print()
print("-" * 75)
print("STEP 6 - Creating geological proxies")
print("-" * 75)


# ------------------------------------------------------------
# Sausar proxy
# ------------------------------------------------------------

grid[
    "Sausar_Group_Proxy"
] = (

    grid[
        "geo_group"
    ]

    .str.upper()

    .str.contains(
        "SAUSAR",
        na=False
    )

    .astype(int)

)


# ------------------------------------------------------------
# Metamorphic host proxy
# ------------------------------------------------------------

metamorphic_keywords = (

    "SCHIST",
    "GNEISS",
    "AMPHIBOLITE",
    "QUARTZITE",
    "PHYLLITE",
    "MARBLE",
    "GRANULITE"

)


def metamorphic_host(
    value
):

    value = str(
        value
    ).upper()


    return int(
        any(
            keyword in value
            for keyword
            in metamorphic_keywords
        )
    )


grid[
    "Metamorphic_Host_Proxy"
] = grid[
    "geo_lithology"
].apply(
    metamorphic_host
)


# ------------------------------------------------------------
# Unknown geology
# ------------------------------------------------------------

grid[
    "Geology_Unknown"
] = (

    grid[
        "geo_group"
    ]

    .str.upper()

    .eq(
        "UNKNOWN"
    )

    .astype(int)

)


print(
    "Geological proxies created."
)


# ============================================================
# 17. ASPECT SIN/COS
# ============================================================

aspect_numeric = pd.to_numeric(
    grid["Aspect"],
    errors="coerce"
)


aspect_numeric = aspect_numeric.fillna(
    aspect_numeric.median()
)


aspect_rad = np.radians(
    aspect_numeric
)


grid[
    "Aspect_Sin"
] = np.sin(
    aspect_rad
)


grid[
    "Aspect_Cos"
] = np.cos(
    aspect_rad
)


# ============================================================
# 18. GEOLOGICAL BOUNDARY FEATURES
# ============================================================

print()
print("-" * 75)
print("STEP 7 - Geological boundary features")
print("-" * 75)


boundary_geometries = []


for geom in geometries:

    try:

        boundary = geom.boundary


        if not boundary.is_empty:

            boundary_geometries.append(
                boundary
            )

    except Exception:

        continue


boundary_tree = STRtree(
    boundary_geometries
)


print(
    "Geological boundaries:",
    len(boundary_geometries)
)


boundary_distance = []
boundary_density_1km = []
boundary_density_3km = []


for i, row in grid.iterrows():

    if i % 2000 == 0:

        print(
            f"Boundary processing: "
            f"{i:,} / {len(grid):,}"
        )


    point = Point(
        float(
            row["longitude"]
        ),
        float(
            row["latitude"]
        )
    )


    # --------------------------------------------------------
    # Approximate degree distances
    # --------------------------------------------------------

    radius_1km = (
        1.0 / 111.32
    )


    radius_3km = (
        3.0 / 111.32
    )


    candidates_1km = candidate_indices(
        boundary_tree,
        boundary_geometries,
        point.buffer(
            radius_1km
        )
    )


    candidates_3km = candidate_indices(
        boundary_tree,
        boundary_geometries,
        point.buffer(
            radius_3km
        )
    )


    # --------------------------------------------------------
    # Boundary distance
    # --------------------------------------------------------

    if len(
        candidates_3km
    ) == 0:

        distance_km = 3.0

    else:

        distances = []


        for idx in candidates_3km:

            geom = boundary_geometries[
                idx
            ]


            distance_km = (

                point.distance(
                    geom
                )
                *
                111.32

            )


            distances.append(
                distance_km
            )


        distance_km = min(
            distances
        )


    boundary_distance.append(
        distance_km
    )


    boundary_density_1km.append(
        len(
            candidates_1km
        )
    )


    boundary_density_3km.append(
        len(
            candidates_3km
        )
    )


grid[
    "Geo_Boundary_Distance_km"
] = boundary_distance


grid[
    "Geo_Boundary_Density_1km"
] = boundary_density_1km


grid[
    "Geo_Boundary_Density_3km"
] = boundary_density_3km


print(
    "Boundary features created."
)


# ============================================================
# 19. GEOLOGICAL DIVERSITY
# ============================================================

print()
print("-" * 75)
print("STEP 8 - 3 km geological diversity")
print("-" * 75)


lithology_diversity = []
formation_diversity = []


for i, row in grid.iterrows():

    if i % 2000 == 0:

        print(
            f"Diversity processing: "
            f"{i:,} / {len(grid):,}"
        )


    point = Point(
        float(
            row["longitude"]
        ),
        float(
            row["latitude"]
        )
    )


    search_radius = (
        3.0 / 111.32
    )


    search_area = point.buffer(
        search_radius
    )


    candidates = candidate_indices(
        geo_tree,
        geometries,
        search_area
    )


    lithologies = set()
    formations = set()


    for idx in candidates:

        geom = geometries[
            idx
        ]


        if not geom.intersects(
            search_area
        ):

            continue


        props = properties[
            idx
        ]


        lithology = str(
            props.get(
                "lithologic",
                "UNKNOWN"
            )
        )


        formation = str(
            props.get(
                "formation",
                "UNKNOWN"
            )
        )


        if lithology != "UNKNOWN":

            lithologies.add(
                lithology
            )


        if formation != "UNKNOWN":

            formations.add(
                formation
            )


    lithology_diversity.append(
        len(
            lithologies
        )
    )


    formation_diversity.append(
        len(
            formations
        )
    )


grid[
    "Lithology_Diversity_3km"
] = lithology_diversity


grid[
    "Formation_Diversity_3km"
] = formation_diversity


print(
    "Geological diversity features created."
)


# ============================================================
# 20. IMPORTANT: DO NOT RECREATE OLD TERRAIN FEATURES
# ============================================================

print()
print("-" * 75)
print("STEP 9 - Phase 4B feature-space verification")
print("-" * 75)


old_sample_terrain_features = [

    "Local_Elevation_STD",
    "TPI_Local",
    "Local_Slope_STD",
    "Local_Aspect_Dispersion"

]


print(
    "The following sample-derived terrain features "
    "are intentionally NOT recreated:"
)


for feature in old_sample_terrain_features:

    print(
        "  -",
        feature
    )


# ============================================================
# 21. RAW MODEL INPUT
# ============================================================

print()
print("-" * 75)
print("STEP 10 - Preparing raw Phase 4B model input")
print("-" * 75)


# These are exactly the seven categorical columns
# used during Phase 4B training.

categorical_columns = [

    "geo_age",

    "geo_supergroup",

    "geo_group",

    "geo_formation",

    "geo_lithology",

    "geo_intrusive",

    "geo_stratigraphy"

]


# Columns that must never enter the model.

excluded_columns = [

    "label",

    "Distance_Mn",

    "longitude",

    "latitude",

    "system:index",

    ".geo",

    "x_km",

    "y_km"

]


# ------------------------------------------------------------
# Remove old prediction columns if this grid has previously
# been processed.
# ------------------------------------------------------------

old_prediction_columns = [

    "Model_Probability",

    "Prospectivity_Score",

    "prospectivity_probability",

    "prospectivity_score",

    "Priority",

    "Target_Zone_ID",

    "Target_Cell_Rank",

    "Target_Level"

]


X_raw = grid.drop(
    columns=excluded_columns,
    errors="ignore"
).copy()


X_raw = X_raw.drop(
    columns=old_prediction_columns,
    errors="ignore"
)


# ============================================================
# 22. CLEAN CATEGORICAL DATA
# ============================================================

for col in categorical_columns:

    if col in X_raw.columns:

        X_raw[col] = (

            X_raw[col]

            .fillna(
                "UNKNOWN"
            )

            .astype(str)

            .replace(
                {
                    "None":
                        "UNKNOWN",

                    "nan":
                        "UNKNOWN",

                    "":
                        "UNKNOWN"
                }
            )

        )


# ============================================================
# 23. CLEAN NUMERIC DATA
# ============================================================

for col in X_raw.columns:

    if col not in categorical_columns:

        X_raw[col] = pd.to_numeric(
            X_raw[col],
            errors="coerce"
        )


X_raw = X_raw.replace(
    [
        np.inf,
        -np.inf
    ],
    np.nan
)


# ============================================================
# 24. USE PREPROCESSOR'S RAW FEATURE LIST
# ============================================================

if not hasattr(
    preprocessor,
    "feature_names_in_"
):

    raise RuntimeError(
        "Saved Phase 4B preprocessor does not expose "
        "feature_names_in_."
    )


RAW_FEATURES = list(
    preprocessor.feature_names_in_
)


print(
    "Raw features expected by preprocessor:",
    len(RAW_FEATURES)
)


missing_raw_features = [

    col

    for col in RAW_FEATURES

    if col not in X_raw.columns

]


if missing_raw_features:

    print()
    print(
        "Missing raw features:"
    )


    for col in missing_raw_features:

        print(
            "  -",
            col
        )


    raise RuntimeError(
        "Phase 5 grid could not recreate all "
        "raw Phase 4B features."
    )


# Keep EXACT raw feature order.
X_raw = X_raw[
    RAW_FEATURES
].copy()


print(
    "Raw feature alignment: PASS"
)


# ============================================================
# 25. APPLY SAVED PREPROCESSOR
# ============================================================

print()
print("-" * 75)
print("STEP 11 - Applying saved Phase 4B preprocessor")
print("-" * 75)


X_transformed = preprocessor.transform(
    X_raw
)


X_transformed = np.asarray(
    X_transformed,
    dtype=float
)


print(
    "Transformed matrix:",
    X_transformed.shape
)


# ============================================================
# 26. CONVERT TO MODEL DATAFRAME
# ============================================================

if X_transformed.shape[1] != len(
    EXPECTED_FEATURES
):

    raise RuntimeError(

        "\nFEATURE COUNT MISMATCH\n"

        f"Preprocessor produced: "
        f"{X_transformed.shape[1]}\n"

        f"Model expects: "
        f"{len(EXPECTED_FEATURES)}"

    )


X = pd.DataFrame(
    X_transformed,
    columns=EXPECTED_FEATURES
)


# ============================================================
# 27. FINAL MODEL FEATURE VERIFICATION
# ============================================================

if list(
    X.columns
) != EXPECTED_FEATURES:

    raise RuntimeError(
        "Final model feature order does not "
        "match XGBoost feature order."
    )


print()
print(
    "Final model matrix:",
    X.shape
)


print(
    "Expected features:",
    len(EXPECTED_FEATURES)
)


print(
    "Missing features: 0"
)


print(
    "Extra features: 0"
)


print(
    "Feature order verification: PASS"
)


# ============================================================
# 28. SAVE MODEL MATRIX
# ============================================================

X.to_csv(
    FEATURE_MATRIX_FILE,
    index=False
)


print()
print(
    "Saved:",
    FEATURE_MATRIX_FILE
)


# ============================================================
# 29. PREDICT
# ============================================================

print()
print("-" * 75)
print("STEP 12 - Generating XGBoost prospectivity predictions")
print("-" * 75)


probability = model.predict_proba(
    X
)[:, 1]


grid[
    "Model_Probability"
] = probability


print(
    "Prediction complete."
)


print(
    "Probability range:",
    f"{probability.min():.6f}",
    "to",
    f"{probability.max():.6f}"
)


print(
    "Mean probability:",
    f"{probability.mean():.6f}"
)


# ============================================================
# 30. RELATIVE PROSPECTIVITY SCORE
# ============================================================

# IMPORTANT:
#
# This is a relative ranking score.
#
# 100 = highest-ranked grid cells
# 0   = lowest-ranked grid cells
#
# It is NOT:
# - manganese concentration
# - ore grade
# - proven reserve
# - economic extraction probability
#
# It is an AI-assisted exploration prioritization score.

grid[
    "Prospectivity_Score"
] = (

    grid[
        "Model_Probability"
    ]

    .rank(
        method="average",
        pct=True
    )

    *
    100.0

)


# ============================================================
# 31. PRIORITY CLASS
# ============================================================

def classify_priority(
    score
):

    if score >= 98:

        return "VERY_HIGH"

    elif score >= 95:

        return "HIGH"

    elif score >= 90:

        return "MODERATE_HIGH"

    elif score >= 75:

        return "MODERATE"

    else:

        return "LOW"


grid[
    "Priority"
] = grid[
    "Prospectivity_Score"
].apply(
    classify_priority
)


grid[
    "Geology_Context"
] = np.where(

    grid[
        "Geology_Unknown"
    ] == 1,

    "UNKNOWN_GEOLOGY",

    "KNOWN_GEOLOGY"

)


# ============================================================
# 32. TARGET CELLS
# ============================================================

target_threshold = np.percentile(

    grid[
        "Prospectivity_Score"
    ],

    TARGET_SCORE_PERCENTILE

)


grid[
    "Target_Cell"
] = (

    grid[
        "Prospectivity_Score"
    ]
    >= target_threshold

)


top_targets = grid[
    grid[
        "Target_Cell"
    ]
].copy()


top_targets = top_targets.sort_values(
    "Prospectivity_Score",
    ascending=False
).reset_index(
    drop=True
)


top_targets[
    "Target_Cell_Rank"
] = np.arange(
    1,
    len(top_targets) + 1
)


top_targets[
    "Target_Level"
] = top_targets[
    "Prospectivity_Score"
].apply(
    lambda score:
        (
            "VERY_HIGH"
            if score >= 98
            else
            "HIGH"
            if score >= 95
            else
            "MODERATE_HIGH"
        )
)


print()
print(
    "Target threshold:",
    f"{target_threshold:.4f}"
)


print(
    "Target cells:",
    len(top_targets)
)


# ============================================================
# 33. CREATE KM COORDINATES FOR CLUSTERING
# ============================================================

mean_lat = grid[
    "latitude"
].mean()


km_per_degree_lat = 111.32


km_per_degree_lon = (

    111.32
    *
    np.cos(
        np.radians(
            mean_lat
        )
    )

)


top_targets[
    "x_km"
] = (

    top_targets[
        "longitude"
    ]
    *
    km_per_degree_lon

)


top_targets[
    "y_km"
] = (

    top_targets[
        "latitude"
    ]
    *
    km_per_degree_lat

)


# ============================================================
# 34. CLUSTER TARGET CELLS
# ============================================================

print()
print("-" * 75)
print("STEP 13 - Clustering exploration target cells")
print("-" * 75)


if len(top_targets) >= CLUSTER_MIN_SAMPLES:

    target_xy = top_targets[
        [
            "x_km",
            "y_km"
        ]
    ].to_numpy()


    clustering = DBSCAN(

        eps=CLUSTER_EPS_KM,

        min_samples=CLUSTER_MIN_SAMPLES

    )


    cluster_labels = clustering.fit_predict(
        target_xy
    )

else:

    cluster_labels = np.full(
        len(top_targets),
        -1
    )


top_targets[
    "Target_Zone_ID"
] = cluster_labels


valid_clusters = [

    cid

    for cid

    in np.unique(
        cluster_labels
    )

    if cid >= 0

]


print(
    "Target cells:",
    len(top_targets)
)


print(
    "Target clusters:",
    len(valid_clusters)
)


# ============================================================
# 35. SAVE TOP TARGET CELLS
# ============================================================

target_output_columns = [

    "Target_Cell_Rank",

    "longitude",

    "latitude",

    "Prospectivity_Score",

    "Model_Probability",

    "Priority",

    "Target_Level",

    "Target_Zone_ID",

    "geo_age",

    "geo_supergroup",

    "geo_group",

    "geo_formation",

    "geo_lithology",

    "geo_intrusive",

    "geo_stratigraphy",

    "Sausar_Group_Proxy",

    "Metamorphic_Host_Proxy",

    "Geology_Unknown",

    "Geo_Boundary_Distance_km",

    "Geo_Boundary_Density_1km",

    "Geo_Boundary_Density_3km",

    "Lithology_Diversity_3km",

    "Formation_Diversity_3km",

    "NDMI",

    "NBR",

    "NDRE",

    "Iron_Oxide_Index",

    "Clay_Alteration_Index",

    "Ferrous_Index",

    "BSI",

    "B11_B12_NormDiff",

    "B8_B12_NormDiff",

    "B4_B2_NormDiff",

    "VV",

    "VH",

    "VV_VH_Difference",

    "Elevation",

    "Slope",

    "Aspect"

]


target_output_columns = [

    col

    for col

    in target_output_columns

    if col in top_targets.columns

]


top_targets[
    target_output_columns
].to_csv(
    TARGET_CELLS_FILE,
    index=False
)


print(
    "Saved:",
    TARGET_CELLS_FILE
)


# ============================================================
# 36. TARGET ZONE SUMMARY
# ============================================================

print()
print("-" * 75)
print("STEP 14 - Creating target-zone summary")
print("-" * 75)


zone_rows = []


# Approximate grid-cell area.
# This is only an estimate for display.

cell_area_km2 = (

    0.01
    *
    111.32

    *

    0.01
    *
    111.32

    *

    np.cos(
        np.radians(
            mean_lat
        )
    )

)


for zone_id in valid_clusters:

    zone = top_targets[
        top_targets[
            "Target_Zone_ID"
        ]
        == zone_id
    ].copy()


    if len(zone) < CLUSTER_MIN_SAMPLES:

        continue


    geology_mode = (
        zone[
            "geo_group"
        ]
        .mode()
    )


    lithology_mode = (
        zone[
            "geo_lithology"
        ]
        .mode()
    )


    formation_mode = (
        zone[
            "geo_formation"
        ]
        .mode()
    )


    dominant_geology = (

        geology_mode.iloc[0]

        if len(
            geology_mode
        )

        else

        "UNKNOWN"

    )


    dominant_lithology = (

        lithology_mode.iloc[0]

        if len(
            lithology_mode
        )

        else

        "UNKNOWN"

    )


    dominant_formation = (

        formation_mode.iloc[0]

        if len(
            formation_mode
        )

        else

        "UNKNOWN"

    )


    zone_rows.append({

        "Target_Zone_ID":
            int(zone_id),

        "Cell_Count":
            int(len(zone)),

        "Approx_Area_km2":
            round(
                len(zone)
                *
                cell_area_km2,
                2
            ),

        "Mean_Score":
            round(
                float(
                    zone[
                        "Prospectivity_Score"
                    ].mean()
                ),
                2
            ),

        "Max_Score":
            round(
                float(
                    zone[
                        "Prospectivity_Score"
                    ].max()
                ),
                2
            ),

        "Mean_Model_Probability":
            float(
                zone[
                    "Model_Probability"
                ].mean()
            ),

        "Centroid_Longitude":
            float(
                zone[
                    "longitude"
                ].mean()
            ),

        "Centroid_Latitude":
            float(
                zone[
                    "latitude"
                ].mean()
            ),

        "Dominant_Geology":
            dominant_geology,

        "Dominant_Lithology":
            dominant_lithology,

        "Dominant_Formation":
            dominant_formation,

        "Interpretation":
            (
                "AI-assisted manganese "
                "exploration priority zone; "
                "field validation and drilling "
                "required."
            )

    })


zones = pd.DataFrame(
    zone_rows
)


if len(zones) > 0:

    zones = zones.sort_values(
        [
            "Mean_Score",
            "Max_Score"
        ],
        ascending=False
    ).reset_index(
        drop=True
    )


    zones[
        "Zone_Rank"
    ] = np.arange(
        1,
        len(zones) + 1
    )


zones.to_csv(
    TARGET_ZONES_FILE,
    index=False
)


print(
    "Saved:",
    TARGET_ZONES_FILE
)


# ============================================================
# 37. TARGET CELL GEOJSON
# ============================================================

print()
print("Creating target-cell GeoJSON...")


cell_features = []


for _, row in top_targets.iterrows():

    point = Point(

        float(
            row["longitude"]
        ),

        float(
            row["latitude"]
        )

    )


    cell_features.append({

        "type":
            "Feature",

        "geometry":
            mapping(
                point
            ),

        "properties": {

            "rank":
                int(
                    row[
                        "Target_Cell_Rank"
                    ]
                ),

            "score":
                round(
                    float(
                        row[
                            "Prospectivity_Score"
                        ]
                    ),
                    2
                ),

            "probability":
                round(
                    float(
                        row[
                            "Model_Probability"
                        ]
                    ),
                    6
                ),

            "priority":
                row[
                    "Priority"
                ],

            "target_level":
                row[
                    "Target_Level"
                ],

            "zone_id":
                int(
                    row[
                        "Target_Zone_ID"
                    ]
                ),

            "geology":
                row[
                    "geo_group"
                ],

            "lithology":
                row[
                    "geo_lithology"
                ],

            "formation":
                row[
                    "geo_formation"
                ]

        }

    })


cells_geojson = {

    "type":
        "FeatureCollection",

    "features":
        cell_features

}


with open(
    TARGET_CELLS_GEOJSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        cells_geojson,
        f
    )


print(
    "Saved:",
    TARGET_CELLS_GEOJSON
)


# ============================================================
# 38. TARGET ZONE GEOJSON
# ============================================================

print(
    "Creating target-zone GeoJSON..."
)


zone_features = []


for zone_id in valid_clusters:

    zone = top_targets[
        top_targets[
            "Target_Zone_ID"
        ]
        == zone_id
    ].copy()


    if len(zone) < CLUSTER_MIN_SAMPLES:

        continue


    points = [

        Point(

            float(
                row["longitude"]
            ),

            float(
                row["latitude"]
            )

        )

        for _, row
        in zone.iterrows()

    ]


    geometry = unary_union(
        points
    ).convex_hull


    # Small display buffer around target cluster.
    geometry = geometry.buffer(
        0.004
    )


    geology_mode = (
        zone[
            "geo_group"
        ]
        .mode()
    )


    lithology_mode = (
        zone[
            "geo_lithology"
        ]
        .mode()
    )


    zone_features.append({

        "type":
            "Feature",

        "geometry":
            mapping(
                geometry
            ),

        "properties": {

            "zone_id":
                int(zone_id),

            "mean_score":
                round(
                    float(
                        zone[
                            "Prospectivity_Score"
                        ].mean()
                    ),
                    2
                ),

            "max_score":
                round(
                    float(
                        zone[
                            "Prospectivity_Score"
                        ].max()
                    ),
                    2
                ),

            "cells":
                int(len(zone)),

            "dominant_geology":
                (
                    geology_mode.iloc[0]
                    if len(
                        geology_mode
                    )
                    else
                    "UNKNOWN"
                ),

            "dominant_lithology":
                (
                    lithology_mode.iloc[0]
                    if len(
                        lithology_mode
                    )
                    else
                    "UNKNOWN"
                ),

            "use":
                (
                    "Exploration prioritization; "
                    "field validation and drilling "
                    "required."
                )

        }

    })


zones_geojson = {

    "type":
        "FeatureCollection",

    "features":
        zone_features

}


with open(
    TARGET_ZONES_GEOJSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        zones_geojson,
        f
    )


print(
    "Saved:",
    TARGET_ZONES_GEOJSON
)


# ============================================================
# 39. SAVE COMPLETE GRID
# ============================================================

print()
print("-" * 75)
print("STEP 15 - Saving final prospectivity grid")
print("-" * 75)


grid.to_csv(
    PREDICTION_FILE,
    index=False
)


print(
    "Saved:",
    PREDICTION_FILE
)


# ============================================================
# 40. PROSPECTIVITY MAP
# ============================================================

print()
print("-" * 75)
print("STEP 16 - Creating prospectivity map")
print("-" * 75)


plt.figure(
    figsize=(
        11,
        8
    )
)


scatter = plt.scatter(

    grid[
        "longitude"
    ],

    grid[
        "latitude"
    ],

    c=grid[
        "Prospectivity_Score"
    ],

    s=7,

    alpha=0.75

)


plt.colorbar(
    scatter,
    label="Relative Prospectivity Score (0-100)"
)


high_targets = grid[
    grid[
        "Prospectivity_Score"
    ] >= 95
]


if len(
    high_targets
) > 0:

    plt.scatter(

        high_targets[
            "longitude"
        ],

        high_targets[
            "latitude"
        ],

        facecolors="none",

        edgecolors="black",

        s=24,

        linewidths=0.5,

        label="High / Very High"

    )


plt.xlabel(
    "Longitude"
)


plt.ylabel(
    "Latitude"
)


plt.title(
    "MOIL AI-Assisted Manganese Prospectivity - Balaghat"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    MAP_FILE,
    dpi=250,
    bbox_inches="tight"
)


plt.close()


print(
    "Saved:",
    MAP_FILE
)


# ============================================================
# 41. SCORE DISTRIBUTION
# ============================================================

print(
    "Creating score distribution..."
)


plt.figure(
    figsize=(
        9,
        6
    )
)


plt.hist(

    grid[
        "Prospectivity_Score"
    ],

    bins=50

)


plt.axvline(

    target_threshold,

    linestyle="--",

    label=(
        f"Top 10% threshold = "
        f"{target_threshold:.2f}"
    )

)


plt.xlabel(
    "Relative Prospectivity Score"
)


plt.ylabel(
    "Number of Grid Cells"
)


plt.title(
    "Balaghat Manganese Prospectivity Score Distribution"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    SCORE_DISTRIBUTION_FILE,
    dpi=250,
    bbox_inches="tight"
)


plt.close()


print(
    "Saved:",
    SCORE_DISTRIBUTION_FILE
)


# ============================================================
# 42. SCORE SUMMARY
# ============================================================

print()
print("-" * 75)
print("STEP 17 - Final prospectivity summary")
print("-" * 75)


scores = grid[
    "Prospectivity_Score"
]


very_high = int(
    (
        scores >= 98
    ).sum()
)


high = int(
    (
        (scores >= 95)
        &
        (scores < 98)
    ).sum()
)


moderate_high = int(
    (
        (scores >= 90)
        &
        (scores < 95)
    ).sum()
)


moderate = int(
    (
        (scores >= 75)
        &
        (scores < 90)
    ).sum()
)


low = int(
    (
        scores < 75
    ).sum()
)


total_cells = len(
    grid
)


summary_rows = [

    (
        "total_grid_cells",
        total_cells
    ),

    (
        "very_high_cells",
        very_high
    ),

    (
        "high_cells",
        high
    ),

    (
        "moderate_high_cells",
        moderate_high
    ),

    (
        "moderate_cells",
        moderate
    ),

    (
        "low_cells",
        low
    ),

    (
        "very_high_percent",
        very_high / total_cells * 100
    ),

    (
        "high_percent",
        high / total_cells * 100
    ),

    (
        "moderate_high_percent",
        moderate_high / total_cells * 100
    ),

    (
        "moderate_percent",
        moderate / total_cells * 100
    ),

    (
        "low_percent",
        low / total_cells * 100
    ),

    (
        "score_min",
        float(scores.min())
    ),

    (
        "score_max",
        float(scores.max())
    ),

    (
        "score_mean",
        float(scores.mean())
    ),

    (
        "score_median",
        float(scores.median())
    ),

    (
        "target_threshold",
        float(target_threshold)
    ),

    (
        "target_cells",
        len(top_targets)
    ),

    (
        "target_zones",
        len(valid_clusters)
    )

]


summary_df = pd.DataFrame(

    summary_rows,

    columns=[
        "metric",
        "value"
    ]

)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


print(
    f"Grid cells       : {total_cells:,}"
)


print(
    f"Score minimum    : {scores.min():.4f}"
)


print(
    f"Score maximum    : {scores.max():.4f}"
)


print(
    f"Score mean       : {scores.mean():.4f}"
)


print(
    f"Score median     : {scores.median():.4f}"
)


print(
    f"Target threshold : {target_threshold:.4f}"
)


print(
    f"Target cells     : {len(top_targets):,}"
)


print(
    f"Target zones     : {len(valid_clusters):,}"
)


# ============================================================
# 43. OUTPUT VERIFICATION
# ============================================================

print()
print("-" * 75)
print("STEP 18 - Output verification")
print("-" * 75)


expected_outputs = [

    FEATURE_MATRIX_FILE,

    PREDICTION_FILE,

    TARGET_CELLS_FILE,

    TARGET_ZONES_FILE,

    TARGET_CELLS_GEOJSON,

    TARGET_ZONES_GEOJSON,

    MAP_FILE,

    SCORE_DISTRIBUTION_FILE,

    SUMMARY_FILE

]


all_ok = True


for filepath in expected_outputs:

    if os.path.exists(
        filepath
    ):

        size_mb = (

            os.path.getsize(
                filepath
            )
            /
            (
                1024 * 1024
            )

        )


        print(
            f"[OK] "
            f"{os.path.basename(filepath)} "
            f"({size_mb:.2f} MB)"
        )

    else:

        print(
            f"[MISSING] "
            f"{os.path.basename(filepath)}"
        )

        all_ok = False


# ============================================================
# 44. FINAL
# ============================================================

print()
print("=" * 75)


if all_ok:

    print(
        "PHASE 5 COMPLETE"
    )

    print(
        "ALL OUTPUTS GENERATED"
    )

else:

    print(
        "PHASE 5 FINISHED WITH MISSING OUTPUTS"
    )


print("=" * 75)


print()
print(
    "NEW 117-FEATURE MODEL USED:"
)


print(
    "  ",
    MODEL_FILE
)


print()
print(
    "117-feature verification: PASS"
)


print()
print(
    "Important interpretation:"
)


print(
    "  Prospectivity score is a relative "
    "exploration-priority ranking."
)


print(
    "  It is NOT manganese concentration, "
    "ore grade, or a proven reserve."
)


print(
    "  High-score zones require geological "
    "field validation and drilling."
)


print()
print(
    "Phase 5 outputs:"
)


for filepath in expected_outputs:

    print(
        "  ",
        filepath
    )


print()
print("=" * 75)