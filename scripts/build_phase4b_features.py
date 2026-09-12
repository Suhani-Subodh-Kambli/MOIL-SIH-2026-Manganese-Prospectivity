import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from shapely.geometry import shape, Point
from shapely.strtree import STRtree
from shapely.ops import nearest_points


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path(
    "data/MOIL_SIH_Phase4B_Geology_Features.csv"
)

GEOLOGY_FILE = Path(
    "data/geology/balaghat/balaghat_lithology.geojsonl"
)

OUTPUT_FILE = Path(
    "data/MOIL_SIH_Phase4B_Enhanced_Features.csv"
)


# ============================================================
# CONSTANTS
# ============================================================

EARTH_KM_PER_DEG = 111.32

RADIUS_1KM_DEG = 1.0 / EARTH_KM_PER_DEG
RADIUS_3KM_DEG = 3.0 / EARTH_KM_PER_DEG


# ============================================================
# SAFE RATIO
# ============================================================

def safe_ratio(a, b):
    """
    Safe division for spectral ratios.
    """
    return a / (b + 1e-6)


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(series):
    return (
        series
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
        .str.strip()
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PHASE 4B.2 + 4B.3")
print("STRUCTURAL + SPECTRAL FEATURE ENGINEERING")
print("=" * 70)

print("\nLoading Phase 4B geology dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Records: {len(df):,}")


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "B2", "B3", "B4",
    "B5", "B6", "B7",
    "B8", "B8A",
    "B11", "B12",
    "Aspect",
    "Elevation",
    "Slope",
    "latitude",
    "longitude",
    "label"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# 4B.3 — SPECTRAL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("4B.3 — SPECTRAL FEATURE ENGINEERING")
print("=" * 70)


# ------------------------------------------------------------
# Vegetation / moisture
# ------------------------------------------------------------

df["NDMI"] = (
    (df["B8"] - df["B11"]) /
    (df["B8"] + df["B11"] + 1e-6)
)

df["NBR"] = (
    (df["B8"] - df["B12"]) /
    (df["B8"] + df["B12"] + 1e-6)
)

df["NDRE"] = (
    (df["B8A"] - df["B5"]) /
    (df["B8A"] + df["B5"] + 1e-6)
)


# ------------------------------------------------------------
# Iron-related spectral proxy
# ------------------------------------------------------------

df["Iron_Oxide_Index"] = safe_ratio(
    df["B4"],
    df["B2"]
)


# ------------------------------------------------------------
# SWIR / alteration proxies
# ------------------------------------------------------------

df["Clay_Alteration_Index"] = safe_ratio(
    df["B11"],
    df["B12"]
)

df["Ferrous_Index"] = safe_ratio(
    df["B12"],
    df["B8"]
)

df["SWIR_Red_Ratio"] = safe_ratio(
    df["B11"],
    df["B4"]
)

df["SWIR_Green_Ratio"] = safe_ratio(
    df["B11"],
    df["B3"]
)

df["NIR_SWIR2_Ratio"] = safe_ratio(
    df["B8"],
    df["B12"]
)


# ------------------------------------------------------------
# Bare Soil Index
# ------------------------------------------------------------

df["BSI"] = (
    (
        (df["B11"] + df["B4"]) -
        (df["B8"] + df["B2"])
    )
    /
    (
        (df["B11"] + df["B4"]) +
        (df["B8"] + df["B2"]) +
        1e-6
    )
)


# ------------------------------------------------------------
# Normalized spectral contrasts
# ------------------------------------------------------------

df["B11_B12_NormDiff"] = (
    (df["B11"] - df["B12"]) /
    (df["B11"] + df["B12"] + 1e-6)
)

df["B8_B12_NormDiff"] = (
    (df["B8"] - df["B12"]) /
    (df["B8"] + df["B12"] + 1e-6)
)

df["B4_B2_NormDiff"] = (
    (df["B4"] - df["B2"]) /
    (df["B4"] + df["B2"] + 1e-6)
)


spectral_features = [
    "NDMI",
    "NBR",
    "NDRE",
    "Iron_Oxide_Index",
    "Clay_Alteration_Index",
    "Ferrous_Index",
    "SWIR_Red_Ratio",
    "SWIR_Green_Ratio",
    "NIR_SWIR2_Ratio",
    "BSI",
    "B11_B12_NormDiff",
    "B8_B12_NormDiff",
    "B4_B2_NormDiff"
]

print("\nAdded spectral features:")

for feature in spectral_features:
    print(f"  + {feature}")


# ============================================================
# GEOLOGICAL HOST PROXIES
# ============================================================

print("\n" + "=" * 70)
print("GEOLOGICAL HOST PROXIES")
print("=" * 70)


df["geo_group_clean"] = clean_text(
    df["geo_group"]
)

df["geo_lithology_clean"] = clean_text(
    df["geo_lithology"]
)

df["geo_formation_clean"] = clean_text(
    df["geo_formation"]
)

df["geo_age_clean"] = clean_text(
    df["geo_age"]
)

df["geo_intrusive_clean"] = clean_text(
    df["geo_intrusive"]
)


# ------------------------------------------------------------
# Sausar Group proxy
# ------------------------------------------------------------

df["Sausar_Group_Proxy"] = (
    df["geo_group_clean"]
    .str.contains(
        "SAUSAR",
        na=False
    )
    .astype(int)
)


# ------------------------------------------------------------
# Metamorphic host proxy
# ------------------------------------------------------------

metamorphic_keywords = [
    "SCHIST",
    "GNEISS",
    "AMPHIBOLITE",
    "QUARTZITE",
    "GRANULITE",
    "PHYLLITE",
    "MARBLE"
]

df["Metamorphic_Host_Proxy"] = (
    df["geo_lithology_clean"]
    .apply(
        lambda x: int(
            any(
                keyword in x
                for keyword in metamorphic_keywords
            )
        )
    )
)


# ------------------------------------------------------------
# Geological unknown flag
# ------------------------------------------------------------

df["Geology_Unknown"] = (
    (
        (df["geo_lithology_clean"] == "UNKNOWN") |
        (df["geo_group_clean"] == "UNKNOWN")
    )
    .astype(int)
)


print(
    f"\nSausar-group points: "
    f"{df['Sausar_Group_Proxy'].sum():,}"
)

print(
    f"Metamorphic-host points: "
    f"{df['Metamorphic_Host_Proxy'].sum():,}"
)


# ============================================================
# LOAD GEOLOGY POLYGONS
# ============================================================

print("\n" + "=" * 70)
print("LOADING GEOLOGICAL POLYGONS")
print("=" * 70)

geometries = []
geo_properties = []

invalid_geometry = 0

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

            feature = json.loads(line)

            geometry_data = feature.get(
                "geometry"
            )

            if not geometry_data:
                continue

            geom = shape(
                geometry_data
            )

            if geom.is_empty:
                continue

            if not geom.is_valid:
                geom = geom.buffer(0)

            if geom.is_empty:
                invalid_geometry += 1
                continue

            geometries.append(geom)

            geo_properties.append(
                feature.get(
                    "properties",
                    {}
                )
            )

        except Exception:
            invalid_geometry += 1


print(
    f"Geology polygons loaded: "
    f"{len(geometries):,}"
)

print(
    f"Invalid/skipped: "
    f"{invalid_geometry:,}"
)


# ============================================================
# SPATIAL INDEX
# ============================================================

print("\nBuilding spatial indexes...")

polygon_tree = STRtree(
    geometries
)

boundaries = [
    geom.boundary
    for geom in geometries
]

boundary_tree = STRtree(
    boundaries
)

print("Spatial indexes ready.")


# ============================================================
# SHAPELY INDEX HELPER
# ============================================================

def get_index(result, objects):
    """
    Shapely 2.x returns integer indices.
    Supports older object-return behaviour too.
    """

    try:
        return int(result)

    except (TypeError, ValueError):

        for i, obj in enumerate(objects):

            if obj is result:
                return i

    return None


# ============================================================
# 4B.2 — GEOLOGICAL / STRUCTURAL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("4B.2 — STRUCTURAL / GEOLOGICAL CONTACT FEATURES")
print("=" * 70)


boundary_distance_km = []
boundary_density_1km = []
boundary_density_3km = []
lithology_diversity_3km = []
formation_diversity_3km = []


for i, row in df.iterrows():

    lon = float(
        row["longitude"]
    )

    lat = float(
        row["latitude"]
    )

    point = Point(
        lon,
        lat
    )


    # --------------------------------------------------------
    # Nearest geological boundary
    # --------------------------------------------------------

    nearest_result = boundary_tree.nearest(
        point
    )

    nearest_idx = get_index(
        nearest_result,
        boundaries
    )

    distance_km = np.nan

    if nearest_idx is not None:

        nearest_boundary = boundaries[
            nearest_idx
        ]

        try:

            _, nearest_point = nearest_points(
                point,
                nearest_boundary
            )

            dx = (
                (
                    nearest_point.x - lon
                )
                *
                EARTH_KM_PER_DEG
                *
                math.cos(
                    math.radians(lat)
                )
            )

            dy = (
                (
                    nearest_point.y - lat
                )
                *
                EARTH_KM_PER_DEG
            )

            distance_km = math.sqrt(
                dx * dx + dy * dy
            )

        except Exception:

            distance_km = np.nan


    # --------------------------------------------------------
    # Boundary density within 1 km
    # --------------------------------------------------------

    search_1km = point.buffer(
        RADIUS_1KM_DEG
    )

    candidates_1km = boundary_tree.query(
        search_1km
    )

    count_1km = len(
        candidates_1km
    )


    # --------------------------------------------------------
    # Boundary density within 3 km
    # --------------------------------------------------------

    search_3km = point.buffer(
        RADIUS_3KM_DEG
    )

    candidates_3km = boundary_tree.query(
        search_3km
    )

    count_3km = len(
        candidates_3km
    )


    # --------------------------------------------------------
    # Geological diversity within 3 km
    # --------------------------------------------------------

    polygon_candidates = polygon_tree.query(
        search_3km
    )

    lithologies = set()
    formations = set()

    for candidate in polygon_candidates:

        idx = get_index(
            candidate,
            geometries
        )

        if idx is None:
            continue

        props = geo_properties[idx]

        lith = props.get(
            "lithologic"
        )

        formation = props.get(
            "formation"
        )

        if lith:

            lithologies.add(
                str(lith)
                .upper()
                .strip()
            )

        if formation:

            formations.add(
                str(formation)
                .upper()
                .strip()
            )


    boundary_distance_km.append(
        distance_km
    )

    boundary_density_1km.append(
        count_1km
    )

    boundary_density_3km.append(
        count_3km
    )

    lithology_diversity_3km.append(
        len(lithologies)
    )

    formation_diversity_3km.append(
        len(formations)
    )


    if (i + 1) % 250 == 0:

        print(
            f"Processed structural points: "
            f"{i + 1:,}/{len(df):,}"
        )


df["Geo_Boundary_Distance_km"] = (
    boundary_distance_km
)

df["Geo_Boundary_Density_1km"] = (
    boundary_density_1km
)

df["Geo_Boundary_Density_3km"] = (
    boundary_density_3km
)

df["Lithology_Diversity_3km"] = (
    lithology_diversity_3km
)

df["Formation_Diversity_3km"] = (
    formation_diversity_3km
)


# ============================================================
# ASPECT CIRCULAR FEATURES
# ============================================================

print("\nAdding circular aspect features...")

aspect_rad = np.deg2rad(
    df["Aspect"].fillna(0)
)

df["Aspect_Sin"] = np.sin(
    aspect_rad
)

df["Aspect_Cos"] = np.cos(
    aspect_rad
)


# ============================================================
# IMPORTANT: NO SAMPLE-BASED TERRAIN FEATURES
# ============================================================

print("\n" + "=" * 70)
print("TERRAIN NEIGHBOURHOOD FEATURES")
print("=" * 70)

print(
    "Skipped Local_Elevation_STD, TPI_Local, "
    "Local_Slope_STD and Local_Aspect_Dispersion."
)

print(
    "Reason: those features were calculated from "
    "the sampled ML points rather than the underlying "
    "SRTM raster, which can encode the sampling design."
)


# ============================================================
# CLEAN TEMPORARY COLUMNS
# ============================================================

df.drop(
    columns=[
        "geo_group_clean",
        "geo_lithology_clean",
        "geo_formation_clean",
        "geo_age_clean",
        "geo_intrusive_clean"
    ],
    inplace=True,
    errors="ignore"
)


# ============================================================
# CLEAN NUMERIC VALUES
# ============================================================

numeric_columns = [
    feature
    for feature in df.columns
    if feature not in [
        "system:index",
        ".geo",
        "label",
        "geo_age",
        "geo_supergroup",
        "geo_group",
        "geo_formation",
        "geo_lithology",
        "geo_intrusive",
        "geo_stratigraphy"
    ]
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    df[column] = df[column].replace(
        [np.inf, -np.inf],
        np.nan
    )


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("4B.2 + 4B.3 COMPLETE")
print("=" * 70)

print(
    f"Records: {len(df):,}"
)

print(
    f"Columns: {len(df.columns):,}"
)

print("\nStructural features added:")

structural_features = [
    "Geo_Boundary_Distance_km",
    "Geo_Boundary_Density_1km",
    "Geo_Boundary_Density_3km",
    "Lithology_Diversity_3km",
    "Formation_Diversity_3km",
    "Aspect_Sin",
    "Aspect_Cos"
]

for feature in structural_features:
    print(f"  + {feature}")


print("\nSpectral features added:")

for feature in spectral_features:
    print(f"  + {feature}")


print("\nRemoved from previous version:")

removed_features = [
    "Local_Elevation_STD",
    "TPI_Local",
    "Local_Slope_STD",
    "Local_Aspect_Dispersion"
]

for feature in removed_features:
    print(f"  - {feature}")


print("\nSaved:")
print(OUTPUT_FILE)

print("\nLabel distribution:")

print(
    df["label"]
    .value_counts()
    .sort_index()
)