# ============================================================
# MOIL SIH 2026
# PHASE 5 - PROSPECTIVITY VALIDATION
# ============================================================

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
GSI_DIR = DATA_DIR / "gsi"

PHASE5_FILE = OUTPUT_DIR / "phase5_prospectivity_grid.csv"

VALIDATION_TOP100_CSV = OUTPUT_DIR / "phase5_validation_top100_cells.csv"
VALIDATION_TOP100_GEOJSON = OUTPUT_DIR / "phase5_validation_top100.geojson"

VALIDATION_MAP = OUTPUT_DIR / "phase5_validation_map.png"
VALIDATION_SCORE_PLOT = OUTPUT_DIR / "phase5_validation_score_distribution.png"

VALIDATION_SUMMARY = OUTPUT_DIR / "phase5_validation_summary.csv"

EXPLORATION_TOP5_CSV = OUTPUT_DIR / "phase5_exploration_shortlist_top5pct.csv"
EXPLORATION_TOP5_GEOJSON = OUTPUT_DIR / "phase5_exploration_shortlist_top5pct.geojson"

RANKED_ZONES_CSV = OUTPUT_DIR / "phase5_ranked_target_zones.csv"

TOP100_N = 100

HIT_THRESHOLDS_KM = [1, 2, 5, 10]


# ============================================================
# HELPERS
# ============================================================

def print_header(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def normalize_column_name(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def find_column(df, candidates):
    normalized = {
        normalize_column_name(c): c
        for c in df.columns
    }

    for candidate in candidates:

        candidate_norm = normalize_column_name(candidate)

        if candidate_norm in normalized:
            return normalized[candidate_norm]

    for candidate in candidates:

        candidate_norm = normalize_column_name(candidate)

        for normalized_name, original_name in normalized.items():

            if (
                candidate_norm in normalized_name
                or normalized_name in candidate_norm
            ):
                return original_name

    return None


def haversine_km(lat1, lon1, lat2, lon2):

    R = 6371.0088

    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))

    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2.0) ** 2
    )

    c = 2 * np.arcsin(
        np.sqrt(np.clip(a, 0, 1))
    )

    return R * c


def calculate_nearest_occurrence_distance(
    target_lat,
    target_lon,
    occurrence_lat,
    occurrence_lon
):

    target_lat = np.asarray(target_lat, dtype=float)
    target_lon = np.asarray(target_lon, dtype=float)

    occurrence_lat = np.asarray(
        occurrence_lat,
        dtype=float
    )

    occurrence_lon = np.asarray(
        occurrence_lon,
        dtype=float
    )

    result = np.empty(
        len(target_lat),
        dtype=float
    )

    chunk_size = 1000

    for start in range(
        0,
        len(target_lat),
        chunk_size
    ):

        end = min(
            start + chunk_size,
            len(target_lat)
        )

        lat_chunk = target_lat[
            start:end,
            None
        ]

        lon_chunk = target_lon[
            start:end,
            None
        ]

        distances = haversine_km(
            lat_chunk,
            lon_chunk,
            occurrence_lat[None, :],
            occurrence_lon[None, :]
        )

        result[start:end] = np.min(
            distances,
            axis=1
        )

    return result


def dataframe_to_geojson(
    df,
    output_path
):

    features = []

    for _, row in df.iterrows():

        try:

            lon = float(row["longitude"])
            lat = float(row["latitude"])

        except Exception:

            continue

        properties = {}

        for col in df.columns:

            if col in [
                "latitude",
                "longitude"
            ]:
                continue

            value = row[col]

            if pd.isna(value):

                value = None

            elif isinstance(
                value,
                (np.integer,)
            ):

                value = int(value)

            elif isinstance(
                value,
                (np.floating,)
            ):

                value = float(value)

            properties[str(col)] = value

        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        lon,
                        lat
                    ]
                },
                "properties": properties
            }
        )

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            geojson,
            f,
            indent=2
        )

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# START
# ============================================================

print_header(
    "MOIL SIH 2026 - PHASE 5 VALIDATION"
)

print(
    f"\nBase directory: {BASE_DIR}"
)

print(
    f"Output directory: {OUTPUT_DIR}"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# STEP 1
# ============================================================

print_header(
    "STEP 1 - Loading Phase 5 prospectivity grid"
)

if not PHASE5_FILE.exists():

    raise FileNotFoundError(
        f"\nPhase 5 prediction file not found:\n"
        f"{PHASE5_FILE}"
    )

grid = pd.read_csv(
    PHASE5_FILE
)

print(
    f"Grid loaded successfully: {grid.shape}"
)

print("\nColumns:")
print(list(grid.columns))


# ============================================================
# STEP 2
# ============================================================

print_header(
    "STEP 2 - Identifying coordinates and score"
)

lat_col = find_column(
    grid,
    [
        "latitude",
        "lat",
        "y"
    ]
)

lon_col = find_column(
    grid,
    [
        "longitude",
        "lon",
        "long",
        "x"
    ]
)

score_col = find_column(
    grid,
    [
        "Prospectivity_Score",
        "prospectivity_score",
        "score"
    ]
)

if lat_col is None:
    raise ValueError(
        "Could not identify latitude column."
    )

if lon_col is None:
    raise ValueError(
        "Could not identify longitude column."
    )

if score_col is None:
    raise ValueError(
        "Could not identify prospectivity score column."
    )

print(
    f"Latitude column : {lat_col}"
)

print(
    f"Longitude column: {lon_col}"
)

print(
    f"Score column    : {score_col}"
)

grid["latitude"] = pd.to_numeric(
    grid[lat_col],
    errors="coerce"
)

grid["longitude"] = pd.to_numeric(
    grid[lon_col],
    errors="coerce"
)

grid["Prospectivity_Score"] = pd.to_numeric(
    grid[score_col],
    errors="coerce"
)

grid = grid.dropna(
    subset=[
        "latitude",
        "longitude",
        "Prospectivity_Score"
    ]
).copy()


# ============================================================
# STEP 3
# ============================================================

print_header(
    "STEP 3 - Prospectivity score statistics"
)

score = grid[
    "Prospectivity_Score"
]

print(
    f"Minimum : {score.min():.4f}"
)

print(
    f"Maximum : {score.max():.4f}"
)

print(
    f"Mean    : {score.mean():.4f}"
)

print(
    f"Median  : {score.median():.4f}"
)

print(
    f"Std     : {score.std():.4f}"
)


# ============================================================
# STEP 4
# ============================================================

print_header(
    "STEP 4 - Creating relative prospectivity rank"
)

grid["Prospectivity_Percentile"] = (
    grid["Prospectivity_Score"]
    .rank(pct=True)
    * 100
)

print(
    "Percentile ranking created."
)


# ============================================================
# STEP 5
# ============================================================

print_header(
    "STEP 5 - Classifying exploration priority"
)


def classify_priority(percentile):

    if percentile >= 98:
        return "VERY_HIGH"

    elif percentile >= 95:
        return "HIGH"

    elif percentile >= 90:
        return "MODERATE_HIGH"

    elif percentile >= 75:
        return "MODERATE"

    return "LOW"


grid["Priority"] = (
    grid["Prospectivity_Percentile"]
    .apply(classify_priority)
)

priority_counts = (
    grid["Priority"]
    .value_counts()
)

priority_order = [
    "VERY_HIGH",
    "HIGH",
    "MODERATE_HIGH",
    "MODERATE",
    "LOW"
]

for priority in priority_order:

    count = int(
        priority_counts.get(
            priority,
            0
        )
    )

    percentage = (
        count / len(grid) * 100
    )

    print(
        f"{priority:<14}: "
        f"{count:6d} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# STEP 6
# ============================================================

print_header(
    "STEP 6 - Loading known manganese occurrences"
)

candidate_files = [
    GSI_DIR / "Manganese_Ore_1.xls",
    GSI_DIR / "Manganese_Ore_1.xlsx",
    GSI_DIR / "Manganese_Ore_1.csv"
]

occurrence_file = None

for file_path in candidate_files:

    if file_path.exists():

        occurrence_file = file_path
        break


if occurrence_file is None:

    if GSI_DIR.exists():

        supported_files = []

        for pattern in [
            "*.csv",
            "*.xls",
            "*.xlsx"
        ]:

            supported_files.extend(
                GSI_DIR.glob(pattern)
            )

        if supported_files:

            occurrence_file = (
                supported_files[0]
            )


occurrence_df = None


if occurrence_file is None:

    print(
        "\nWARNING: GSI occurrence file was not found."
    )

    print(
        f"Expected folder:\n{GSI_DIR}"
    )

else:

    print(
        "\nGSI occurrence file found:"
    )

    print(
        f"  {occurrence_file}"
    )

    try:

        suffix = (
            occurrence_file
            .suffix
            .lower()
        )

        if suffix == ".xls":

            occurrence_df = pd.read_excel(
                occurrence_file,
                engine="xlrd"
            )

        elif suffix == ".xlsx":

            occurrence_df = pd.read_excel(
                occurrence_file,
                engine="openpyxl"
            )

        elif suffix == ".csv":

            occurrence_df = pd.read_csv(
                occurrence_file
            )

        else:

            raise ValueError(
                f"Unsupported file type: "
                f"{suffix}"
            )

        print(
            f"\nLoaded successfully: "
            f"{occurrence_df.shape}"
        )

        print(
            "\nOccurrence dataset columns:"
        )

        print(
            list(occurrence_df.columns)
        )

        print(
            "\nFirst 5 rows:"
        )

        print(
            occurrence_df.head()
        )

    except Exception as e:

        print(
            "\nERROR while loading GSI occurrence file:"
        )

        print(e)

        occurrence_df = None


# ============================================================
# STEP 7
# IMPORTANT: PREFER LATDD / LONDD
# ============================================================

print_header(
    "STEP 7 - Identifying GSI occurrence coordinates"
)

occ_lat_col = None
occ_lon_col = None


if occurrence_df is not None:

    print(
        "\nSearching for decimal-degree "
        "latitude/longitude columns..."
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Prefer LATDD / LONDD over LATITUDE / LONGITUDE.
    #
    # LATITUDE/LONGITUDE in the GSI table are commonly formatted
    # geographic coordinates, while LATDD/LONDD are decimal
    # degree coordinates.
    # --------------------------------------------------------

    occ_lat_col = find_column(
        occurrence_df,
        [
            "LATDD",
            "LAT_DD",
            "LATITUDE_DD",
            "LAT_DECIMAL",
            "LATITUDE_DECIMAL",
            "latitude_decimal"
        ]
    )

    occ_lon_col = find_column(
        occurrence_df,
        [
            "LONDD",
            "LON_DD",
            "LONGITUDE_DD",
            "LON_DECIMAL",
            "LONGITUDE_DECIMAL",
            "longitude_decimal"
        ]
    )

    # Fallback only if decimal columns don't exist.
    if occ_lat_col is None:

        occ_lat_col = find_column(
            occurrence_df,
            [
                "LATITUDE",
                "latitude",
                "LAT",
                "lat"
            ]
        )

    if occ_lon_col is None:

        occ_lon_col = find_column(
            occurrence_df,
            [
                "LONGITUDE",
                "longitude",
                "LON",
                "lon",
                "LONG",
                "long"
            ]
        )

    print(
        f"Detected latitude column : "
        f"{occ_lat_col}"
    )

    print(
        f"Detected longitude column: "
        f"{occ_lon_col}"
    )

    if (
        occ_lat_col is None
        or occ_lon_col is None
    ):

        print(
            "\nWARNING: Could not identify "
            "coordinate columns."
        )

        print(
            "\nAvailable columns:"
        )

        for col in occurrence_df.columns:

            print(
                f"  - {col}"
            )

        occurrence_df = None


# ============================================================
# STEP 8
# ============================================================

print_header(
    "STEP 8 - Cleaning known GSI occurrences"
)

occurrence_validation_available = False


if occurrence_df is not None:

    # Convert decimal degree columns.
    occurrence_df[
        "occ_latitude"
    ] = pd.to_numeric(
        occurrence_df[
            occ_lat_col
        ],
        errors="coerce"
    )

    occurrence_df[
        "occ_longitude"
    ] = pd.to_numeric(
        occurrence_df[
            occ_lon_col
        ],
        errors="coerce"
    )

    print(
        "\nCoordinate conversion:"
    )

    print(
        f"  Original rows: "
        f"{len(occurrence_df)}"
    )

    print(
        f"  Valid latitude values: "
        f"{occurrence_df['occ_latitude'].notna().sum()}"
    )

    print(
        f"  Valid longitude values: "
        f"{occurrence_df['occ_longitude'].notna().sum()}"
    )

    occurrence_df = occurrence_df.dropna(
        subset=[
            "occ_latitude",
            "occ_longitude"
        ]
    ).copy()

    # Geographic sanity check.
    occurrence_df = occurrence_df[
        occurrence_df[
            "occ_latitude"
        ].between(-90, 90)
        &
        occurrence_df[
            "occ_longitude"
        ].between(-180, 180)
    ].copy()

    print(
        f"\nValid occurrence coordinates: "
        f"{len(occurrence_df)}"
    )

    if len(occurrence_df) > 0:

        print(
            "\nSample decimal coordinates:"
        )

        print(
            occurrence_df[
                [
                    "occ_latitude",
                    "occ_longitude"
                ]
            ].head(10).to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Restrict occurrences to Phase 5 area + small buffer.
    # --------------------------------------------------------

    min_lat = grid[
        "latitude"
    ].min()

    max_lat = grid[
        "latitude"
    ].max()

    min_lon = grid[
        "longitude"
    ].min()

    max_lon = grid[
        "longitude"
    ].max()

    buffer_deg = 0.10

    occurrence_df = occurrence_df[
        occurrence_df[
            "occ_latitude"
        ].between(
            min_lat - buffer_deg,
            max_lat + buffer_deg
        )
        &
        occurrence_df[
            "occ_longitude"
        ].between(
            min_lon - buffer_deg,
            max_lon + buffer_deg
        )
    ].copy()

    print(
        f"\nOccurrences inside/near "
        f"Phase 5 AOI: {len(occurrence_df)}"
    )

    if len(occurrence_df) > 0:

        occurrence_validation_available = True

        print(
            "\nGSI occurrence validation is AVAILABLE."
        )

    else:

        print(
            "\nNo GSI occurrences fall inside/near "
            "the Phase 5 Balaghat AOI."
        )

else:

    print(
        "Known occurrence validation unavailable."
    )


# ============================================================
# STEP 9
# ============================================================

print_header(
    "STEP 9 - Independent validation against GSI occurrences"
)


if occurrence_validation_available:

    occurrence_lat = (
        occurrence_df[
            "occ_latitude"
        ].to_numpy()
    )

    occurrence_lon = (
        occurrence_df[
            "occ_longitude"
        ].to_numpy()
    )

    print(
        "\nCalculating distance from every "
        "prediction cell to nearest GSI occurrence..."
    )

    grid[
        "Nearest_GSI_Occurrence_km"
    ] = calculate_nearest_occurrence_distance(
        grid["latitude"].to_numpy(),
        grid["longitude"].to_numpy(),
        occurrence_lat,
        occurrence_lon
    )

    print(
        "Distance calculation complete."
    )

    # --------------------------------------------------------
    # Top 5%, 10%, 20%
    # --------------------------------------------------------

    print(
        "\nOccurrence hit-rate validation:"
    )

    validation_rows = []

    for percentile in [
        95,
        90,
        80
    ]:

        subset = grid[
            grid[
                "Prospectivity_Percentile"
            ] >= percentile
        ].copy()

        print(
            f"\nTop {100 - percentile}% cells:"
        )

        print(
            f"  Cells: {len(subset)}"
        )

        row = {
            "prediction_percentile":
                percentile,
            "top_fraction_percent":
                100 - percentile,
            "cell_count":
                len(subset)
        }

        for threshold in HIT_THRESHOLDS_KM:

            hits = (
                subset[
                    "Nearest_GSI_Occurrence_km"
                ]
                <= threshold
            )

            hit_rate = (
                hits.mean() * 100
                if len(subset) > 0
                else 0
            )

            print(
                f"  Within {threshold:>2} km: "
                f"{hits.sum():>5} cells "
                f"({hit_rate:.2f}%)"
            )

            row[
                f"hit_rate_within_{threshold}km_percent"
            ] = hit_rate

        validation_rows.append(
            row
        )

    occurrence_validation_df = (
        pd.DataFrame(
            validation_rows
        )
    )

    # --------------------------------------------------------
    # Overall distance statistics
    # --------------------------------------------------------

    print(
        "\nNearest occurrence distance statistics:"
    )

    print(
        f"  Minimum : "
        f"{grid['Nearest_GSI_Occurrence_km'].min():.3f} km"
    )

    print(
        f"  Median  : "
        f"{grid['Nearest_GSI_Occurrence_km'].median():.3f} km"
    )

    print(
        f"  Mean    : "
        f"{grid['Nearest_GSI_Occurrence_km'].mean():.3f} km"
    )

    print(
        f"  Maximum : "
        f"{grid['Nearest_GSI_Occurrence_km'].max():.3f} km"
    )

    # --------------------------------------------------------
    # Top 100
    # --------------------------------------------------------

    top100_for_validation = (
        grid
        .sort_values(
            "Prospectivity_Score",
            ascending=False
        )
        .head(TOP100_N)
        .copy()
    )

    print(
        "\nTop-100 nearest-occurrence statistics:"
    )

    print(
        f"  Median distance: "
        f"{top100_for_validation['Nearest_GSI_Occurrence_km'].median():.3f} km"
    )

    print(
        f"  Mean distance: "
        f"{top100_for_validation['Nearest_GSI_Occurrence_km'].mean():.3f} km"
    )

else:

    occurrence_validation_df = (
        pd.DataFrame()
    )

    print(
        "\nGSI occurrence validation skipped."
    )


# ============================================================
# STEP 10
# ============================================================

print_header(
    "STEP 10 - Auditing highest-priority target cells"
)

top100 = (
    grid
    .sort_values(
        "Prospectivity_Score",
        ascending=False
    )
    .head(TOP100_N)
    .copy()
)

top100.to_csv(
    VALIDATION_TOP100_CSV,
    index=False
)

print(
    "Top 100 cells saved."
)

print(
    f"Saved: {VALIDATION_TOP100_CSV}"
)


# ============================================================
# STEP 11
# ============================================================

print_header(
    "STEP 11 - Geological plausibility audit"
)

geology_columns = [
    "geo_group",
    "geo_lithology",
    "geo_formation",
    "geo_age",
    "geo_intrusive",
    "geo_supergroup"
]

available_geology_columns = [
    col
    for col in geology_columns
    if col in top100.columns
]

print(
    "Available geology columns:"
)

for col in available_geology_columns:

    print(
        f"  - {col}"
    )


def print_top_geology_values(
    df,
    column
):

    if column not in df.columns:
        return

    print(
        f"\nTop geological values: "
        f"{column}"
    )

    values = (
        df[column]
        .fillna("UNKNOWN")
        .astype(str)
        .replace("", "UNKNOWN")
        .value_counts()
        .head(10)
    )

    print(values)


for column in available_geology_columns:

    print_top_geology_values(
        top100,
        column
    )


# ============================================================
# STEP 12
# ============================================================

print_header(
    "STEP 12 - Geological indicator check"
)

if "Sausar_Group_Proxy" in top100.columns:

    sausar_mean = pd.to_numeric(
        top100[
            "Sausar_Group_Proxy"
        ],
        errors="coerce"
    ).mean()

    print(
        f"Top-100 Sausar proxy mean: "
        f"{sausar_mean:.3f}"
    )

else:

    sausar_mean = np.nan

    print(
        "Sausar_Group_Proxy not available."
    )


if "Metamorphic_Host_Proxy" in top100.columns:

    metamorphic_mean = pd.to_numeric(
        top100[
            "Metamorphic_Host_Proxy"
        ],
        errors="coerce"
    ).mean()

    print(
        f"Top-100 metamorphic-host proxy mean: "
        f"{metamorphic_mean:.3f}"
    )

else:

    metamorphic_mean = np.nan

    print(
        "Metamorphic_Host_Proxy not available."
    )


# ============================================================
# STEP 13
# ============================================================

print_header(
    "STEP 13 - Target-zone audit"
)

target_zone_file = (
    OUTPUT_DIR /
    "phase5_target_zones.csv"
)

target_zones = None

if target_zone_file.exists():

    target_zones = pd.read_csv(
        target_zone_file
    )

    print(
        f"Target zones loaded: "
        f"{len(target_zones)}"
    )

    print(
        "\nTarget-zone columns:"
    )

    print(
        list(target_zones.columns)
    )

    if "Zone_Rank" in target_zones.columns:

        target_zones = (
            target_zones
            .sort_values("Zone_Rank")
        )

    elif "Mean_Score" in target_zones.columns:

        target_zones = (
            target_zones
            .sort_values(
                "Mean_Score",
                ascending=False
            )
        )

    target_zones.to_csv(
        RANKED_ZONES_CSV,
        index=False
    )

    print(
        f"\nSaved: {RANKED_ZONES_CSV}"
    )

else:

    print(
        "Target-zone file not found."
    )


# ============================================================
# STEP 14
# ============================================================

print_header(
    "STEP 14 - Creating validation GeoJSON"
)

dataframe_to_geojson(
    top100,
    VALIDATION_TOP100_GEOJSON
)


# ============================================================
# STEP 15
# ============================================================

print_header(
    "STEP 15 - Creating validation map"
)

plt.figure(
    figsize=(10, 8)
)

scatter = plt.scatter(
    grid["longitude"],
    grid["latitude"],
    c=grid[
        "Prospectivity_Score"
    ],
    s=3,
    alpha=0.35
)

plt.colorbar(
    scatter,
    label="Prospectivity Score"
)

plt.scatter(
    top100["longitude"],
    top100["latitude"],
    s=12,
    marker="x",
    label="Top 100 Targets"
)

if occurrence_validation_available:

    plt.scatter(
        occurrence_df[
            "occ_longitude"
        ],
        occurrence_df[
            "occ_latitude"
        ],
        s=30,
        marker="o",
        facecolors="none",
        label="Known GSI Mn Occurrences"
    )

plt.xlabel(
    "Longitude"
)

plt.ylabel(
    "Latitude"
)

plt.title(
    "MOIL SIH 2026 - Phase 5 Prospectivity Validation"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    VALIDATION_MAP,
    dpi=200
)

plt.close()

print(
    f"Saved: {VALIDATION_MAP}"
)


# ============================================================
# STEP 16
# ============================================================

print_header(
    "STEP 16 - Creating score distribution plot"
)

plt.figure(
    figsize=(10, 6)
)

plt.hist(
    grid[
        "Prospectivity_Score"
    ],
    bins=50
)

plt.xlabel(
    "Prospectivity Score"
)

plt.ylabel(
    "Number of Cells"
)

plt.title(
    "Phase 5 Prospectivity Score Distribution"
)

plt.tight_layout()

plt.savefig(
    VALIDATION_SCORE_PLOT,
    dpi=200
)

plt.close()

print(
    f"Saved: {VALIDATION_SCORE_PLOT}"
)


# ============================================================
# STEP 17
# ============================================================

print_header(
    "STEP 17 - Creating final validation summary"
)

summary_rows = [
    {
        "metric":
            "total_prediction_cells",
        "value":
            len(grid)
    },
    {
        "metric":
            "very_high_cells",
        "value":
            int(
                (
                    grid["Priority"]
                    == "VERY_HIGH"
                ).sum()
            )
    },
    {
        "metric":
            "high_cells",
        "value":
            int(
                (
                    grid["Priority"]
                    == "HIGH"
                ).sum()
            )
    },
    {
        "metric":
            "moderate_high_cells",
        "value":
            int(
                (
                    grid["Priority"]
                    == "MODERATE_HIGH"
                ).sum()
            )
    },
    {
        "metric":
            "moderate_cells",
        "value":
            int(
                (
                    grid["Priority"]
                    == "MODERATE"
                ).sum()
            )
    },
    {
        "metric":
            "low_cells",
        "value":
            int(
                (
                    grid["Priority"]
                    == "LOW"
                ).sum()
            )
    },
    {
        "metric":
            "score_min",
        "value":
            float(
                grid[
                    "Prospectivity_Score"
                ].min()
            )
    },
    {
        "metric":
            "score_max",
        "value":
            float(
                grid[
                    "Prospectivity_Score"
                ].max()
            )
    },
    {
        "metric":
            "score_mean",
        "value":
            float(
                grid[
                    "Prospectivity_Score"
                ].mean()
            )
    },
    {
        "metric":
            "score_median",
        "value":
            float(
                grid[
                    "Prospectivity_Score"
                ].median()
            )
    },
    {
        "metric":
            "top_100_cells",
        "value":
            len(top100)
    },
    {
        "metric":
            "occurrence_validation_available",
        "value":
            int(
                occurrence_validation_available
            )
    }
]


if occurrence_validation_available:

    for threshold in HIT_THRESHOLDS_KM:

        hit_rate = (
            (
                top100[
                    "Nearest_GSI_Occurrence_km"
                ]
                <= threshold
            ).mean()
            * 100
        )

        summary_rows.append(
            {
                "metric":
                    (
                        "top100_hit_rate_within_"
                        f"{threshold}km_percent"
                    ),
                "value":
                    float(hit_rate)
            }
        )

    summary_rows.extend(
        [
            {
                "metric":
                    (
                        "top100_median_distance_"
                        "to_gsi_km"
                    ),
                "value":
                    float(
                        top100[
                            "Nearest_GSI_Occurrence_km"
                        ].median()
                    )
            },
            {
                "metric":
                    (
                        "top100_mean_distance_"
                        "to_gsi_km"
                    ),
                "value":
                    float(
                        top100[
                            "Nearest_GSI_Occurrence_km"
                        ].mean()
                    )
            },
            {
                "metric":
                    "gsi_occurrences_used_for_validation",
                "value":
                    len(occurrence_df)
            }
        ]
    )


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    VALIDATION_SUMMARY,
    index=False
)

print(
    f"Saved: {VALIDATION_SUMMARY}"
)


# ============================================================
# STEP 18
# ============================================================

print_header(
    "STEP 18 - Creating ranked exploration targets"
)

top5 = (
    grid[
        grid[
            "Prospectivity_Percentile"
        ] >= 95
    ]
    .sort_values(
        "Prospectivity_Score",
        ascending=False
    )
    .copy()
)

print(
    f"Top 5% cells: {len(top5)}"
)

top5.to_csv(
    EXPLORATION_TOP5_CSV,
    index=False
)

print(
    f"Saved: {EXPLORATION_TOP5_CSV}"
)


# ============================================================
# STEP 19
# ============================================================

print_header(
    "STEP 19 - Creating exploration shortlist GeoJSON"
)

dataframe_to_geojson(
    top5,
    EXPLORATION_TOP5_GEOJSON
)


# ============================================================
# STEP 20
# ============================================================

print_header(
    "STEP 20 - FINAL VALIDATION OUTPUT CHECK"
)

output_files = [
    OUTPUT_DIR /
    "phase5_prospectivity_grid.csv",

    OUTPUT_DIR /
    "phase5_top_target_cells.csv",

    OUTPUT_DIR /
    "phase5_target_zones.csv",

    OUTPUT_DIR /
    "phase5_target_cells.geojson",

    OUTPUT_DIR /
    "phase5_target_zones.geojson",

    OUTPUT_DIR /
    "phase5_prospectivity_map.png",

    OUTPUT_DIR /
    "phase5_summary.csv",

    VALIDATION_TOP100_CSV,

    VALIDATION_TOP100_GEOJSON,

    VALIDATION_MAP,

    VALIDATION_SCORE_PLOT,

    VALIDATION_SUMMARY,

    EXPLORATION_TOP5_CSV,

    EXPLORATION_TOP5_GEOJSON,

    RANKED_ZONES_CSV
]


for file_path in output_files:

    if file_path.exists():

        size_mb = (
            file_path.stat().st_size
            /
            (1024 * 1024)
        )

        print(
            f"[OK]      "
            f"{file_path.name:<55} "
            f"{size_mb:.2f} MB"
        )

    else:

        print(
            f"[MISSING] "
            f"{file_path.name}"
        )


# ============================================================
# FINAL
# ============================================================

print_header(
    "PHASE 5 VALIDATION COMPLETE"
)

print(
    """
IMPORTANT INTERPRETATION:

The prospectivity score is an exploration-priority ranking.
It does NOT represent manganese concentration or a proven reserve.

High-score areas should be treated as targets for:
  - geological field validation
  - detailed mapping
  - sampling
  - geophysical investigation
  - drilling prioritization

The GSI occurrence comparison is an independent spatial
validation of whether high-score areas are associated with
known manganese occurrences.

Distance_Mn was NOT used for this validation because it was
part of the earlier label-generation process.
"""
)


if occurrence_validation_available:

    print(
        "\nGSI occurrence validation: AVAILABLE"
    )

    print(
        f"GSI occurrences used: "
        f"{len(occurrence_df)}"
    )

    for threshold in HIT_THRESHOLDS_KM:

        hit_rate = (
            (
                top100[
                    "Nearest_GSI_Occurrence_km"
                ]
                <= threshold
            ).mean()
            * 100
        )

        print(
            f"Top-100 cells within "
            f"{threshold} km of GSI occurrence: "
            f"{hit_rate:.2f}%"
        )

else:

    print(
        "\nGSI occurrence validation: NOT AVAILABLE"
    )


print(
    "\nProduction-shortfall prediction remains a "
    "separate model/module."
)

print("\nDone.")