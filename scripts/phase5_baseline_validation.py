from pathlib import Path
import numpy as np
import pandas as pd

# ============================================================
# MOIL SIH 2026
# PHASE 5 - MODEL VS RANDOM SPATIAL BASELINE
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
GSI_DIR = BASE_DIR / "data" / "gsi"

GRID_FILE = OUTPUT_DIR / "phase5_prospectivity_grid.csv"

N_RANDOM = 5000
RANDOM_SEED = 42

print("=" * 75)
print("MOIL SIH 2026 - PHASE 5 MODEL VS RANDOM BASELINE")
print("=" * 75)

# ============================================================
# STEP 1 - LOAD PROSPECTIVITY GRID
# ============================================================

print("\n" + "=" * 75)
print("STEP 1 - Loading prospectivity grid")
print("=" * 75)

grid = pd.read_csv(GRID_FILE)

print(f"Grid shape: {grid.shape}")

# ------------------------------------------------------------
# Identify columns
# ------------------------------------------------------------

def find_column(df, candidates):
    lookup = {c.lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


lat_col = find_column(
    grid,
    ["latitude", "lat", "LATITUDE"]
)

lon_col = find_column(
    grid,
    ["longitude", "lon", "LONGITUDE"]
)

score_col = find_column(
    grid,
    ["Prospectivity_Score", "Model_Probability"]
)

if lat_col is None or lon_col is None or score_col is None:
    raise ValueError(
        "Could not identify latitude, longitude, or prospectivity score columns."
    )

print(f"Latitude : {lat_col}")
print(f"Longitude: {lon_col}")
print(f"Score    : {score_col}")

grid[lat_col] = pd.to_numeric(grid[lat_col], errors="coerce")
grid[lon_col] = pd.to_numeric(grid[lon_col], errors="coerce")
grid[score_col] = pd.to_numeric(grid[score_col], errors="coerce")

grid = grid.dropna(
    subset=[lat_col, lon_col, score_col]
).reset_index(drop=True)

print(f"Valid prediction cells: {len(grid)}")


# ============================================================
# STEP 2 - LOAD ALL-INDIA GSI DATA
# ============================================================

print("\n" + "=" * 75)
print("STEP 2 - Loading all-India GSI manganese occurrences")
print("=" * 75)

gsi_files = list(GSI_DIR.glob("*.xls")) + list(GSI_DIR.glob("*.xlsx"))

if not gsi_files:
    raise FileNotFoundError(
        f"No GSI Excel file found in {GSI_DIR}"
    )

gsi_file = gsi_files[0]

print(f"Using: {gsi_file}")

if gsi_file.suffix.lower() == ".xls":
    occurrence_df = pd.read_excel(
        gsi_file,
        engine="xlrd"
    )
else:
    occurrence_df = pd.read_excel(gsi_file)

print(f"All-India occurrence rows: {len(occurrence_df)}")


# ============================================================
# STEP 3 - IDENTIFY DECIMAL COORDINATES
# ============================================================

print("\n" + "=" * 75)
print("STEP 3 - Extracting decimal-degree coordinates")
print("=" * 75)

occ_lat_col = find_column(
    occurrence_df,
    [
        "LATDD",
        "LAT_DD",
        "LATITUDE_DD",
        "LAT_DECIMAL",
        "LATITUDE_DECIMAL"
    ]
)

occ_lon_col = find_column(
    occurrence_df,
    [
        "LONDD",
        "LON_DD",
        "LONGITUDE_DD",
        "LON_DECIMAL",
        "LONGITUDE_DECIMAL"
    ]
)

if occ_lat_col is None:
    occ_lat_col = find_column(
        occurrence_df,
        ["LATITUDE"]
    )

if occ_lon_col is None:
    occ_lon_col = find_column(
        occurrence_df,
        ["LONGITUDE"]
    )

if occ_lat_col is None or occ_lon_col is None:
    raise ValueError(
        "Could not identify GSI latitude/longitude columns."
    )

print(f"GSI latitude : {occ_lat_col}")
print(f"GSI longitude: {occ_lon_col}")

occurrence_df["occ_latitude"] = pd.to_numeric(
    occurrence_df[occ_lat_col],
    errors="coerce"
)

occurrence_df["occ_longitude"] = pd.to_numeric(
    occurrence_df[occ_lon_col],
    errors="coerce"
)

occurrence_df = occurrence_df.dropna(
    subset=["occ_latitude", "occ_longitude"]
).copy()

# sanity check
occurrence_df = occurrence_df[
    occurrence_df["occ_latitude"].between(-90, 90)
    &
    occurrence_df["occ_longitude"].between(-180, 180)
].copy()

print(
    f"Valid all-India GSI coordinates: {len(occurrence_df)}"
)


# ============================================================
# STEP 4 - FILTER GSI OCCURRENCES TO BALAGHAT AREA
# ============================================================

print("\n" + "=" * 75)
print("STEP 4 - Filtering GSI occurrences to Phase 5 AOI")
print("=" * 75)

# Use prediction-grid extent rather than hard-coding AOI.

min_lat = grid[lat_col].min()
max_lat = grid[lat_col].max()
min_lon = grid[lon_col].min()
max_lon = grid[lon_col].max()

BUFFER_DEG = 0.10

print(
    f"Grid latitude range : {min_lat:.4f} to {max_lat:.4f}"
)

print(
    f"Grid longitude range: {min_lon:.4f} to {max_lon:.4f}"
)

gsi_local = occurrence_df[
    (occurrence_df["occ_latitude"] >= min_lat - BUFFER_DEG)
    &
    (occurrence_df["occ_latitude"] <= max_lat + BUFFER_DEG)
    &
    (occurrence_df["occ_longitude"] >= min_lon - BUFFER_DEG)
    &
    (occurrence_df["occ_longitude"] <= max_lon + BUFFER_DEG)
].copy()

print(
    f"GSI occurrences inside/near AOI: {len(gsi_local)}"
)

if len(gsi_local) == 0:
    raise ValueError(
        "No GSI occurrences found near the Phase 5 AOI."
    )

print("\nLocal GSI occurrences:")

display_cols = [
    c for c in
    ["LOCALITY", "STATE", "FORMATION",
     "LATDD", "LONDD",
     "occ_latitude", "occ_longitude"]
    if c in gsi_local.columns
]

print(
    gsi_local[display_cols].to_string(index=False)
)


# ============================================================
# STEP 5 - HAVERSINE DISTANCE
# ============================================================

print("\n" + "=" * 75)
print("STEP 5 - Calculating nearest GSI occurrence distance")
print("=" * 75)

def haversine_matrix(
    lat1,
    lon1,
    lat2,
    lon2
):
    """
    Calculate distances in km.

    lat1/lon1 = prediction cells
    lat2/lon2 = GSI occurrences
    """

    R = 6371.0088

    lat1 = np.radians(
        np.asarray(lat1)[:, None]
    )

    lon1 = np.radians(
        np.asarray(lon1)[:, None]
    )

    lat2 = np.radians(
        np.asarray(lat2)[None, :]
    )

    lon2 = np.radians(
        np.asarray(lon2)[None, :]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    distance = (
        2
        * R
        * np.arcsin(np.sqrt(a))
    )

    return distance


distances = haversine_matrix(
    grid[lat_col].values,
    grid[lon_col].values,
    gsi_local["occ_latitude"].values,
    gsi_local["occ_longitude"].values
)

grid["Nearest_GSI_Distance_km"] = distances.min(axis=1)

print("Distance calculation complete.")

print(
    f"Minimum distance: "
    f"{grid['Nearest_GSI_Distance_km'].min():.3f} km"
)

print(
    f"Median distance: "
    f"{grid['Nearest_GSI_Distance_km'].median():.3f} km"
)


# ============================================================
# STEP 6 - DEFINE TOP MODEL GROUPS
# ============================================================

print("\n" + "=" * 75)
print("STEP 6 - Evaluating model-selected cells")
print("=" * 75)

grid = grid.sort_values(
    score_col,
    ascending=False
).reset_index(drop=True)

N = len(grid)

fractions = {
    "Top_1pct": 0.01,
    "Top_5pct": 0.05,
    "Top_10pct": 0.10,
    "Top_20pct": 0.20,
}

thresholds_km = [
    1,
    2,
    5,
    10,
    20,
]


# ============================================================
# STEP 7 - MODEL HIT RATES
# ============================================================

print("\n" + "=" * 75)
print("STEP 7 - Model hit rates")
print("=" * 75)

model_results = []

for name, fraction in fractions.items():

    n_cells = max(
        1,
        int(round(N * fraction))
    )

    selected = grid.head(n_cells)

    result = {
        "Selection": name,
        "Cells": n_cells,
    }

    print(f"\n{name}: {n_cells} cells")

    for threshold in thresholds_km:

        hits = (
            selected["Nearest_GSI_Distance_km"]
            <= threshold
        ).sum()

        rate = hits / n_cells

        result[f"Within_{threshold}km_Rate"] = rate

        print(
            f"  Within {threshold:>2} km: "
            f"{hits:>4} cells "
            f"({rate * 100:.2f}%)"
        )

    result["Mean_Distance_km"] = (
        selected["Nearest_GSI_Distance_km"].mean()
    )

    result["Median_Distance_km"] = (
        selected["Nearest_GSI_Distance_km"].median()
    )

    model_results.append(result)


model_results_df = pd.DataFrame(model_results)


# ============================================================
# STEP 8 - RANDOM BASELINE
# ============================================================

print("\n" + "=" * 75)
print("STEP 8 - Creating random spatial baselines")
print("=" * 75)

print(
    f"Random simulations: {N_RANDOM:,}"
)

rng = np.random.default_rng(RANDOM_SEED)

random_results = {}

for name, fraction in fractions.items():

    n_cells = max(
        1,
        int(round(N * fraction))
    )

    print(
        f"\nRunning {name}: "
        f"{n_cells} random cells per simulation..."
    )

    random_hit_rates = {
        threshold: []
        for threshold in thresholds_km
    }

    random_mean_distances = []
    random_median_distances = []

    all_indices = np.arange(N)

    for _ in range(N_RANDOM):

        indices = rng.choice(
            all_indices,
            size=n_cells,
            replace=False
        )

        selected_distances = (
            grid.iloc[indices]
            ["Nearest_GSI_Distance_km"]
            .values
        )

        for threshold in thresholds_km:

            hit_rate = (
                selected_distances <= threshold
            ).mean()

            random_hit_rates[threshold].append(
                hit_rate
            )

        random_mean_distances.append(
            selected_distances.mean()
        )

        random_median_distances.append(
            np.median(selected_distances)
        )

    random_results[name] = {
        "n_cells": n_cells,
        "hit_rates": random_hit_rates,
        "mean_distances": np.array(
            random_mean_distances
        ),
        "median_distances": np.array(
            random_median_distances
        ),
    }

print("\nRandom baseline simulations complete.")


# ============================================================
# STEP 9 - MODEL VS RANDOM COMPARISON
# ============================================================

print("\n" + "=" * 75)
print("STEP 9 - MODEL VS RANDOM COMPARISON")
print("=" * 75)

comparison_rows = []

for _, model_row in model_results_df.iterrows():

    name = model_row["Selection"]

    random_data = random_results[name]

    for threshold in thresholds_km:

        model_rate = model_row[
            f"Within_{threshold}km_Rate"
        ]

        random_rates = np.array(
            random_data["hit_rates"][threshold]
        )

        random_mean = random_rates.mean()

        random_std = random_rates.std()

        random_p95 = np.percentile(
            random_rates,
            95
        )

        random_p99 = np.percentile(
            random_rates,
            99
        )

        # One-sided empirical p-value:
        # probability random >= model
        p_value = (
            np.sum(
                random_rates >= model_rate
            ) + 1
        ) / (
            len(random_rates) + 1
        )

        enrichment = (
            model_rate / random_mean
            if random_mean > 0
            else np.nan
        )

        comparison_rows.append({
            "Selection": name,
            "Cells": int(model_row["Cells"]),
            "Distance_km": threshold,
            "Model_Hit_Rate": model_rate,
            "Random_Mean_Hit_Rate": random_mean,
            "Random_Std": random_std,
            "Random_P95": random_p95,
            "Random_P99": random_p99,
            "Enrichment": enrichment,
            "Empirical_p_value": p_value,
        })


comparison_df = pd.DataFrame(
    comparison_rows
)

print(
    comparison_df.to_string(
        index=False
    )
)


# ============================================================
# STEP 10 - DISTANCE COMPARISON
# ============================================================

print("\n" + "=" * 75)
print("STEP 10 - Distance distribution comparison")
print("=" * 75)

distance_rows = []

for name, fraction in fractions.items():

    n_cells = max(
        1,
        int(round(N * fraction))
    )

    selected = grid.head(n_cells)

    model_mean = (
        selected["Nearest_GSI_Distance_km"]
        .mean()
    )

    model_median = (
        selected["Nearest_GSI_Distance_km"]
        .median()
    )

    random_data = random_results[name]

    random_mean = (
        random_data["mean_distances"].mean()
    )

    random_median = (
        np.median(
            random_data["median_distances"]
        )
    )

    mean_enrichment = (
        random_mean / model_mean
        if model_mean > 0
        else np.nan
    )

    median_enrichment = (
        random_median / model_median
        if model_median > 0
        else np.nan
    )

    print(f"\n{name}")

    print(
        f"  Model mean distance  : "
        f"{model_mean:.3f} km"
    )

    print(
        f"  Random mean distance : "
        f"{random_mean:.3f} km"
    )

    print(
        f"  Model median distance : "
        f"{model_median:.3f} km"
    )

    print(
        f"  Random median distance: "
        f"{random_median:.3f} km"
    )

    print(
        f"  Mean-distance improvement: "
        f"{mean_enrichment:.3f}x"
    )

    print(
        f"  Median-distance improvement: "
        f"{median_enrichment:.3f}x"
    )

    distance_rows.append({
        "Selection": name,
        "Cells": n_cells,
        "Model_Mean_Distance_km": model_mean,
        "Random_Mean_Distance_km": random_mean,
        "Model_Median_Distance_km": model_median,
        "Random_Median_Distance_km": random_median,
        "Mean_Distance_Improvement": mean_enrichment,
        "Median_Distance_Improvement": median_enrichment,
    })


distance_df = pd.DataFrame(
    distance_rows
)


# ============================================================
# STEP 11 - SAVE OUTPUTS
# ============================================================

print("\n" + "=" * 75)
print("STEP 11 - Saving validation outputs")
print("=" * 75)

comparison_file = (
    OUTPUT_DIR
    / "phase5_model_vs_random_comparison.csv"
)

distance_file = (
    OUTPUT_DIR
    / "phase5_model_vs_random_distance.csv"
)

model_file = (
    OUTPUT_DIR
    / "phase5_model_hit_rates.csv"
)

gsi_local_file = (
    OUTPUT_DIR
    / "phase5_balaghat_gsi_occurrences.csv"
)

comparison_df.to_csv(
    comparison_file,
    index=False
)

distance_df.to_csv(
    distance_file,
    index=False
)

model_results_df.to_csv(
    model_file,
    index=False
)

gsi_local.to_csv(
    gsi_local_file,
    index=False
)

print(f"Saved: {comparison_file}")
print(f"Saved: {distance_file}")
print(f"Saved: {model_file}")
print(f"Saved: {gsi_local_file}")


# ============================================================
# STEP 12 - FINAL INTERPRETATION
# ============================================================

print("\n" + "=" * 75)
print("STEP 12 - FINAL INTERPRETATION")
print("=" * 75)

print("""
This test compares the current XGBoost prospectivity ranking
against random spatial selections of the same size.

Interpretation:

1. Enrichment > 1
   = model performs better than random for that distance threshold.

2. Empirical p-value <= 0.05
   = model association is statistically stronger than
     almost all random selections in this simulation.

3. Enrichment around 1
   = model is approximately random for that metric.

4. Enrichment < 1
   = model is worse than random for that metric.

IMPORTANT:
This is NOT proof that the model detects underground manganese.

It only tests whether the prospectivity ranking has spatial
association with independently known GSI manganese occurrences.

Distance_Mn is NOT used in the model or this validation.
""")

# ------------------------------------------------------------
# Compact final table
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("KEY RESULTS")
print("=" * 75)

key = comparison_df[
    comparison_df["Distance_km"].isin([2, 5, 10])
].copy()

key["Model_%"] = (
    key["Model_Hit_Rate"] * 100
)

key["Random_%"] = (
    key["Random_Mean_Hit_Rate"] * 100
)

key["Enrichment"] = (
    key["Enrichment"].round(2)
)

key["p"] = (
    key["Empirical_p_value"].round(4)
)

print(
    key[
        [
            "Selection",
            "Distance_km",
            "Model_%",
            "Random_%",
            "Enrichment",
            "p",
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 75)
print("PHASE 5 MODEL VS RANDOM BASELINE COMPLETE")
print("=" * 75)