import json
import pandas as pd
from pathlib import Path
from shapely.geometry import shape, Point
from shapely.strtree import STRtree


# ============================================================
# PATHS
# ============================================================

ML_FILE = Path("data/MOIL_SIH_Phase3_ML_Features.csv")

GEOLOGY_FILE = Path(
    "data/geology/balaghat/balaghat_lithology.geojsonl"
)

OUTPUT_FILE = Path(
    "data/MOIL_SIH_Phase4B_Geology_Features.csv"
)


# ============================================================
# LOAD ML DATA
# ============================================================

print("=" * 60)
print("PHASE 4B.1 - ATTACH NGDR GEOLOGY")
print("=" * 60)

print("\nLoading ML feature dataset...")

df = pd.read_csv(ML_FILE)

print(f"ML records: {len(df)}")


# ============================================================
# LOAD GEOLOGY POLYGONS
# ============================================================

print("\nLoading Balaghat geology polygons...")

geometries = []
properties = []

invalid_geometry = 0
count = 0

with open(GEOLOGY_FILE, "r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        try:
            feature = json.loads(line)

            geom_data = feature.get("geometry")

            if not geom_data:
                continue

            geom = shape(geom_data)

            if geom.is_empty:
                continue

            # Repair simple invalid geometries
            if not geom.is_valid:
                geom = geom.buffer(0)

            if geom.is_empty:
                invalid_geometry += 1
                continue

            geometries.append(geom)
            properties.append(feature.get("properties", {}))

            count += 1

            if count % 1000 == 0:
                print(f"Loaded geology polygons: {count:,}")

        except Exception:
            invalid_geometry += 1


print("\nGeology loading complete.")
print(f"Geology polygons loaded : {len(geometries):,}")
print(f"Invalid/skipped         : {invalid_geometry:,}")


# ============================================================
# BUILD SPATIAL INDEX
# ============================================================

print("\nBuilding spatial index...")

tree = STRtree(geometries)

print("Spatial index ready.")


# ============================================================
# MATCH EACH ML POINT
# ============================================================

print("\nMatching ML points to geological polygons...")

geology_columns = {
    "geo_age": [],
    "geo_supergroup": [],
    "geo_group": [],
    "geo_formation": [],
    "geo_lithology": [],
    "geo_intrusive": [],
    "geo_stratigraphy": []
}

matched = 0
unmatched = 0
multiple_matches = 0


for i, row in df.iterrows():

    lon = row["longitude"]
    lat = row["latitude"]

    point = Point(float(lon), float(lat))

    # Find candidate polygons
    candidates = tree.query(point)

    matches = []

    for idx in candidates:

        geom = geometries[idx]

        if geom.intersects(point):
            matches.append(idx)

    # --------------------------------------------------------
    # NO GEOLOGY MATCH
    # --------------------------------------------------------

    if len(matches) == 0:

        unmatched += 1

        for key in geology_columns:
            geology_columns[key].append(None)

        continue

    # --------------------------------------------------------
    # MULTIPLE POLYGON MATCH
    # --------------------------------------------------------

    if len(matches) > 1:
        multiple_matches += 1

    # Use first valid polygon
    idx = matches[0]

    p = properties[idx]

    geology_columns["geo_age"].append(
        p.get("age")
    )

    geology_columns["geo_supergroup"].append(
        p.get("supergroup")
    )

    geology_columns["geo_group"].append(
        p.get("group_name")
    )

    geology_columns["geo_formation"].append(
        p.get("formation")
    )

    geology_columns["geo_lithology"].append(
        p.get("lithologic")
    )

    geology_columns["geo_intrusive"].append(
        p.get("intrusive")
    )

    geology_columns["geo_stratigraphy"].append(
        p.get("stratigraphy_new")
    )

    matched += 1

    if (i + 1) % 250 == 0:
        print(
            f"Processed points: {i + 1:,} | "
            f"Matched: {matched:,} | "
            f"Unmatched: {unmatched:,}"
        )


# ============================================================
# ADD GEOLOGY COLUMNS
# ============================================================

for column, values in geology_columns.items():
    df[column] = values


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

print("\n" + "=" * 60)
print("GEOLOGY ATTACHMENT COMPLETE")
print("=" * 60)

print(f"Total ML points       : {len(df):,}")
print(f"Geology matched       : {matched:,}")
print(f"Geology unmatched     : {unmatched:,}")
print(f"Multiple matches      : {multiple_matches:,}")

print("\nGeology columns added:")

for column in geology_columns:
    print(f"  {column}")

print("\nOutput:")
print(OUTPUT_FILE)

print("\nLithology distribution:")

print(
    df["geo_lithology"]
    .value_counts(dropna=False)
    .head(20)
)

print("\nFormation distribution:")

print(
    df["geo_formation"]
    .value_counts(dropna=False)
    .head(20)
)