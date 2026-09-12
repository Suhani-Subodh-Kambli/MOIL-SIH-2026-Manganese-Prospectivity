import geopandas as gpd
from pathlib import Path

# --------------------------------------------------
# Find the extracted GeoJSONL automatically
# --------------------------------------------------

folder = Path("data/geology/extracted")

files = list(folder.glob("*.geojsonl"))

if not files:
    files = list(folder.glob("*.geojson"))

if not files:
    raise FileNotFoundError(
        "No .geojsonl or .geojson file found in "
        "data/geology/extracted"
    )

path = files[0]

print("=" * 60)
print("FILE")
print("=" * 60)
print(path)

# --------------------------------------------------
# Read file
# --------------------------------------------------

print("\nReading geological data...")

gdf = gpd.read_file(path)

print("\nShape:")
print(gdf.shape)

print("\nCRS:")
print(gdf.crs)

print("\nColumns:")
for col in gdf.columns:
    print(" -", col)

print("\nFirst 5 records:")
print(gdf.head())

print("\nGeometry types:")
print(gdf.geometry.geom_type.value_counts())

# --------------------------------------------------
# Print unique values for non-geometry fields
# --------------------------------------------------

print("\n" + "=" * 60)
print("ATTRIBUTE VALUES")
print("=" * 60)

for col in gdf.columns:

    if col == "geometry":
        continue

    print("\n----------------------------------------")
    print(col)
    print("----------------------------------------")

    try:
        print(
            gdf[col]
            .dropna()
            .astype(str)
            .value_counts()
            .head(20)
        )
    except Exception as e:
        print("Could not inspect:", e)