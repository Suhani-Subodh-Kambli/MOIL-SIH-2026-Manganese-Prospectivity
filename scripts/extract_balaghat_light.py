import json
from pathlib import Path

# ============================================================
# INPUT
# ============================================================

INPUT = Path(
    "data/geology/extracted/"
    "NGDR_Lithology_50k.geojsonl"
)

# ============================================================
# OUTPUT
# ============================================================

OUTPUT = Path(
    "data/geology/balaghat/"
    "balaghat_lithology.geojsonl"
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# BALAGHAT AOI
# ============================================================

MIN_LON = 79.50
MAX_LON = 80.80

MIN_LAT = 21.30
MAX_LAT = 22.40

# ============================================================
# CHECK FILE
# ============================================================

if not INPUT.exists():

    raise FileNotFoundError(
        f"\nCould not find:\n{INPUT}\n\n"
        "Run:\n"
        "Get-ChildItem data\\geology\\extracted -Recurse\n"
        "and check the exact filename."
    )

print("=" * 60)
print("LIGHTWEIGHT NGDR BALAGHAT EXTRACTION")
print("=" * 60)

print("\nInput:")
print(INPUT)

print("\nOutput:")
print(OUTPUT)

print("\nAOI:")
print(
    f"Longitude: {MIN_LON} to {MAX_LON}"
)

print(
    f"Latitude : {MIN_LAT} to {MAX_LAT}"
)

# ============================================================
# STREAM PROCESS
# ============================================================

total = 0
matched = 0
invalid = 0

with open(
    INPUT,
    "r",
    encoding="utf-8"
) as infile, open(
    OUTPUT,
    "w",
    encoding="utf-8"
) as outfile:

    for line in infile:

        total += 1

        if not line.strip():
            continue

        try:
            feature = json.loads(line)

        except json.JSONDecodeError:

            invalid += 1
            continue

        # ----------------------------------------------------
        # Get geometry
        # ----------------------------------------------------

        geometry = feature.get(
            "geometry"
        )

        if not geometry:
            continue

        geom_type = geometry.get(
            "type"
        )

        coordinates = geometry.get(
            "coordinates"
        )

        if coordinates is None:
            continue

        # ----------------------------------------------------
        # Recursively extract coordinate pairs
        # ----------------------------------------------------

        def extract_points(coords):

            if (
                isinstance(coords, list)
                and len(coords) >= 2
                and isinstance(coords[0], (int, float))
                and isinstance(coords[1], (int, float))
            ):

                yield coords[0], coords[1]

            elif isinstance(coords, list):

                for item in coords:

                    yield from extract_points(
                        item
                    )

        # ----------------------------------------------------
        # Check whether geometry intersects AOI
        # ----------------------------------------------------

        intersects = False

        for lon, lat in extract_points(
            coordinates
        ):

            if (
                MIN_LON <= lon <= MAX_LON
                and
                MIN_LAT <= lat <= MAX_LAT
            ):

                intersects = True
                break

        if intersects:

            outfile.write(
                json.dumps(
                    feature,
                    ensure_ascii=False
                )
                + "\n"
            )

            matched += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if total % 100000 == 0:

            print(
                f"Processed: {total:,} | "
                f"Balaghat matches: {matched:,}"
            )

print("\n" + "=" * 60)
print("EXTRACTION COMPLETE")
print("=" * 60)

print(
    f"Total records processed : {total:,}"
)

print(
    f"Balaghat records         : {matched:,}"
)

print(
    f"Invalid records          : {invalid:,}"
)

print("\nSaved:")
print(OUTPUT)