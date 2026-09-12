import json
from collections import Counter

FILE = (
    "data/geology/balaghat/"
    "balaghat_lithology.geojsonl"
)

MAX_RECORDS = 5000

print("=" * 60)
print("BALAGHAT NGDR ATTRIBUTE INSPECTION")
print("=" * 60)

attribute_values = {}
geometry_types = Counter()

count = 0

with open(
    FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        if not line.strip():
            continue

        feature = json.loads(line)

        count += 1

        # ----------------------------------------------------
        # Geometry
        # ----------------------------------------------------

        geometry = feature.get(
            "geometry"
        )

        if geometry:

            geometry_types[
                geometry.get("type")
            ] += 1

        # ----------------------------------------------------
        # Properties
        # ----------------------------------------------------

        properties = feature.get(
            "properties",
            {}
        )

        for key, value in properties.items():

            if key not in attribute_values:

                attribute_values[key] = Counter()

            if value is not None:

                attribute_values[key][
                    str(value)
                ] += 1

        if count >= MAX_RECORDS:
            break

print("\nRecords inspected:", count)

print("\n" + "=" * 60)
print("GEOMETRY TYPES")
print("=" * 60)

for key, value in geometry_types.items():

    print(
        f"{key}: {value}"
    )

print("\n" + "=" * 60)
print("ATTRIBUTE COLUMNS")
print("=" * 60)

for key in attribute_values:

    print(
        "\n---",
        key,
        "---"
    )

    for value, frequency in (
        attribute_values[key]
        .most_common(20)
    ):

        print(
            f"{value} : {frequency}"
        )