"""
tests/test_geology_index.py

Tests GeologyIndexer spatial lookups and structural proxy calculations
across multiple Indian manganese metallogenic belts.
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.geology_index import get_geology_indexer


def test_geology_indexer():
    print("Testing GeologyIndexer across India...")
    indexer = get_geology_indexer()

    test_points = [
        ("Balaghat Pilot (MP)", 21.82, 80.17),
        ("Sandur Belt (Karnataka)", 15.08, 76.55),
        ("Bonai-Keonjhar Belt (Odisha)", 21.90, 85.25),
        ("Colamba Deposit (Goa)", 15.13, 74.12),
        ("Vizianagaram (Andhra Pradesh)", 18.27, 83.55)
    ]

    for name, lat, lon in test_points:
        t0 = time.time()
        geo = indexer.query_geology(lat, lon)
        struct = indexer.query_structural_features(lat, lon)
        elapsed_ms = (time.time() - t0) * 1000.0

        print(f"\n=== {name} [{lat}, {lon}] ({elapsed_ms:.2f} ms) ===")
        print(f"  Source Resolution: {geo['resolution']}")
        print(f"  Geological Group: {geo['geo_group']}")
        print(f"  Stratigraphy: {geo['geo_stratigraphy'][:50]}")
        print(f"  Boundary Distance: {struct['Geo_Boundary_Distance_km']:.3f} km")
        print(f"  Boundary Density (1km/3km): {struct['Geo_Boundary_Density_1km']} / {struct['Geo_Boundary_Density_3km']}")
        print(f"  Lithology Diversity (3km): {struct['Lithology_Diversity_3km']}")
        print(f"  Proxies -> Sausar: {struct['Sausar_Group_Proxy']}, Metamorphic: {struct['Metamorphic_Host_Proxy']}, Unknown: {struct['Geology_Unknown']}")

        # Assertions
        assert "Geo_Boundary_Distance_km" in struct
        assert struct["Geo_Boundary_Distance_km"] >= 0.0
        assert "geo_group" in geo
        assert geo["geo_group"] != ""

    print("\nAll GeologyIndexer tests passed successfully!")


if __name__ == "__main__":
    test_geology_indexer()
