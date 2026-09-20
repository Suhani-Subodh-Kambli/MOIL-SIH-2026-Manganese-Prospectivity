"""
tests/test_inference_engine.py

Tests UnifiedProspectivityEngine for arbitrary coordinate and polygon
predictions across India, multi-model support (XGBoost, Naive Bayes, Ensemble),
and target zone clustering.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.inference_engine import get_prospectivity_engine


def test_inference_engine():
    print("Testing UnifiedProspectivityEngine...")
    engine = get_prospectivity_engine()

    # 1. Point prediction inside Balaghat Pilot
    print("\n--- [1] Balaghat Pilot Point Query ---")
    pt_bala = engine.predict_point(21.82, 80.17, model_type="ensemble")
    print(f"Balaghat [21.82, 80.17] -> Score: {pt_bala['prospectivity_score']:.1f}, Priority: {pt_bala['priority']}, Signal: {pt_bala['model_signal']}")
    print(f"  Geology: {pt_bala['geology']['group']} | Nearest Site: {pt_bala['nearest_known_manganese_site']['name']} ({pt_bala['nearest_known_manganese_site']['distance_km']:.2f} km)")
    assert 0 <= pt_bala['prospectivity_score'] <= 100
    assert pt_bala['priority'] in ["VERY HIGH", "HIGH", "MODERATE", "LOW", "VERY LOW"]

    # 2. Multi-Model Comparison on Same Point
    print("\n--- [2] Model Comparison (XGBoost vs Naive Bayes vs Ensemble) ---")
    p_xgb = engine.predict_point(21.82, 80.17, model_type="xgboost")
    p_nb = engine.predict_point(21.82, 80.17, model_type="naive_bayes")
    p_ens = engine.predict_point(21.82, 80.17, model_type="ensemble")
    print(f"  XGBoost Score:     {p_xgb['prospectivity_score']:.2f}")
    print(f"  Naive Bayes Score: {p_nb['prospectivity_score']:.2f}")
    print(f"  Ensemble Score:    {p_ens['prospectivity_score']:.2f}")
    assert abs(p_ens['prospectivity_score'] - 0.5 * (p_xgb['prospectivity_score'] + p_nb['prospectivity_score'])) < 1e-4

    # 3. Arbitrary Coordinate Outside Pilot (Sandur, Karnataka)
    print("\n--- [3] Nationwide Arbitrary Point Query (Sandur, Bellary) ---")
    pt_sandur = engine.predict_point(15.08, 76.55, model_type="ensemble")
    print(f"Sandur [15.08, 76.55] -> Score: {pt_sandur['prospectivity_score']:.1f}, Priority: {pt_sandur['priority']}, Resolution: {pt_sandur['resolution']}")
    print(f"  Geology: {pt_sandur['geology']['group']} | Stratigraphy: {pt_sandur['geology']['stratigraphy'][:40]}")
    print(f"  Nearest Site: {pt_sandur['nearest_known_manganese_site']['name']} ({pt_sandur['nearest_known_manganese_site']['distance_km']:.2f} km, Belt: {pt_sandur['nearest_known_manganese_site']['belt']})")
    assert 0 <= pt_sandur['prospectivity_score'] <= 100

    # 4. Arbitrary Coordinate Outside Pilot (Koira / Bonai, Odisha)
    print("\n--- [4] Nationwide Arbitrary Point Query (Koira, Odisha) ---")
    pt_koira = engine.predict_point(21.90, 85.25, model_type="ensemble")
    print(f"Koira [21.90, 85.25] -> Score: {pt_koira['prospectivity_score']:.1f}, Priority: {pt_koira['priority']}, Resolution: {pt_koira['resolution']}")
    print(f"  Geology: {pt_koira['geology']['group']} | Nearest Site: {pt_koira['nearest_known_manganese_site']['name']} ({pt_koira['nearest_known_manganese_site']['distance_km']:.2f} km)")
    assert 0 <= pt_koira['prospectivity_score'] <= 100

    # 5. Polygon AOI Analysis (Outside Balaghat — Sandur Region Polygon)
    print("\n--- [5] Arbitrary Polygon Analysis (Sandur Belt, Karnataka) ---")
    sandur_poly = [
        (76.50, 15.00),
        (76.65, 15.00),
        (76.65, 15.15),
        (76.50, 15.15),
        (76.50, 15.00)
    ]
    area_res = engine.predict_area(sandur_poly, resolution_km=3.0, model_type="ensemble")
    print(f"Sandur AOI Analysis -> Cells: {area_res['cell_count']}, Mean Score: {area_res['mean_score']:.1f}, Max: {area_res['maximum_score']:.1f}")
    print(f"  High-Priority Cells (>=60): {area_res['high_priority_cells']}, Very High: {area_res['very_high_priority_cells']}")
    print(f"  Target Zones Identified: {area_res['target_zones_count']}")
    assert area_res['cell_count'] > 0
    assert 0 <= area_res['mean_score'] <= 100

    print("\nAll UnifiedProspectivityEngine tests passed successfully!")


if __name__ == "__main__":
    test_inference_engine()

