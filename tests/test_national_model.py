"""
tests/test_national_model.py

Unit and integration tests for Phase 3:
1. Macro-Lithological & Craton Domain Classification Engine
2. National Multi-Belt Dataset Integrity & Leakage Verification
3. National Preprocessor & Model Artifacts
4. National Prospectivity Inference Engine across India's Major Belts
5. Multi-Model Selection (XGBoost, Naive Bayes, Ensemble)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import joblib
import numpy as np
import pandas as pd

from core.macro_lithology import (
    classify_macro_lithology,
    classify_craton_domain,
    enrich_with_macro_lithology,
    MACRO_LITHOLOGY_CLASSES,
    CRATON_DOMAINS
)
from core.inference_engine import get_prospectivity_engine, UnifiedProspectivityEngine


def test_macro_lithology_classifier():
    print("Testing Macro-Lithology Classifier...")

    # 1. Gondite Metamorphic (Sausar, MP/MH)
    m, aff = classify_macro_lithology(geo_group="SAUSAR", geo_lithology="SCHIST", host_rock="GONDITE")
    assert m == "GONDITE_METAMORPHIC", f"Expected GONDITE_METAMORPHIC, got {m}"
    assert aff >= 0.80

    # 2. BIF Greenstone (Sandur, Dharwar, Karnataka)
    m, aff = classify_macro_lithology(geo_group="SANDUR", geo_lithology="BHQ", host_rock="BANDED IRON")
    assert m == "BIF_GREENSTONE", f"Expected BIF_GREENSTONE, got {m}"
    assert aff >= 0.80

    # 3. Shaly-Tuffaceous IOG (Bonai-Keonjhar, Odisha)
    m, aff = classify_macro_lithology(geo_group="LOWER BONAI", geo_lithology="SHALE", host_rock="TUFFACEOUS SHALE")
    assert m == "SHALY_TUFFACEOUS_IOG", f"Expected SHALY_TUFFACEOUS_IOG, got {m}"
    assert aff >= 0.80

    # 4. Granulite Khondalite (Eastern Ghats, AP)
    m, aff = classify_macro_lithology(geo_group="MIGMATITE", geo_lithology="GNEISS", host_rock="KHONDALITE")
    assert m == "GRANULITE_KHONDALITE", f"Expected GRANULITE_KHONDALITE, got {m}"
    assert aff >= 0.80

    # 5. Laterite Weathering (Goa)
    m, aff = classify_macro_lithology(geo_group="GOA", geo_lithology="LATERITE", host_rock="LATERITE")
    assert m == "LATERITE_WEATHERING", f"Expected LATERITE_WEATHERING, got {m}"
    assert aff >= 0.70

    # 6. Carbonate Sedimentary (Penganga, Adilabad)
    m, aff = classify_macro_lithology(geo_group="PENGANGA", geo_lithology="LIMESTONE")
    assert m == "CARBONATE_SEDIMENTARY", f"Expected CARBONATE_SEDIMENTARY, got {m}"
    assert aff >= 0.60

    # 7. Cratonic Basement
    m, aff = classify_macro_lithology(geo_group="TIRODI GNEISSIC COMPLEX", geo_lithology="GRANITE")
    assert m == "CRATONIC_BASEMENT", f"Expected CRATONIC_BASEMENT, got {m}"
    assert aff <= 0.40

    print("  Macro-lithology classifier passed for all 7 lithological settings!")


def test_craton_domain_classifier():
    print("Testing Craton Domain Classifier...")

    # Bastar Craton (Balaghat)
    c = classify_craton_domain(21.82, 80.17, state="Madhya Pradesh", belt="SAUSAR Mn ZONE")
    assert c == "BASTAR_CRATON", f"Expected BASTAR_CRATON, got {c}"

    # Dharwar Craton (Sandur)
    c = classify_craton_domain(15.08, 76.55, state="Karnataka", belt="SANDUR Fe-Mn-Au PROVINCE")
    assert c == "DHARWAR_CRATON", f"Expected DHARWAR_CRATON, got {c}"

    # Singhbhum Craton (Koira)
    c = classify_craton_domain(21.90, 85.25, state="Odisha", belt="BONAI-NOAMUNDI-JAMDA Fe-Mn BELT")
    assert c == "SINGHBHUM_CRATON", f"Expected SINGHBHUM_CRATON, got {c}"

    # Eastern Ghats (Vizianagaram)
    c = classify_craton_domain(18.27, 83.55, state="Andhra Pradesh", belt="EASTERNGHAT AI-Mn PROVINCE")
    assert c == "EASTERN_GHATS_MOBILE_BELT", f"Expected EASTERN_GHATS_MOBILE_BELT, got {c}"

    # Western Coast (Goa)
    c = classify_craton_domain(15.13, 74.12, state="Goa", belt="GOA Fe-Mn BELT")
    assert c == "WESTERN_COAST_GOA", f"Expected WESTERN_COAST_GOA, got {c}"

    print("  Craton domain classifier passed for all cratonic provinces!")


def test_national_dataset():
    print("Testing National Multi-Belt Dataset Integrity...")
    data_path = PROJECT_ROOT / "data" / "MOIL_National_MultiBelt_Dataset.csv"
    assert data_path.exists(), f"Missing dataset: {data_path}"

    df = pd.read_csv(data_path)
    assert len(df) >= 700, f"Expected >= 700 samples, got {len(df)}"
    assert "label" in df.columns
    assert "craton_domain" in df.columns
    assert "macro_lithology" in df.columns
    assert "spatial_block" in df.columns
    assert "metallogenic_host_affinity" in df.columns

    # Strict Leakage Checks
    assert "Distance_Mn" not in df.columns, "LEAKAGE: Distance_Mn present in dataset!"
    assert df["label"].nunique() == 2, "Dataset must have both positive (1) and background (0) labels"

    pos_count = (df["label"] == 1).sum()
    bg_count = (df["label"] == 0).sum()
    print(f"  Dataset verified: {len(df)} samples ({pos_count} positives, {bg_count} background). Zero leakage!")


def test_national_model_inference():
    print("Testing National Model Inference across Major Belts...")
    engine = get_prospectivity_engine()

    assert engine.national_preprocessor is not None, "Missing national preprocessor!"
    assert engine.national_xgb is not None, "Missing national XGBoost model!"
    assert engine.national_nb is not None, "Missing national Naive Bayes model!"

    test_locations = [
        ("Sandur (Karnataka)", 15.08, 76.55, "BIF_GREENSTONE", "DHARWAR_CRATON"),
        ("Koira (Odisha)", 21.90, 85.25, "SHALY_TUFFACEOUS_IOG", "SINGHBHUM_CRATON"),
        ("Vizianagaram (AP)", 18.27, 83.55, "GRANULITE_KHONDALITE", "EASTERN_GHATS_MOBILE_BELT"),
        ("Colamba (Goa)", 15.13, 74.12, "BIF_GREENSTONE", "WESTERN_COAST_GOA"),
        ("Balaghat (MP)", 21.82, 80.17, "GONDITE_METAMORPHIC", "BASTAR_CRATON")
    ]

    for name, lat, lon, exp_lith, exp_craton in test_locations:
        res = engine.predict_point(lat, lon, pipeline="national", model_type="ensemble")

        assert 0.0 <= res["prospectivity_score"] <= 100.0
        assert 0.0 <= res["model_probability"] <= 1.0
        assert res["priority"] in ["VERY HIGH", "HIGH", "MODERATE", "LOW", "VERY LOW"]
        assert res["pipeline"] == "NATIONAL"
        assert res["macro_lithology"] == exp_lith
        assert res["craton_domain"] == exp_craton

        print(f"  {name:25s} -> Score: {res['prospectivity_score']:.1f} ({res['priority']}) | "
              f"Lith: {res['macro_lithology']} | Craton: {res['craton_domain']}")

    # Multi-model selection test
    res_xgb = engine.predict_point(15.08, 76.55, pipeline="national", model_type="xgboost")
    res_nb = engine.predict_point(15.08, 76.55, pipeline="national", model_type="naive_bayes")
    res_ens = engine.predict_point(15.08, 76.55, pipeline="national", model_type="ensemble")

    assert res_xgb["model_type"] == "XGBOOST"
    assert res_nb["model_type"] == "NAIVE_BAYES"
    assert res_ens["model_type"] == "ENSEMBLE"

    # Verify ensemble is within reasonable convex bounds of individual models
    min_prob = min(res_xgb["model_probability"], res_nb["model_probability"])
    max_prob = max(res_xgb["model_probability"], res_nb["model_probability"])
    assert min_prob - 1e-5 <= res_ens["model_probability"] <= max_prob + 1e-5

    print(f"  Multi-model evaluation passed (XGB: {res_xgb['prospectivity_score']:.1f}, "
          f"NB: {res_nb['prospectivity_score']:.1f}, Ens: {res_ens['prospectivity_score']:.1f})")


def test_national_area_prediction():
    print("Testing National Area Prediction with Adaptive Grid...")
    engine = get_prospectivity_engine()

    # Sandur Fe-Mn Belt AOI
    sandur_polygon = [
        (76.45, 15.00),
        (76.60, 15.00),
        (76.60, 15.15),
        (76.45, 15.15),
        (76.45, 15.00)
    ]

    res = engine.predict_area(sandur_polygon, resolution_km=3.0, model_type="ensemble", pipeline="national")
    assert res["cell_count"] > 0
    assert 0.0 <= res["mean_score"] <= 100.0
    assert res["pipeline"] == "NATIONAL"
    print(f"  Area analysis passed: {res['cell_count']} cells evaluated (Mean: {res['mean_score']:.1f}, Max: {res['maximum_score']:.1f})")


def test_national_suite():
    print("=" * 70)
    print("RUNNING NATIONAL MULTI-BELT MODEL TEST SUITE (PHASE 3)")
    print("=" * 70)
    test_macro_lithology_classifier()
    test_craton_domain_classifier()
    test_national_dataset()
    test_national_model_inference()
    test_national_area_prediction()
    print("=" * 70)
    print("ALL NATIONAL MULTI-BELT MODEL TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_national_suite()
