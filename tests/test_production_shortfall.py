"""
tests/test_production_shortfall.py

Unit tests for Phase 4: Production Shortfall Intelligence:
1. ShortfallPredictor (Prediction of extraction deficit, risk tiers, and grade dilution)
2. RootCauseAttributor (Attribution of loss factors across monsoon, fleet, stripping, shaft)
3. OreBlendingOptimizer (Linear programming solution for multi-stockpile blending)
4. CorrectiveActionEngine (Prescriptive prioritized operational interventions)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.production_shortfall import (
    get_shortfall_predictor,
    ShortfallPredictor,
    OreBlendingOptimizer,
    CorrectiveActionEngine
)


def test_shortfall_predictor_baseline():
    print("Testing ShortfallPredictor under dry-season baseline conditions...")
    predictor = get_shortfall_predictor()

    # Dry season, high fleet availability on Balaghat
    res = predictor.predict_shortfall(
        mine_name="Balaghat Mine",
        mine_type="Underground",
        target_tonnes=40000,
        rainfall_mm=10.0,
        fleet_availability_pct=88.0,
        unplanned_downtime_hrs=15.0,
        target_grade_pct=44.0,
        actual_grade_pct=43.8
    )

    assert res["predicted_shortfall_pct"] < 20.0
    assert res["risk_tier"] in ["ON TARGET", "MODERATE", "HIGH"]
    assert res["predicted_actual_extraction_tonnes"] > 30000
    assert 0.0 <= res["grade_deficit_pct"] <= 1.5
    print(f"  Dry Season Balaghat: Predicted Shortfall {res['predicted_shortfall_pct']:.1f}% ({res['risk_tier']})")


def test_shortfall_monsoon_opencast():
    print("Testing ShortfallPredictor under extreme monsoon opencast conditions...")
    predictor = get_shortfall_predictor()

    # Dongri Buzurg opencast under severe monsoon inundation (380 mm)
    res = predictor.predict_shortfall(
        mine_name="Dongri Buzurg Mine",
        mine_type="Opencast",
        target_tonnes=32000,
        rainfall_mm=380.0,
        fleet_availability_pct=68.0,
        unplanned_downtime_hrs=75.0,
        target_grade_pct=41.0,
        actual_grade_pct=38.5,
        stripping_ratio_deficit=0.8
    )

    assert res["predicted_shortfall_pct"] >= 20.0
    assert res["risk_tier"] in ["HIGH", "CRITICAL"]
    assert "Monsoon" in res["primary_root_cause"] or "Fleet" in res["primary_root_cause"]
    assert res["grade_deficit_probability"] > 0.40

    # Verify Root Cause Attribution
    rc = res["root_cause_attribution"]
    assert abs(sum(rc.values()) - 100.0) < 1.0, f"Root causes must sum to ~100%, got {sum(rc.values())}"
    print(f"  Monsoon Dongri Buzurg: Predicted Shortfall {res['predicted_shortfall_pct']:.1f}% ({res['risk_tier']})")
    print(f"  Top Cause: {res['primary_root_cause']} ({rc[res['primary_root_cause']]}%)")


def test_ore_blending_optimizer():
    print("Testing OreBlendingOptimizer Linear Programming Solver...")
    optimizer = OreBlendingOptimizer()

    stockpiles = [
        {"name": "Balaghat High Grade Face", "available_tonnes": 8000, "grade_mn": 46.0, "sio2": 6.0, "p": 0.08, "cost_per_ton": 3200},
        {"name": "Tirodi Medium Grade ROM", "available_tonnes": 12000, "grade_mn": 38.0, "sio2": 10.0, "p": 0.12, "cost_per_ton": 2400},
        {"name": "Low-Grade Ferruginous Stockpile", "available_tonnes": 15000, "grade_mn": 31.0, "sio2": 16.0, "p": 0.20, "cost_per_ton": 1500}
    ]

    res = optimizer.optimize_blend(
        stockpiles=stockpiles,
        target_tonnes=10000,
        min_mn_grade=42.0,
        max_sio2=8.5,
        max_p=0.12
    )

    assert res["status"] == "Optimal", f"Solver failed: {res.get('message')}"
    assert res["blended_grade_mn"] >= 41.95, f"Expected >= 42.0% Mn, got {res['blended_grade_mn']}"
    assert res["blended_sio2"] <= 8.55, f"Expected <= 8.5% SiO2, got {res['blended_sio2']}"
    assert res["blended_p"] <= 0.125, f"Expected <= 0.12% P, got {res['blended_p']}"

    total_allocated = sum(item["allocated_tonnes"] for item in res["recipe"])
    assert abs(total_allocated - 10000) < 1.0, f"Expected 10,000 tonnes allocated, got {total_allocated}"

    print(f"  Optimal Blend Found: Blended Mn {res['blended_grade_mn']}%, SiO2 {res['blended_sio2']}%, P {res['blended_p']}%")
    print(f"  Cost Savings: INR {res['cost_savings_vs_pure_highgrade_inr']:,.2f} vs pure high-grade")


def test_corrective_action_engine():
    print("Testing CorrectiveActionEngine Prescriptive Mitigation Actions...")
    engine = CorrectiveActionEngine()

    actions = engine.generate_recommendations(
        mine_name="Dongri Buzurg Mine",
        mine_type="Opencast",
        shortfall_tonnes=9500,
        shortfall_pct=29.6,
        root_causes={
            "Monsoon Flooding / Water Logging": 0.55,
            "HEMM Fleet Breakdown / Unavailability": 0.25,
            "Overburden Stripping Lag": 0.12,
            "Equipment Maintenance Backlog": 0.08
        },
        grade_deficit=2.5
    )

    assert len(actions) >= 3, f"Expected >= 3 actions, got {len(actions)}"
    categories = [a["category"] for a in actions]
    assert any("Dewatering" in c for c in categories)
    assert any("Fleet" in c for c in categories)

    total_rec = sum(a["estimated_recovery_tonnes"] for a in actions)
    assert total_rec > 0
    print(f"  Generated {len(actions)} prioritized interventions. Total recoverable tonnage: {total_rec} tonnes.")


def test_production_shortfall_suite():
    print("=" * 70)
    print("RUNNING PRODUCTION SHORTFALL INTELLIGENCE TEST SUITE (PHASE 4)")
    print("=" * 70)
    test_shortfall_predictor_baseline()
    test_shortfall_monsoon_opencast()
    test_ore_blending_optimizer()
    test_corrective_action_engine()
    print("=" * 70)
    print("ALL PRODUCTION SHORTFALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_production_shortfall_suite()
