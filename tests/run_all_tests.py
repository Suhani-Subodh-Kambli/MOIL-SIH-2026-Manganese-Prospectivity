"""
tests/run_all_tests.py

Master Test Suite for MOIL SIH 2026 AI Exploration & Production Intelligence:
1. GeologyIndexer (Hierarchical NGDR 1:2M + 1:50k)
2. ProspectivityFeatureExtractor (Zero-leakage numeric + categorical)
3. SpatiallyBalancedSampler (PU learning, buffer >= 3km, spatial blocks)
4. UnifiedProspectivityEngine (Decoupled & Multi-Model)
5. National Multi-Belt Model (Macro-lithology, Cratons, National Models)
6. Production Shortfall Intelligence (Prediction, Risk, Ore Blending LP, Actions)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_geology_index import test_geology_indexer
from tests.test_feature_extractor import test_feature_extractor
from tests.test_sampling import test_sampler
from tests.test_inference_engine import test_inference_engine
from tests.test_national_model import test_national_suite
from tests.test_production_shortfall import test_production_shortfall_suite


def run_all():
    print("=" * 75)
    print("RUNNING ALL MOIL SIH 2026 TEST SUITES (PHASES 1, 2, 3, & 4)")
    print("=" * 75)

    print("\n>>> [1/6] Testing GeologyIndexer...")
    test_geology_indexer()

    print("\n>>> [2/6] Testing ProspectivityFeatureExtractor...")
    test_feature_extractor()

    print("\n>>> [3/6] Testing SpatiallyBalancedSampler...")
    test_sampler()

    print("\n>>> [4/6] Testing UnifiedProspectivityEngine (Decoupled & Multi-Model)...")
    test_inference_engine()

    print("\n>>> [5/6] Testing National Multi-Belt Models & Macro-Lithology...")
    test_national_suite()

    print("\n>>> [6/6] Testing Production Shortfall & Ore Blending Intelligence...")
    test_production_shortfall_suite()

    print("\n" + "=" * 75)
    print("ALL 6 TEST SUITES PASSED SUCCESSFULLY! (6/6, 100%)")
    print("=" * 75)


if __name__ == "__main__":
    run_all()
