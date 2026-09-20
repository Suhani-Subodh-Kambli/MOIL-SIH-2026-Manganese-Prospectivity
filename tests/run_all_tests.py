"""
tests/run_all_tests.py

Runs all Phase 1, Phase 2, and Phase 3 test suites:
1. GeologyIndexer (Hierarchical NGDR 1:2M + 1:50k)
2. ProspectivityFeatureExtractor (Zero-leakage numeric + categorical)
3. SpatiallyBalancedSampler (PU learning, buffer >= 3km, spatial blocks)
4. UnifiedProspectivityEngine (Decoupled & Multi-Model)
5. National Multi-Belt Model (Macro-lithology, Cratons, National Models)
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


def run_all():
    print("=" * 75)
    print("RUNNING ALL MOIL SIH 2026 TEST SUITES (PHASES 1, 2, & 3)")
    print("=" * 75)

    print("\n>>> [1/5] Testing GeologyIndexer...")
    test_geology_indexer()

    print("\n>>> [2/5] Testing ProspectivityFeatureExtractor...")
    test_feature_extractor()

    print("\n>>> [3/5] Testing SpatiallyBalancedSampler...")
    test_sampler()

    print("\n>>> [4/5] Testing UnifiedProspectivityEngine (Decoupled & Multi-Model)...")
    test_inference_engine()

    print("\n>>> [5/5] Testing National Multi-Belt Models & Macro-Lithology...")
    test_national_suite()

    print("\n" + "=" * 75)
    print("ALL TEST SUITES PASSED SUCCESSFULLY! (5/5)")
    print("=" * 75)


if __name__ == "__main__":
    run_all()
