"""
tests/run_all_tests.py

Runs all Phase 1 test suites.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_geology_index import test_geology_indexer
from tests.test_feature_extractor import test_feature_extractor
from tests.test_sampling import test_sampler


def run_all():
    print("=" * 75)
    print("RUNNING ALL MOIL SIH 2026 PHASE 1 TESTS")
    print("=" * 75)

    print("\n>>> [1/3] Testing GeologyIndexer...")
    test_geology_indexer()

    print("\n>>> [2/3] Testing ProspectivityFeatureExtractor...")
    test_feature_extractor()

    print("\n>>> [3/3] Testing SpatiallyBalancedSampler...")
    test_sampler()

    print("\n" + "=" * 75)
    print("ALL PHASE 1 TESTS PASSED SUCCESSFULLY! (3/3)")
    print("=" * 75)


if __name__ == "__main__":
    run_all()
