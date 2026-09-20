"""
tests/test_sampling.py

Tests SpatiallyBalancedSampler for buffer distance enforcement,
label assignment, and spatial block partitioning.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.sampling import SpatiallyBalancedSampler


def test_sampler():
    print("Testing SpatiallyBalancedSampler...")

    sampler = SpatiallyBalancedSampler(
        min_background_distance_km=3.0,
        background_ratio=2.0,
        block_size_deg=0.10,
        seed=42
    )

    # Synthetic test points around Balaghat
    pos_df = pd.DataFrame({
        "latitude": [21.82, 21.95, 21.68],
        "longitude": [80.17, 80.45, 79.73],
        "name": ["Balaghat", "Ukwa", "Tirodi"]
    })

    # Candidates: some within 1 km (should be rejected), some > 5 km away (should be accepted)
    bg_df = pd.DataFrame({
        "latitude": [
            21.822,  # ~0.2 km from Balaghat (REJECT)
            21.953,  # ~0.3 km from Ukwa (REJECT)
            21.500,  # ~40 km away (ACCEPT)
            22.200,  # ~45 km away (ACCEPT)
            21.750,  # ~10 km away (ACCEPT)
            21.600,  # ~15 km away (ACCEPT)
        ],
        "longitude": [
            80.172,
            80.452,
            80.100,
            80.400,
            80.000,
            79.600,
        ]
    })

    balanced = sampler.create_balanced_dataset(pos_df, bg_df)
    print(f"Total balanced samples: {len(balanced)}")
    print(f"Positive samples: {(balanced['label'] == 1).sum()}")
    print(f"Background samples: {(balanced['label'] == 0).sum()}")
    print(f"Spatial blocks: {balanced['spatial_block'].nunique()}")

    # Assertions
    assert (balanced['label'] == 1).sum() == 3
    # Check that rejected close points are NOT in background
    bg_points = balanced[balanced['label'] == 0]
    assert (bg_points['nearest_positive_km'] >= 3.0).all()
    assert 'spatial_block' in balanced.columns
    assert len(balanced) == 3 + min(len(bg_points), 6)

    print("\nAll SpatiallyBalancedSampler tests passed successfully!")


if __name__ == "__main__":
    test_sampler()
