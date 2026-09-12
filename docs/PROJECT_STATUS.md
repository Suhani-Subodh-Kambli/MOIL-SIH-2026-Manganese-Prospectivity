# Current Project Status

## Completed

Phase 1
- Sentinel-2 pipeline

Phase 2
- GSI manganese occurrences

Phase 3
- Initial multimodal feature dataset

Phase 4B.1
- NGDR geological/lithological data

Phase 4B.2
- Geological boundary and structural features

Phase 4B.3
- Spectral features

Phase 4B.4
- Spatially balanced sampling

Phase 4B.5
- XGBoost

Phase 4B.6
- 5-fold spatial validation

Phase 4B.7
- SHAP explainability

Phase 5
- Balaghat prospectivity mapping

## Current model

117 transformed features

5-fold spatial ROC-AUC:
0.6210 ± 0.2085

5-fold spatial PR-AUC:
0.4746 ± 0.1422

## Current Phase 5

Prediction cells:
14,061

Target cells:
1,407

Target zones:
94

## Important interpretation

Prospectivity is a relative exploration-priority ranking.

It does not represent:
- ore grade
- manganese concentration
- proven reserve
- economic extraction probability

## Next task

Phase 6:
Production shortfall prediction.

## Do not change without discussion

The current exploration architecture:

GSI occurrences
+
Sentinel-2
+
Sentinel-1
+
SRTM
+
Geological evidence
↓
Feature dataset
↓
XGBoost
↓
Spatial validation
↓
Prospectivity score
↓
Target zones
↓
Dashboard