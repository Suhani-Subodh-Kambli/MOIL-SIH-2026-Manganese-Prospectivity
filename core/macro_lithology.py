"""
core/macro_lithology.py

Scientific Macro-Lithological & Metallogenic Domain Classification Engine
for India-Wide Manganese Prospectivity Modeling.

Generalizes regional formation strings and lithologies from NGDR (1:2M and 1:50k)
and GSI mineral inventories into standardized macro-lithological setting classes
and cratonic domains based on Geological Survey of India (GSI) & IBM metallogenic
frameworks.
"""

from typing import Dict, Any, Tuple, Optional
import re
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# STANDARDIZED CLASSES
# ----------------------------------------------------------------------

MACRO_LITHOLOGY_CLASSES = [
    "GONDITE_METAMORPHIC",       # Sausar, Gangpur, Aravalli (Gondite, Pelitic Schist, Braunite)
    "BIF_GREENSTONE",            # Dharwar: Sandur, Chitradurga, Bababudan (BHQ, BMQ, Greenstones)
    "SHALY_TUFFACEOUS_IOG",      # Singhbhum: Bonai-Keonjhar, Koira, Noamundi (Tuffaceous Shale, IOG)
    "GRANULITE_KHONDALITE",      # Eastern Ghats: Vizianagaram, Koraput (Khondalite, Kodurite, Charnockite)
    "LATERITE_WEATHERING",       # Goa, North Kanara (Supergene Lateritic enrichment caps)
    "CARBONATE_SEDIMENTARY",     # Penganga, Pakhal, Bhima, Cuddapah (Limestone, Dolomite, Chert)
    "CRATONIC_BASEMENT",         # Peninsular Gneiss, Tirodi Gneiss, TTG, Granitoids
    "OTHER_UNDIVIDED"            # Basalts, Alluvium, Unclassified
]

CRATON_DOMAINS = [
    "BASTAR_CRATON",             # MP & Maharashtra (Sausar Mn Belt, Tirodi, Chilpi)
    "DHARWAR_CRATON",            # Karnataka (Sandur, Chitradurga, Shimoga)
    "SINGHBHUM_CRATON",          # Odisha & Jharkhand (Bonai-Keonjhar, Noamundi, Gangpur)
    "EASTERN_GHATS_MOBILE_BELT", # Andhra Pradesh & South Odisha (Vizianagaram, Srikakulam)
    "WESTERN_COAST_GOA",         # Goa & North Kanara
    "ARAVALLI_CRATON",           # Gujarat & Rajasthan (Panchmahals, Banswara)
    "PRANHITA_GODAVARI_BASIN",   # Telangana & Maharashtra border (Penganga, Adilabad)
    "OTHER_DOMAIN"               # Undivided Indian domains
]

# Metallogenic Host Affinity Score (0.0 to 1.0)
# Calibrated empirical likelihood of hosting economic Mn mineralization
AFFINITY_WEIGHTS: Dict[str, float] = {
    "GONDITE_METAMORPHIC": 0.95,
    "SHALY_TUFFACEOUS_IOG": 0.92,
    "BIF_GREENSTONE": 0.88,
    "GRANULITE_KHONDALITE": 0.85,
    "LATERITE_WEATHERING": 0.78,
    "CARBONATE_SEDIMENTARY": 0.70,
    "CRATONIC_BASEMENT": 0.30,
    "OTHER_UNDIVIDED": 0.10
}


# ----------------------------------------------------------------------
# CLASSIFICATION LOGIC
# ----------------------------------------------------------------------

def classify_macro_lithology(
    geo_group: Optional[str] = None,
    geo_formation: Optional[str] = None,
    geo_lithology: Optional[str] = None,
    geo_stratigraphy: Optional[str] = None,
    host_rock: Optional[str] = None
) -> Tuple[str, float]:
    """
    Classifies raw geological attributes into one of the 8 macro-lithology classes
    and returns (macro_lithology_class, host_affinity_score).
    """
    text = " ".join([
        str(geo_group or ""),
        str(geo_formation or ""),
        str(geo_lithology or ""),
        str(geo_stratigraphy or ""),
        str(host_rock or "")
    ]).upper()

    # 1. Laterite / Weathering Cap (Supergene enrichment)
    if re.search(r"\b(LATERITE|FERRICRETE|BAUXITE|SUPERGENE)\b", text):
        return "LATERITE_WEATHERING", AFFINITY_WEIGHTS["LATERITE_WEATHERING"]

    # 2. Gondite / Sausar Pelitic Metamorphic
    if re.search(r"\b(GONDITE|SAUSAR|MANSAR|CHORBAOLI|LOHANGI|BICHUA|SITASAONGI|CHILPI|GANGPUR|LUNAVADA|CHAMPANER|ARAVALLI)\b", text):
        return "GONDITE_METAMORPHIC", AFFINITY_WEIGHTS["GONDITE_METAMORPHIC"]

    # 3. Shaly-Tuffaceous IOG (Singhbhum / Bonai-Keonjhar / Koira)
    if re.search(r"\b(BONAI|LOWER BONAI|IRON ORE GROUP|IOG|KOIRA|NOAMUNDI|JAMDA|TUFFACEOUS SHALE|MANGANIFEROUS SHALE)\b", text):
        return "SHALY_TUFFACEOUS_IOG", AFFINITY_WEIGHTS["SHALY_TUFFACEOUS_IOG"]

    # 4. Granulite / Khondalite / Kodurite (Eastern Ghats)
    if re.search(r"\b(KHONDALITE|KODURITE|EASTERN GHAT|CHARNOCKITE|MIGMATITE|GARNET-SILLIMANITE)\b", text):
        return "GRANULITE_KHONDALITE", AFFINITY_WEIGHTS["GRANULITE_KHONDALITE"]

    # 5. BIF / Greenstone / Dharwar (Sandur, Chitradurga, Bababudan)
    if re.search(r"\b(SANDUR|BABABUDAN|CHITRADURGA|DHARWAR|SHIMOGA|BHQ|BMQ|BHJ|METAVOLCANIC|BANDED IRON|BANDED MAGNETITE)\b", text):
        return "BIF_GREENSTONE", AFFINITY_WEIGHTS["BIF_GREENSTONE"]

    # 6. Carbonate Sedimentary Platform (Penganga, Pakhal, Bhima, Cuddapah)
    if re.search(r"\b(PENGANGA|PAKHAL|BHIMA|KURNOOL|CUDDAPAH|LIMESTONE|DOLOMITE|MARBLE|CALC-SILICATE)\b", text):
        return "CARBONATE_SEDIMENTARY", AFFINITY_WEIGHTS["CARBONATE_SEDIMENTARY"]

    # 7. Pelitic / Metamorphic schists/phyllites not caught by group names
    if re.search(r"\b(MANGANIFEROUS PHYLLITE|PHYLLITE|SCHIST|MICA SCHIST|QUARTZITE)\b", text):
        # Default to metamorphic setting
        return "GONDITE_METAMORPHIC", 0.75

    # 8. Cratonic Basement / Gneiss / Granitoids
    if re.search(r"\b(TIRODI|PENINSULAR GNEISS|GNEISS|GRANITE|GRANITOID|AMGAON|BASEMENT|TTG)\b", text):
        return "CRATONIC_BASEMENT", AFFINITY_WEIGHTS["CRATONIC_BASEMENT"]

    # 9. Default Undivided
    return "OTHER_UNDIVIDED", AFFINITY_WEIGHTS["OTHER_UNDIVIDED"]


def classify_craton_domain(
    lat: float,
    lon: float,
    state: Optional[str] = None,
    belt: Optional[str] = None
) -> str:
    """
    Classifies a coordinate into its regional Cratonic Domain for out-of-craton validation.
    """
    state_str = str(state or "").upper()
    belt_str = str(belt or "").upper()

    if "SAUSAR" in belt_str or ("MADHYA PRADESH" in state_str and lat >= 21.0) or ("MAHARASHTRA" in state_str and lon >= 78.5):
        return "BASTAR_CRATON"

    if "SANDUR" in belt_str or "CHITRADURGA" in belt_str or "SHIMOGA" in belt_str or "KARNATAKA" in state_str:
        return "DHARWAR_CRATON"

    if "BONAI" in belt_str or "NOAMUNDI" in belt_str or "JAMDA" in belt_str or ("ODISHA" in state_str and lat >= 21.3) or "JHARKHAND" in state_str:
        return "SINGHBHUM_CRATON"

    if "EASTERNGHAT" in belt_str or "EASTERN GHAT" in belt_str or "ANDHRA PRADESH" in state_str or ("ODISHA" in state_str and lat < 20.5 and lon >= 82.5):
        return "EASTERN_GHATS_MOBILE_BELT"

    if "GOA" in belt_str or "GOA" in state_str or (lat >= 14.8 and lat <= 15.9 and lon <= 74.5):
        return "WESTERN_COAST_GOA"

    if "GUJARAT" in state_str or "RAJASTHAN" in state_str or (lon <= 74.5 and lat >= 22.0):
        return "ARAVALLI_CRATON"

    if "PENGANGA" in belt_str or "TELANGANA" in state_str or (lat >= 19.0 and lat <= 20.2 and lon >= 78.0 and lon <= 79.5):
        return "PRANHITA_GODAVARI_BASIN"

    # Geometric heuristic fallback based on Indian cratonic layout
    if 21.0 <= lat <= 22.8 and 78.5 <= lon <= 81.5:
        return "BASTAR_CRATON"
    elif 13.0 <= lat <= 16.0 and 75.0 <= lon <= 77.5:
        return "DHARWAR_CRATON"
    elif 21.3 <= lat <= 23.0 and 84.5 <= lon <= 86.5:
        return "SINGHBHUM_CRATON"
    elif 17.5 <= lat <= 19.8 and 82.5 <= lon <= 84.5:
        return "EASTERN_GHATS_MOBILE_BELT"

    return "OTHER_DOMAIN"


def enrich_with_macro_lithology(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a dataframe containing geological columns and lat/lon,
    and appends:
    - 'macro_lithology'
    - 'metallogenic_host_affinity'
    - 'craton_domain'
    """
    df = df.copy()

    macro_liths = []
    affinities = []
    cratons = []

    for _, row in df.iterrows():
        lat = float(row.get("latitude", row.get("lat", 0.0)))
        lon = float(row.get("longitude", row.get("lon", 0.0)))

        group = row.get("geo_group", row.get("geology_group", None))
        formation = row.get("geo_formation", row.get("formation", None))
        lithology = row.get("geo_lithology", row.get("host_rock", None))
        strat = row.get("geo_stratigraphy", None)
        host = row.get("host_rock", None)
        state = row.get("state", None)
        belt = row.get("metallogenic_belt", None)

        m_class, aff = classify_macro_lithology(
            geo_group=group,
            geo_formation=formation,
            geo_lithology=lithology,
            geo_stratigraphy=strat,
            host_rock=host
        )
        c_dom = classify_craton_domain(lat, lon, state=state, belt=belt)

        macro_liths.append(m_class)
        affinities.append(aff)
        cratons.append(c_dom)

    df["macro_lithology"] = macro_liths
    df["metallogenic_host_affinity"] = affinities
    df["craton_domain"] = cratons

    return df

