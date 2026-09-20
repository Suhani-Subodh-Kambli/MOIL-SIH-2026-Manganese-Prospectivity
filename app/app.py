import streamlit as st
import folium
from folium.plugins import Draw
import sys
from pathlib import Path

from folium.plugins import Draw
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_folium import st_folium
import pandas as pd
from prediction_engine import ProspectivityEngine


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="MOIL AI Exploration Intelligence — India-Wide",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 2px;
        color: #1E3A8A;
    }

    .subtitle {
        font-size: 16px;
        opacity: 0.80;
        margin-bottom: 20px;
    }

    .score-card {
        padding: 20px;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.25);
        background: rgba(255,255,255,0.03);
        text-align: center;
        min-height: 125px;
    }

    .score-number {
        font-size: 34px;
        font-weight: 750;
        color: #2563EB;
    }

    .score-label {
        font-size: 13px;
        font-weight: 600;
        opacity: 0.70;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .info-card {
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.20);
        margin-top: 10px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# PRESET LOCATIONS
# =========================================================

PRESETS = {
    "Custom Coordinates": None,
    "Balaghat Mine & Pilot (Madhya Pradesh)": (21.820000, 80.170000),
    "Ukwa Mine (Madhya Pradesh)": (21.966667, 80.466667),
    "Tirodi Mine (Madhya Pradesh)": (21.683333, 79.733333),
    "Dongri Buzurg Mine (Maharashtra)": (21.550000, 79.710000),
    "Sandur Fe-Mn Belt (Bellary, Karnataka)": (15.080000, 76.550000),
    "Koira, Bonai-Keonjhar Belt (Odisha)": (21.900000, 85.250000),
    "Garbham, Eastern Ghats (Andhra Pradesh)": (18.366667, 83.483333),
    "Colamba / Sanvordem (Goa)": (15.133333, 74.116667),
    "Harenaballi, Bababudan (Karnataka)": (13.316667, 76.716667)
}


# =========================================================
# SESSION STATE
# =========================================================

if "point_result" not in st.session_state:
    st.session_state.point_result = None

if "area_result" not in st.session_state:
    st.session_state.area_result = None

if "selected_latitude" not in st.session_state:
    st.session_state.selected_latitude = 21.82

if "selected_longitude" not in st.session_state:
    st.session_state.selected_longitude = 80.17


# =========================================================
# LOAD MODEL / ENGINE
# =========================================================

@st.cache_resource
def load_engine():
    return ProspectivityEngine()


try:
    engine = load_engine()
except Exception as error:
    st.error("Prediction engine could not be loaded.")
    st.code(str(error))
    st.stop()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">⛏️ MOIL AI Exploration Intelligence</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    AI-assisted manganese mineral prospectivity & exploration-priority mapping
    integrating Sentinel-2 optical, Sentinel-1 SAR, SRTM terrain, and National NGDR lithology.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🔎 Navigation")

mode = st.sidebar.radio(
    "Analysis Mode",
    [
        "📍 Single Location",
        "⬡ Select Area",
    ],
)

st.sidebar.markdown("---")

st.sidebar.markdown("### 🤖 Prospectivity Model")

model_choice = st.sidebar.selectbox(
    "Select AI Model",
    [
        "Ensemble (XGBoost + Naive Bayes)",
        "XGBoost (Phase 4B)",
        "Naive Bayes (GaussianNB)"
    ],
    index=0
)

model_type_map = {
    "Ensemble (XGBoost + Naive Bayes)": "ensemble",
    "XGBoost (Phase 4B)": "xgboost",
    "Naive Bayes (GaussianNB)": "naive_bayes"
}
selected_model_type = model_type_map[model_choice]

st.sidebar.markdown("### 🗺️ Geographic Domain")
st.sidebar.write("🇮🇳 **India-Wide Coverage**")
st.sidebar.caption(
    "Includes Sausar Belt (MP/MH), Dharwar Craton (Karnataka), "
    "Bonai-Keonjhar (Odisha/JH), Eastern Ghats (AP), and Goa."
)

st.sidebar.markdown("### 📊 Target Metric")
st.sidebar.write("**Exploration Priority Score: 0–100**")

st.sidebar.write(
    "Exploration Priority Score: 0–100"
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "The score represents exploration priority based on "
    "the current model and available evidence. It does not "
    "represent measured manganese concentration or a proven reserve."
    "**Scientific Disclaimer**: The 0–100 score is an AI exploration-priority ranking "
    "indicating prospective geologic and remote sensing conditions. It does not represent "
    "drilled ore grade, measured reserves, or proof of underground mineralization."
)


# =========================================================
# SINGLE LOCATION MODE
# =========================================================

if mode == "📍 Single Location":

    st.subheader("📍 Single Location Analysis")

    st.subheader("📍 Single Location Prospectivity Analysis")
    st.write(
        "Enter coordinates inside the current Balaghat pilot area."
        "Enter geographic coordinates anywhere in India or select from key manganese mining hubs."
    )

    preset_name = st.selectbox(
        "📍 Quick-Jump to Manganese Province / Mine:",
        list(PRESETS.keys()),
        index=1
    )

    if preset_name != "Custom Coordinates" and PRESETS[preset_name] is not None:
        p_lat, p_lon = PRESETS[preset_name]
        st.session_state.selected_latitude = p_lat
        st.session_state.selected_longitude = p_lon

    col1, col2 = st.columns(2)

    with col1:

        latitude = st.number_input(
            "Latitude",
            min_value=21.30,
            max_value=22.40,
            "Latitude (°N)",
            min_value=6.00,
            max_value=38.00,
            value=st.session_state.selected_latitude,
            step=0.001,
            format="%.6f",
        )

    with col2:

        longitude = st.number_input(
            "Longitude",
            min_value=79.50,
            max_value=80.80,
            "Longitude (°E)",
            min_value=68.00,
            max_value=98.00,
            value=st.session_state.selected_longitude,
            step=0.001,
            format="%.6f",
        )

    analyse = st.button(
        "🚀 Analyse Prospectivity",
        "🚀 Evaluate Mineral Prospectivity",
        type="primary",
        use_container_width=True,
    )

    # -----------------------------------------------------
    # RUN PREDICTION
    # -----------------------------------------------------

    if analyse:

        try:

            result = engine.predict_point(
                latitude,
                longitude,
            )

            # IMPORTANT:
            # Store result so it survives Streamlit reruns.
            st.session_state.point_result = result

            st.session_state.selected_latitude = latitude
            st.session_state.selected_longitude = longitude

            with st.spinner(f"Evaluating prospectivity using {model_choice}..."):
                result = engine.predict_point(
                    latitude,
                    longitude,
                    model_type=selected_model_type
                )
                st.session_state.point_result = result
                st.session_state.selected_latitude = latitude
                st.session_state.selected_longitude = longitude
        except Exception as error:

            st.error("Prediction failed.")
            st.code(str(error))

    # -----------------------------------------------------
    # DISPLAY STORED RESULT
    # -----------------------------------------------------

    result = st.session_state.point_result

    if result is not None:

        st.markdown("---")

        st.subheader("🧠 Prospectivity Assessment")

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.markdown(
                f"""
                <div class="score-card">
                    <div class="score-label">
                        PROSPECTIVITY SCORE
                    </div>
                    <div class="score-number">
                        {result["prospectivity_score"]:.1f}
                    </div>
                    <div class="score-label">
                        out of 100
                    </div>
                    <div class="score-label">PROSPECTIVITY SCORE</div>
                    <div class="score-number">{result["prospectivity_score"]:.1f}</div>
                    <div class="score-label">out of 100</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:

            st.markdown(
                f"""
                <div class="score-card">
                    <div class="score-label">
                        EXPLORATION PRIORITY
                    </div>
                    <div class="score-number">
                        {result["priority"]}
                    </div>
                    <div class="score-label">EXPLORATION PRIORITY</div>
                    <div class="score-number">{result["priority"]}</div>
                    <div class="score-label">Priority Band</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c3:

            st.markdown(
                f"""
                <div class="score-card">
                    <div class="score-label">
                        GRID CELL DISTANCE
                    </div>
                    <div class="score-number">
                        {result["distance_to_grid_cell_km"]:.2f}
                    </div>
                    <div class="score-label">
                        km
                    </div>
                    <div class="score-label">AI MODEL SIGNAL</div>
                    <div class="score-number" style="font-size: 20px; padding-top: 8px;">{result["model_signal"]}</div>
                    <div class="score-label">Model Signal Strength</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c4:

            st.markdown(
                f"""
                <div class="score-card">
                    <div class="score-label">
                        MODEL SIGNAL
                    </div>
                    <div class="score-number">
                        {result["model_signal"]}
                    </div>
                    <div class="score-label">AI ENGINE USED</div>
                    <div class="score-number" style="font-size: 20px; padding-top: 8px;">{result.get("model_type", "ENSEMBLE")}</div>
                    <div class="score-label">{result.get("resolution", "NGDR Geology")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # -------------------------------------------------
        # GEOLOGICAL & STRUCTURAL DIAGNOSTICS
        # -------------------------------------------------
        st.markdown("---")
        st.subheader("🔬 Geological & Structural Evidence")

        g1, g2 = st.columns(2)

        geo = result.get("geology", {})
        struct = result.get("structural", {})
        nearest = result.get("nearest_known_manganese_site", {})

        with g1:
            st.markdown(
                f"""
                <div class="info-card">
                    <h4>🏛️ Lithological Environment</h4>
                    <p><b>Geological Group:</b> <code>{geo.get("group", "Unknown")}</code></p>
                    <p><b>Formation:</b> <code>{geo.get("formation", "Unknown")}</code></p>
                    <p><b>Lithology / Unit:</b> <code>{geo.get("lithology", "Unknown")}</code></p>
                    <p><b>Stratigraphic Horizon:</b> <code>{geo.get("stratigraphy", "Unknown")}</code></p>
                    <p><b>Data Resolution:</b> {result.get("resolution", "National NGDR")}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with g2:
            st.markdown(
                f"""
                <div class="info-card">
                    <h4>📐 Structural & Proximity Proxies</h4>
                    <p><b>Nearest Geological Contact:</b> <code>{struct.get("boundary_distance_km", 0.0):.2f} km</code></p>
                    <p><b>Contact Density (1 km / 3 km):</b> <code>{struct.get("boundary_density_1km", 0)}</code> / <code>{struct.get("boundary_density_3km", 0)}</code> contacts</p>
                    <p><b>Lithological Diversity (3 km):</b> <code>{struct.get("lithology_diversity_3km", 1)}</code> distinct units</p>
                    <p><b>Metamorphic Host Rock Proxy:</b> {"✅ Matched" if struct.get("metamorphic_host") else "❌ Unmatched"}</p>
                    <p><b>Nearest Known Mn Site:</b> <code>{nearest.get("name", "N/A")}</code> ({nearest.get("distance_km", 0.0):.2f} km)</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # -------------------------------------------------
        # MAP
        # -------------------------------------------------
        st.markdown("---")
        st.subheader("🗺️ Exploration Context Map")

        st.subheader("🗺️ Location on Exploration Map")

        m = folium.Map(
            location=[
                result["latitude"],
                result["longitude"],
            ],
            zoom_start=10,
            location=[result["latitude"], result["longitude"]],
            zoom_start=10 if result.get("distance_to_grid_cell_km", 0) < 5 else 8,
            tiles="OpenStreetMap",
        )

        # Selected Point Marker
        folium.Marker(
            [
                result["latitude"],
                result["longitude"],
            ],
            tooltip="Selected Location",
            [result["latitude"], result["longitude"]],
            tooltip="Target Query Location",
            popup=(
                f"<b>Prospectivity:</b> "
                f"{result['prospectivity_score']:.1f}/100<br>"
                f"<b>Priority:</b> "
                f"{result['priority']}"
                f"<b>Target Query Point</b><br>"
                f"Prospectivity: <b>{result['prospectivity_score']:.1f}/100</b><br>"
                f"Priority: <b>{result['priority']}</b><br>"
                f"Model: {result.get('model_type', 'ENSEMBLE')}<br>"
                f"Group: {geo.get('group', 'Unknown')}"
            ),
            icon=folium.Icon(color="red", icon="crosshairs", prefix="fa")
        ).add_to(m)

        # Add nearby known GSI manganese occurrences
        if engine.engine.occurrences_df is not None:
            occ_df = engine.engine.occurrences_df
            for _, occ in occ_df.head(100).iterrows():
                # Show occurrences within ~100 km
                dlat = abs(occ["latitude"] - result["latitude"])
                dlon = abs(occ["longitude"] - result["longitude"])
                if dlat < 1.5 and dlon < 1.5:
                    folium.CircleMarker(
                        location=[occ["latitude"], occ["longitude"]],
                        radius=5,
                        color="#10B981",
                        fill=True,
                        fill_color="#10B981",
                        fill_opacity=0.8,
                        tooltip=f"GSI Site: {occ['name']} ({occ['state']})",
                        popup=(
                            f"<b>{occ['name']}</b><br>"
                            f"State: {occ['state']}<br>"
                            f"Belt: {occ.get('metallogenic_belt', 'N/A')}<br>"
                            f"Host: {occ.get('host_rock', 'N/A')}<br>"
                            f"Type: {occ.get('deposit_type', 'Deposit')}"
                        )
                    ).add_to(m)

        st_folium(
            m,
            width=None,
            height=500,
            key="single_location_map",
        )

        # -------------------------------------------------
        # EXPLANATION
        # -------------------------------------------------

        st.subheader("🔬 AI Interpretation")

        st.subheader("💡 AI Exploration Summary & Recommendation")
        st.info(
            f"""
            **Selected location**

            Latitude: `{result["latitude"]:.6f}`

            Longitude: `{result["longitude"]:.6f}`

            **Prospectivity score: {result["prospectivity_score"]:.1f}/100**

            Exploration priority: **{result["priority"]}**

            Model signal: **{result["model_signal"]}**

            The prediction represents an exploration-priority
            ranking derived from the available satellite,
            geological, structural and terrain indicators.

            It is not proof of manganese mineralization,
            ore grade or a proven reserve.
            **Target Assessment Summary**:
            - Coordinates: `{result["latitude"]:.6f}°N, {result["longitude"]:.6f}°E`
            - Prospectivity Score: **{result["prospectivity_score"]:.1f}/100** ({result["priority"]} Priority)
            - Model Signal: **{result["model_signal"]}** via **{result.get("model_type", "ENSEMBLE")}**
            - Nearest Known Manganese Mineralization: **{nearest.get("name", "N/A")}** ({nearest.get("distance_km", 0.0):.2f} km away in the *{nearest.get("belt", "Region")}*)
            
            **Geological Context**: Area belongs to the `{geo.get("group", "Unknown")}` formation.
            {"Structural contact complexity is high within 3 km, indicating prospective tectonic/stratigraphic trapping environments." if struct.get("boundary_density_3km", 0) > 10 else "Structural contact density is moderate."}
            
            **Actionable Next Step**: {"Prioritize for detailed geological field traverse, geochemical soil sampling, and geophysical magnetic/resistivity surveys." if result["prospectivity_score"] >= 60 else "Maintain for regional monitoring; prioritize higher-ranking contiguous target zones."}
            """
        )


# =========================================================
# AREA MODE
# =========================================================

else:

    st.subheader("⬡ Exploration Area Analysis")

    st.write(
        "Draw a rectangle or polygon on the map to analyse "
        "the prospectivity of the selected area."
        "Draw a rectangle or polygon anywhere on the map to evaluate regional prospectivity "
        "and cluster high-priority target zones."
    )

    # -----------------------------------------------------
    # CREATE MAP
    # -----------------------------------------------------
    area_center_preset = st.selectbox(
        "Focus Map On Mineral Province:",
        [
            "Central India (Balaghat / Nagpur / Sausar Belt)",
            "Sandur Fe-Mn Belt (Bellary, Karnataka)",
            "Bonai-Keonjhar Fe-Mn Belt (Odisha / Jharkhand)",
            "Goa Fe-Mn Belt",
            "Eastern Ghats Belt (Andhra Pradesh / Odisha)",
            "All India Overview"
        ],
        index=0
    )

    center_coords = {
        "Central India (Balaghat / Nagpur / Sausar Belt)": ([21.82, 80.17], 9),
        "Sandur Fe-Mn Belt (Bellary, Karnataka)": ([15.08, 76.55], 9),
        "Bonai-Keonjhar Fe-Mn Belt (Odisha / Jharkhand)": ([21.90, 85.25], 9),
        "Goa Fe-Mn Belt": ([15.25, 74.15], 10),
        "Eastern Ghats Belt (Andhra Pradesh / Odisha)": ([18.35, 83.50], 9),
        "All India Overview": ([20.59, 78.96], 5)
    }
    c_loc, c_zoom = center_coords[area_center_preset]

    m = folium.Map(
        location=[21.82, 80.17],
        zoom_start=9,
        location=c_loc,
        zoom_start=c_zoom,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # -----------------------------------------------------
    # DRAW TOOL
    # -----------------------------------------------------

    Draw(
        export=False,
        position="topleft",
        draw_options={
            "polyline": False,
            "rectangle": True,
            "circle": False,
            "marker": False,
            "circlemarker": False,
            "polygon": True,
        },
        edit_options={
            "edit": True,
            "remove": True,
        },
    ).add_to(m)

    # -----------------------------------------------------
    # SHOW MAP
    # -----------------------------------------------------

    map_result = st_folium(
        m,
        width=None,
        height=650,
        height=600,
        key="area_selection_map",
    )

    # -----------------------------------------------------
    # PROCESS DRAWING
    # -----------------------------------------------------

    drawings = map_result.get("all_drawings")

    if drawings:

        selected_geometry = drawings[-1]
        geometry = selected_geometry.get("geometry", {})

        geometry = selected_geometry.get(
            "geometry",
            {},
        )

        if geometry.get("type") == "Polygon":

            coordinates = geometry["coordinates"][0]

            try:

                result = engine.predict_area(
                    coordinates
                )

                if result["cell_count"] > 0:

                    # Store result so it survives reruns.
                    st.session_state.area_result = result

                with st.spinner("Analyzing exploration area and clustering target zones..."):
                    result = engine.predict_area(
                        coordinates,
                        model_type=selected_model_type
                    )
                    if result.get("cell_count", 0) > 0:
                        st.session_state.area_result = result
            except Exception as error:
                st.error("Area analysis failed.")
                st.code(str(error))

                st.error(
                    "Area analysis failed."
                )

                st.code(
                    str(error)
                )

    # -----------------------------------------------------
    # DISPLAY STORED AREA RESULT
    # -----------------------------------------------------

    result = st.session_state.area_result

    if result is not None:

    if result is not None and result.get("cell_count", 0) > 0:
        st.markdown("---")
        st.subheader("📊 Exploration Area Intelligence")

        st.subheader("📊 Area Intelligence")

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.markdown(
                f"""
                <div class="score-card">
                    <div class="score-label">
                        AVERAGE SCORE
                    </div>
                    <div class="score-number">
                        {result["mean_score"]:.1f}
                    </div>
                    <div class="score-label">
                        out of 100
                    </div>
                    <div class="score-label">AVERAGE PROSPECTIVITY</div>
                    <div class="score-number">{result["mean_score"]:.1f}</div>
                    <div class="score-label">out of 100</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:

            st.markdown(
                f"""
                <div class="score-card">
                    <div class="score-label">
                        MAXIMUM SCORE
                    </div>
                    <div class="score-number">
                        {result["maximum_score"]:.1f}
                    </div>
                    <div class="score-label">
                        out of 100
                    </div>
                    <div class="score-label">PEAK PROSPECTIVITY</div>
                    <div class="score-number">{result["maximum_score"]:.1f}</div>
                    <div class="score-label">out of 100</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c3:

            st.markdown(
                f"""
                <div class="score-card">
                    <div class="score-label">
                        HIGH-PRIORITY CELLS
                    </div>
                    <div class="score-number">
                        {result["high_priority_cells"]}
                    </div>
                    <div class="score-label">HIGH-PRIORITY CELLS</div>
                    <div class="score-number">{result["high_priority_cells"]}</div>
                    <div class="score-label">Cells >= 60 Score</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c4:

            st.markdown(
                f"""
                <div class="score-card">
                    <div class="score-label">
                        EXPLORATION PRIORITY
                    </div>
                    <div class="score-number">
                        {result["priority"]}
                    </div>
                    <div class="score-label">TARGET CLUSTERS</div>
                    <div class="score-number">{result.get("target_zones_count", 0)}</div>
                    <div class="score-label">DBSCAN Target Zones</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Target zones breakdown
        zones = result.get("target_zones", [])
        if zones:
            st.markdown("### 🎯 Clustered Target Exploration Zones")
            zones_df = pd.DataFrame(zones)
            st.dataframe(
                zones_df[["zone_id", "cell_count", "mean_score", "max_score", "centroid_latitude", "centroid_longitude", "priority"]],
                use_container_width=True
            )

        st.markdown("---")

        st.subheader("🧠 AI Exploration Recommendation")

        st.subheader("🧠 Area Exploration Recommendation")
        st.info(
            f"""
            The selected area contains **{result["cell_count"]}**
            model prediction cells.

            **Average prospectivity:** {result["mean_score"]:.1f}/100

            **Maximum prospectivity:** {result["maximum_score"]:.1f}/100

            **High-priority cells:** {result["high_priority_cells"]}

            **Very-high-priority cells:**
            {result["very_high_priority_cells"]}

            **Overall priority:** {result["priority"]}

            **Model signal:** {result["model_signal"]}

            Recommended next step: geological field
            investigation, sampling and detailed exploration.
            Selected polygon contains **{result["cell_count"]}** model prediction cells evaluated at **{result.get("resolution", "Adaptive Grid")}**.
            
            - **Mean Prospectivity:** {result["mean_score"]:.1f}/100
            - **Peak Prospectivity:** {result["maximum_score"]:.1f}/100
            - **High-Priority Cells (Score >= 60):** {result["high_priority_cells"]}
            - **Very-High-Priority Cells (Score >= 80):** {result["very_high_priority_cells"]}
            - **Overall Priority Rating:** **{result["priority"]}**
            - **Evaluated with Model:** **{result.get("model_type", "ENSEMBLE")}**
            
            **Strategic Recommendation**: {"Concentrate reconnaissance exploration and core drill positioning on the identified target clusters." if result["high_priority_cells"] > 0 else "Area indicates low baseline exploration priority. Re-examine perimeter boundaries or focus on primary metallogenic provinces."}
            """
        )

    else:

        st.info(
            "👆 Use the drawing tools in the upper-left "
            "corner of the map to select an exploration area."
            "👆 Use the rectangle or polygon drawing tools in the upper-left corner of the map to select any exploration area."
        )