import streamlit as st
import folium

from folium.plugins import Draw
from streamlit_folium import st_folium

from prediction_engine import ProspectivityEngine


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="MOIL AI Exploration Intelligence",
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
        font-size: 42px;
        font-weight: 750;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 17px;
        opacity: 0.72;
        margin-bottom: 25px;
    }

    .score-card {
        padding: 24px;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
        min-height: 135px;
    }

    .score-number {
        font-size: 38px;
        font-weight: 750;
    }

    .score-label {
        font-size: 14px;
        opacity: 0.65;
    }

    .section-card {
        padding: 20px;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,0.22);
        margin-top: 12px;
        margin-bottom: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


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
    AI-assisted manganese mineral prospectivity mapping using
    satellite, geological, structural and terrain indicators.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🔎 Explore")

mode = st.sidebar.radio(
    "Analysis mode",
    [
        "📍 Single Location",
        "⬡ Select Area",
    ],
)

st.sidebar.markdown("---")

st.sidebar.markdown("### Current Pilot")

st.sidebar.write(
    "📍 Balaghat, Madhya Pradesh"
)

st.sidebar.markdown("### AI Model")

st.sidebar.write(
    "XGBoost Prospectivity Model"
)

st.sidebar.markdown("### Output")

st.sidebar.write(
    "Exploration Priority Score: 0–100"
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "The score represents exploration priority based on "
    "the current model and available evidence. It does not "
    "represent measured manganese concentration or a proven reserve."
)


# =========================================================
# SINGLE LOCATION MODE
# =========================================================

if mode == "📍 Single Location":

    st.subheader("📍 Single Location Analysis")

    st.write(
        "Enter coordinates inside the current Balaghat pilot area."
    )

    col1, col2 = st.columns(2)

    with col1:

        latitude = st.number_input(
            "Latitude",
            min_value=21.30,
            max_value=22.40,
            value=st.session_state.selected_latitude,
            step=0.001,
            format="%.6f",
        )

    with col2:

        longitude = st.number_input(
            "Longitude",
            min_value=79.50,
            max_value=80.80,
            value=st.session_state.selected_longitude,
            step=0.001,
            format="%.6f",
        )

    analyse = st.button(
        "🚀 Analyse Prospectivity",
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
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # -------------------------------------------------
        # MAP
        # -------------------------------------------------

        st.subheader("🗺️ Location on Exploration Map")

        m = folium.Map(
            location=[
                result["latitude"],
                result["longitude"],
            ],
            zoom_start=10,
            tiles="OpenStreetMap",
        )

        folium.Marker(
            [
                result["latitude"],
                result["longitude"],
            ],
            tooltip="Selected Location",
            popup=(
                f"<b>Prospectivity:</b> "
                f"{result['prospectivity_score']:.1f}/100<br>"
                f"<b>Priority:</b> "
                f"{result['priority']}"
            ),
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
    )

    # -----------------------------------------------------
    # CREATE MAP
    # -----------------------------------------------------

    m = folium.Map(
        location=[21.82, 80.17],
        zoom_start=9,
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
        key="area_selection_map",
    )

    # -----------------------------------------------------
    # PROCESS DRAWING
    # -----------------------------------------------------

    drawings = map_result.get("all_drawings")

    if drawings:

        selected_geometry = drawings[-1]

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

            except Exception as error:

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

        st.markdown("---")

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
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        st.subheader("🧠 AI Exploration Recommendation")

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
            """
        )

    else:

        st.info(
            "👆 Use the drawing tools in the upper-left "
            "corner of the map to select an exploration area."
        )