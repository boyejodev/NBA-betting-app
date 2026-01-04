import streamlit as st
import time
from datetime import datetime

# 🔗 Import from your existing app.py logic
from app import (
    get_live_nba_games,
    load_model,
    prepare_live_features,
    predict_game,
    calculate_expected_line,
    calculate_roi
)

# =============================
# PAGE CONFIG (MOBILE FIRST)
# =============================
st.set_page_config(
    page_title="NBA Live Betting",
    page_icon="🏀",
    layout="centered"
)

st.markdown("""
<style>
div[data-testid="stMetric"] {
    background-color: #111;
    padding: 12px;
    border-radius: 12px;
}
.game-card {
    background-color: #f8f9fa;
    padding: 14px;
    border-radius: 14px;
    margin-bottom: 14px;
    border-left: 5px solid #1E88E5;
}
</style>
""", unsafe_allow_html=True)

# =============================
# HEADER
# =============================
st.title("🏀 NBA Live Betting Model")
st.caption("Real-time ESPN data • Model-driven • No intuition")

# =============================
# CONTROLS
# =============================
refresh = st.toggle("🔄 Auto Refresh (60s)", value=True)
confidence_filter = st.slider(
    "Minimum Confidence",
    min_value=0.0,
    max_value=1.0,
    value=0.30
)

st.divider()

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def get_model():
    model, default_threshold = load_model()
    return model, default_threshold

model, default_conf_threshold = get_model()

if model is None:
    st.error("❌ Model not loaded")
    st.stop()

# =============================
# FETCH LIVE GAMES
# =============================
with st.spinner("Fetching live NBA games..."):
    games = get_live_nba_games()

if not games:
    st.warning("No live NBA games right now")
    st.stop()

# =============================
# DISPLAY GAMES
# =============================
for game in games:
    features = prepare_live_features(game)
    prediction = predict_game(model, features, default_conf_threshold)
    expected_line = calculate_expected_line(game, features, model)
    roi = calculate_roi(prediction)

    # Confidence filter
    if prediction["confidence"] < confidence_filter:
        continue

    with st.container():
        st.markdown('<div class="game-card">', unsafe_allow_html=True)

        # Header
        st.subheader(f"{game['away_team']} @ {game['home_team']}")
        st.caption(f"Q{game['period']} • {game['clock']}")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Score",
            f"{game['home_score']} - {game['away_score']}"
        )

        col2.metric(
            "Expected Line",
            f"{expected_line:.1f}"
        )

        col3.metric(
            "Model Pick",
            prediction["bet"]
        )

        # Confidence bar
        st.progress(prediction["confidence"])

        # Details
        st.markdown(
            f"""
**Over Probability:** {prediction['prob_over']:.1%}  
**Confidence:** {prediction['confidence']:.0%}  
**ROI:** {roi['roi_percent']:.1f}%  
"""
        )

        # EV indicator
        if roi["is_positive_ev"] and prediction["bet"] != "NO_BET":
            st.success("✅ Positive Expected Value")
        else:
            st.info("ℹ️ No betting edge detected")

        st.markdown('</div>', unsafe_allow_html=True)

# =============================
# AUTO REFRESH (SAFE)
# =============================
if refresh:
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()

    if time.time() - st.session_state.last_refresh > 60:
        st.session_state.last_refresh = time.time()
        st.rerun()
