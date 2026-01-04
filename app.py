import streamlit as st
import time
from engine import run_once

# -----------------------------
# PAGE CONFIG (MOBILE FRIENDLY)
# -----------------------------
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
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.title("🏀 NBA Live Betting Model")
st.caption("Real-time ESPN data • Model-driven • No intuition")

# -----------------------------
# CONTROLS
# -----------------------------
refresh = st.toggle("🔄 Auto Refresh (60s)", value=True)
confidence_filter = st.slider("Minimum Confidence", 0.0, 1.0, 0.30)

st.divider()

# -----------------------------
# DATA FETCH
# -----------------------------
with st.spinner("Fetching live games..."):
    results = run_once()

if not results:
    st.warning("No live NBA games right now")
    st.stop()

# -----------------------------
# DISPLAY GAMES (CARD STYLE)
# -----------------------------
for r in results:
    game = r["game"]
    pred = r["prediction"]
    roi = r["roi"]

    if pred["confidence"] < confidence_filter:
        continue

    with st.container():
        st.subheader(f"{game['home_team']} vs {game['away_team']}")
        st.caption(f"Q{game['period']} • {game['clock']}")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Score",
            f"{game['home_score']} - {game['away_score']}"
        )

        col2.metric(
            "Expected Line",
            f"{r['expected_line']}"
        )

        col3.metric(
            "Model Pick",
            pred["bet"]
        )

        st.progress(pred["confidence"])

        st.write(
            f"**Over Probability:** {pred['prob_over']:.1%}  \n"
            f"**Confidence:** {pred['confidence']:.0%}  \n"
            f"**ROI:** {roi['roi_percent']:.1f}%  \n"
            f"**Kelly:** {roi['kelly_fraction']:.1%}"
        )

        if roi["is_positive_ev"] and r["is_valid"]:
            st.success("✅ Positive EV Opportunity")
        else:
            st.info("ℹ️ No edge detected")

        if r["issues"]:
            with st.expander("⚠️ Warnings"):
                for i in r["issues"]:
                    st.write(f"- {i}")

        st.divider()

# -----------------------------
# AUTO REFRESH
# -----------------------------
if refresh:
    time.sleep(60)
    st.rerun()
