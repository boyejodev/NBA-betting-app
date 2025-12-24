# app.py
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
import time
from datetime import datetime
import pytz
import requests
import plotly.graph_objects as go
import plotly.express as px
import hashlib

# ========================================
# 📁 CONFIGURATION
# ========================================
st.set_page_config(
    page_title="NBA Live Betting Predictions",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B35;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .positive {
        color: #4CAF50;
        font-weight: bold;
    }
    .negative {
        color: #F44336;
        font-weight: bold;
    }
    .card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .game-card {
        border-left: 5px solid #1E88E5;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# 📊 INITIALIZE SESSION STATE
# ========================================
if 'model' not in st.session_state:
    st.session_state.model = None
if 'games' not in st.session_state:
    st.session_state.games = []
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}

# ========================================
# 🔗 ESPN API FUNCTIONS
# ========================================
def get_live_nba_games():
    """Fetch currently live NBA games from ESPN API"""
    try:
        ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
        response = requests.get(ESPN_SCOREBOARD_URL, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        live_games = []
        
        for event in data.get("events", []):
            competition = event.get("competitions", [{}])[0]
            status = competition.get("status", {})
            status_type = status.get("type", {})
            
            # Process live games and scheduled games
            if status_type.get("state") not in ["in", "pre"]:
                continue
            
            teams = competition.get("competitors", [])
            if len(teams) < 2:
                continue
            
            home_team = None
            away_team = None
            
            for team in teams:
                if team.get("homeAway") == "home":
                    home_team = team
                elif team.get("homeAway") == "away":
                    away_team = team
            
            if not home_team or not away_team:
                continue
            
            # Extract scores safely
            home_score = 0
            away_score = 0
            
            try:
                home_score = int(home_team.get("score", "0"))
                away_score = int(away_team.get("score", "0"))
            except (ValueError, TypeError):
                pass
            
            game_status = status_type.get("description", "Scheduled")
            if status_type.get("state") == "in":
                game_status = f"Q{status.get('period', 0)} - {status.get('displayClock', '00:00')}"
            
            live_games.append({
                "game_id": event.get("id", "unknown"),
                "home_team": home_team.get("team", {}).get("displayName", "Unknown"),
                "away_team": away_team.get("team", {}).get("displayName", "Unknown"),
                "home_abbreviation": home_team.get("team", {}).get("abbreviation", "UNK"),
                "away_abbreviation": away_team.get("team", {}).get("abbreviation", "UNK"),
                "home_score": home_score,
                "away_score": away_score,
                "period": int(status.get("period", 0)),
                "clock": status.get("displayClock", "00:00"),
                "status": game_status,
                "state": status_type.get("state", "pre"),
                "venue": competition.get("venue", {}).get("fullName", "TBD")
            })
        
        return live_games
        
    except Exception as e:
        st.error(f"Failed to fetch ESPN data: {e}")
        return []

# ========================================
# 🧠 MODEL LOADING
# ========================================
@st.cache_resource
def load_model():
    """Load the trained XGBoost model"""
    try:
        MODEL_DIR = 'models'
        model_path = os.path.join(MODEL_DIR, "profit_model.json")
        
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}")
            st.info("Please run the training scripts first to generate the model")
            return None, 0.55
        
        # Load model
        model = xgb.Booster()
        model.load_model(model_path)
        
        # Load training results if available
        results_path = os.path.join(MODEL_DIR, "training_results.json")
        confidence_threshold = 0.55
        
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                results = json.load(f)
                confidence_threshold = float(results.get("confidence_threshold", 0.55))
        
        return model, confidence_threshold
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, 0.55

# ========================================
# ⚙️ FEATURE ENGINEERING
# ========================================
def is_late_season():
    """Check if we're in late NBA season (March onward)"""
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)
    return int(now.month >= 3)

def get_minutes_played(game):
    """Calculate minutes played in the game"""
    period = max(game.get("period", 1), 1)
    minutes_played = (period - 1) * 12
    if game.get("clock") and ':' in game["clock"]:
        try:
            clock_min, clock_sec = map(int, game["clock"].split(':'))
            minutes_played += (12 - clock_min - clock_sec / 60.0)
        except:
            minutes_played += 6  # Fallback
    else:
        minutes_played += 6
    return max(minutes_played, 0.1)

def prepare_live_features(game):
    """Prepare features for prediction with live game context"""
    # Current NBA averages (2024 season)
    HOME_PPG_10 = 115.1
    AWAY_PPG_10 = 115.1
    HOME_PACE_10 = 100.3
    AWAY_PACE_10 = 100.3
    LEAGUE_PACE_30D = 100.3
    
    # Calculate live pace factor
    total_points = game.get("home_score", 0) + game.get("away_score", 0)
    period = max(game.get("period", 1), 1)
    
    if period <= 4:
        minutes_played = get_minutes_played(game)
        
        if minutes_played > 0:
            points_per_48 = (total_points / minutes_played) * 48
            live_pace_factor = points_per_48 / 226.5
        else:
            live_pace_factor = 1.0
    else:
        live_pace_factor = 1.1
    
    # Apply pace adjustment
    HOME_PPG_10 *= live_pace_factor
    AWAY_PPG_10 *= live_pace_factor
    
    # Momentum based on score difference
    score_diff = game.get("home_score", 0) - game.get("away_score", 0)
    period_factor = max(period, 1)
    HOME_MOMENTUM = float(np.clip(score_diff / (10 * period_factor), -1, 1))
    AWAY_MOMENTUM = -HOME_MOMENTUM
    
    # Line value (difference from average)
    minutes_played = get_minutes_played(game)
    fraction_played = minutes_played / 48.0
    projected_total = total_points / fraction_played if fraction_played > 0 else 226.5
    LINE_VALUE = abs(projected_total - 226.5)
    
    # Season timing
    LATE_SEASON = is_late_season()
    
    # Pace differentials
    HOME_PACE_DIFF = HOME_PACE_10 - LEAGUE_PACE_30D
    AWAY_PACE_DIFF = AWAY_PACE_10 - LEAGUE_PACE_30D
    
    features = {
        "HOME_PPG_10": float(HOME_PPG_10),
        "AWAY_PPG_10": float(AWAY_PPG_10),
        "HOME_PACE_10": float(HOME_PACE_10),
        "AWAY_PACE_10": float(AWAY_PACE_10),
        "LEAGUE_PACE_30D": float(LEAGUE_PACE_30D),
        "LINE_VALUE": float(LINE_VALUE),
        "HOME_PACE_DIFF": float(HOME_PACE_DIFF),
        "AWAY_PACE_DIFF": float(AWAY_PACE_DIFF),
        "HOME_MOMENTUM": float(HOME_MOMENTUM),
        "AWAY_MOMENTUM": float(AWAY_MOMENTUM),
        "LATE_SEASON": LATE_SEASON
    }
    
    return pd.DataFrame([features])

# ========================================
# 🔮 PREDICTION FUNCTIONS
# ========================================
def predict_game(model, features_df, confidence_threshold):
    """Make prediction using the trained model"""
    try:
        dmatrix = xgb.DMatrix(features_df)
        prob_over = float(model.predict(dmatrix)[0])
        bet_over = prob_over > confidence_threshold
        confidence = float(abs(prob_over - 0.5) * 2)
        
        return {
            "prob_over": prob_over,
            "bet": "OVER" if bet_over else "UNDER",
            "confidence": confidence
        }
        
    except Exception as e:
        return {
            "prob_over": 0.5,
            "bet": "NO_BET",
            "confidence": 0.0
        }

def calculate_expected_line(game, features_df, model):
    """Calculate expected total points line"""
    try:
        dmatrix = xgb.DMatrix(features_df)
        pred_prob = float(model.predict(dmatrix)[0])
        
        home_score = game.get("home_score", 0)
        away_score = game.get("away_score", 0)
        current_total = home_score + away_score
        period = max(game.get("period", 1), 1)
        
        minutes_played = get_minutes_played(game)
        fraction_played = minutes_played / 48.0
        projected_total = current_total / fraction_played if fraction_played > 0 else 226.5
        
        adjustment = 1.0 + (pred_prob - 0.5) * 0.15
        expected_line = projected_total * adjustment
        
        return float(round(expected_line, 1))
        
    except Exception as e:
        return 226.5

def calculate_roi(prediction):
    """Calculate expected ROI for the bet at -110 odds"""
    prob_over = prediction['prob_over']
    bet_over = prediction['bet'] == 'OVER'
    
    if bet_over:
        expected_value = (prob_over * 100) - ((1 - prob_over) * 110)
    elif prediction['bet'] == 'UNDER':
        expected_value = ((1 - prob_over) * 100) - (prob_over * 110)
    else:
        expected_value = 0
    
    roi_percent = (expected_value / 110) * 100 if expected_value != 0 else 0
    
    return {
        'expected_value': float(expected_value),
        'roi_percent': float(roi_percent),
        'is_positive_ev': expected_value > 0
    }

# ========================================
# 📊 VISUALIZATION FUNCTIONS
# ========================================
def create_prediction_gauge(prediction, game_id):
    """Create a gauge chart for prediction probability with unique ID"""
    prob = prediction['prob_over'] * 100
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"OVER Probability", 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#1E88E5"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 100], 'color': "lightblue"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=50, b=20),
        title_font_size=16
    )
    return fig

def create_score_projection_chart(game, expected_line, game_id):
    """Create a chart showing score progression with unique ID"""
    total_points = game.get('home_score', 0) + game.get('away_score', 0)
    minutes_played = get_minutes_played(game)
    fraction_played = minutes_played / 48.0
    projected_total = total_points / fraction_played if fraction_played > 0 else 226.5
    
    periods = ['Q1', 'Q2', 'Q3', 'Q4', 'Final']
    projected_points = []
    
    for i in range(5):
        if i == 4:  # Final
            projected_points.append(expected_line)
        else:
            projected_points.append(total_points * ((i+1)*0.25))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=periods,
        y=projected_points,
        mode='lines+markers',
        name='Projected Points',
        line=dict(color='blue', width=3),
        marker=dict(size=10)
    ))
    
    fig.add_hline(y=expected_line, line_dash="dash", 
                 annotation_text=f"Expected Line: {expected_line:.1f}", 
                 annotation_position="bottom right")
    
    fig.update_layout(
        title="Score Projection",
        xaxis_title="Period",
        yaxis_title="Total Points",
        height=280,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False
    )
    
    return fig

def create_confidence_bar(prediction, game_id):
    """Create a horizontal bar chart for confidence level"""
    confidence = prediction['confidence'] * 100
    
    fig = go.Figure(go.Bar(
        x=[confidence],
        y=['Confidence'],
        orientation='h',
        marker=dict(
            color='lightgreen' if confidence > 50 else 'lightcoral',
            line=dict(color='darkgreen' if confidence > 50 else 'darkred', width=2)
        ),
        text=[f"{confidence:.1f}%"],
        textposition='inside',
        insidetextanchor='middle'
    ))
    
    fig.update_layout(
        title="Model Confidence",
        xaxis=dict(range=[0, 100]),
        height=150,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
        xaxis_title="Percentage"
    )
    
    return fig

def generate_unique_id(game_id, chart_type):
    """Generate a unique ID for each chart"""
    return f"{game_id}_{chart_type}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

# ========================================
# 🏀 MAIN APP LAYOUT
# ========================================
def main():
    # Header
    st.markdown('<h1 class="main-header">🏀 NBA Live Betting Predictions</h1>', unsafe_allow_html=True)
    st.markdown("Real-time predictions for NBA game totals using machine learning")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 Settings")
        
        # Auto-refresh option
        auto_refresh = st.checkbox("🔄 Auto-refresh", value=True)
        refresh_interval = st.slider("Refresh interval (seconds)", 30, 300, 60)
        
        # Confidence threshold adjustment
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.50, 
            max_value=0.70, 
            value=0.55,
            help="Minimum probability to recommend a bet"
        )
        
        # Betting settings
        st.markdown("---")
        st.markdown("### 💰 Betting Settings")
        bankroll = st.number_input("Bankroll ($)", 100, 10000, 1000)
        bet_percentage = st.slider("Bet Size (% of bankroll)", 1, 10, 2)
        
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        if st.session_state.model:
            st.success("✅ Model loaded successfully")
        else:
            st.warning("⚠️ Model not loaded")
        
        # Last update info
        if st.session_state.last_update:
            st.caption(f"Last updated: {st.session_state.last_update}")
        
        if st.button("Clear Cache", type="secondary"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared!")
        
        st.markdown("---")
        st.markdown("### 📈 About")
        st.caption("""
        This app uses historical NBA data and live game stats to predict 
        whether games will go OVER or UNDER the expected total points line.
        
        **Odds Assumption**: -110 (Standard sportsbook)
        
        **Model**: XGBoost trained on 2021-2025 NBA data
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">🎯 Live Game Predictions</h2>', unsafe_allow_html=True)
    
    with col2:
        refresh_col, _ = st.columns([1, 1])
        with refresh_col:
            if st.button("🔄 Refresh Now", use_container_width=True):
                st.session_state.games = []
                st.rerun()
    
    # Load model if not loaded
    if st.session_state.model is None:
        with st.spinner("Loading prediction model..."):
            model, _ = load_model()
            if model:
                st.session_state.model = model
                st.success("✅ Model loaded successfully!")
            else:
                st.error("❌ Failed to load model. Please check if model files exist.")
                st.stop()
    
    # Fetch live games
    with st.spinner("Fetching live NBA games..."):
        games = get_live_nba_games()
        st.session_state.games = games
        st.session_state.last_update = datetime.now().strftime("%H:%M:%S")
    
    if not games:
        st.warning("⚠️ No live NBA games found. Games may be scheduled or there might be an API issue.")
        st.info("📅 Check back later for scheduled games or during game hours.")
        
        # Show sample predictions or historical data
        with st.expander("Show sample predictions"):
            st.write("Sample predictions would appear here when games are live.")
            st.write("Make sure the NBA season is active and games are scheduled.")
        return
    
    st.success(f"✅ Found {len(games)} game(s)")
    
    # Create tabs for better organization
    if len(games) > 1:
        tab_titles = [f"{game['away_abbreviation']} @ {game['home_abbreviation']}" for game in games]
        tabs = st.tabs(tab_titles)
        
        for i, (game, tab) in enumerate(zip(games, tabs)):
            with tab:
                display_game_prediction(game, i, confidence_threshold, bankroll, bet_percentage)
    else:
        # Single game display
        display_game_prediction(games[0], 0, confidence_threshold, bankroll, bet_percentage)
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

def display_game_prediction(game, game_index, confidence_threshold, bankroll, bet_percentage):
    """Display prediction for a single game"""
    # Game header
    col1, col2, col3 = st.columns([3, 2, 2])
    
    with col1:
        st.markdown(f"### {game['away_team']} @ {game['home_team']}")
        st.caption(f"Venue: {game.get('venue', 'TBD')}")
    
    with col2:
        st.markdown(f"#### {game['away_score']} - {game['home_score']}")
        st.caption(f"Total: {game['away_score'] + game['home_score']}")
    
    with col3:
        status_color = "🟢" if game['state'] == 'in' else "🟡"
        st.markdown(f"**{status_color} {game['status']}**")
        st.caption(f"Quarter: {game['period']}")
    
    # Calculate predictions
    features_df = prepare_live_features(game)
    prediction = predict_game(st.session_state.model, features_df, confidence_threshold)
    expected_line = calculate_expected_line(game, features_df, st.session_state.model)
    roi_info = calculate_roi(prediction)
    
    # Store prediction for later use
    prediction_key = f"{game['game_id']}_{game_index}"
    st.session_state.predictions[prediction_key] = {
        'game': game,
        'prediction': prediction,
        'expected_line': expected_line,
        'roi_info': roi_info,
        'features': features_df
    }
    
    # Create unique IDs for each chart
    gauge_id = generate_unique_id(game['game_id'], "gauge")
    projection_id = generate_unique_id(game['game_id'], "projection")
    confidence_id = generate_unique_id(game['game_id'], "confidence")
    
    # Create columns for display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Charts in columns
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.plotly_chart(
                create_prediction_gauge(prediction, game['game_id']), 
                use_container_width=True,
                key=f"gauge_{gauge_id}"
            )
        
        with chart_col2:
            st.plotly_chart(
                create_score_projection_chart(game, expected_line, game['game_id']),
                use_container_width=True,
                key=f"projection_{projection_id}"
            )
        
        # Confidence bar
        st.plotly_chart(
            create_confidence_bar(prediction, game['game_id']),
            use_container_width=True,
            key=f"confidence_{confidence_id}"
        )
    
    with col2:
        # Betting recommendations
        st.markdown("### 💰 Betting Analysis")
        
        total_points = game['home_score'] + game['away_score']
        minutes_played = get_minutes_played(game)
        fraction_played = minutes_played / 48.0
        projected_total = total_points / fraction_played if fraction_played > 0 else 226.5
        
        # Metrics
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Current Total", f"{total_points}")
        with metric_col2:
            st.metric("Projected Final", f"{projected_total:.1f}")
        
        st.metric("Expected Line", f"{expected_line:.1f}", 
                 delta=f"{expected_line - 226.5:+.1f} vs avg")
        
        # Bet recommendation box
        st.markdown("---")
        
        if prediction['bet'] != 'NO_BET' and roi_info['is_positive_ev']:
            # Calculate bet size
            bet_amount = (bankroll * bet_percentage / 100)
            expected_profit = bet_amount * (roi_info['roi_percent'] / 100)
            
            # Success styling
            st.success(f"### ✅ BET {prediction['bet']}")
            st.markdown(f"**Probability:** {prediction['prob_over']:.1%}")
            st.markdown(f"**Confidence:** {prediction['confidence']:.0%}")
            st.markdown(f"**Expected ROI:** +{roi_info['roi_percent']:.1f}%")
            st.markdown(f"**Recommended Bet:** ${bet_amount:.2f}")
            st.markdown(f"*Expected profit: ${expected_profit:.2f}*")
            
            # Bet button (simulated)
            if st.button(f"Place ${bet_amount:.0f} Bet on {prediction['bet']}", 
                        key=f"bet_{game['game_id']}", 
                        use_container_width=True):
                st.toast(f"💰 ${bet_amount:.0f} bet placed on {prediction['bet']}!")
        else:
            st.warning("### ⚠️ No Bet Recommended")
            st.markdown("No positive expected value detected")
            if prediction['confidence'] < 0.3:
                st.caption("Low confidence in prediction")
    
    # Detailed statistics in expander
    with st.expander("📊 Detailed Statistics & Features", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Game Details:**")
            st.write(f"- Game ID: {game['game_id'][:8]}...")
            st.write(f"- Quarter: {game['period']}")
            st.write(f"- Clock: {game['clock']}")
            st.write(f"- Minutes Played: {minutes_played:.1f}")
            st.write(f"- Completion: {fraction_played:.1%}")
        
        with col2:
            st.markdown("**Model Features:**")
            if not features_df.empty:
                st.write(f"- Home PPG: {features_df['HOME_PPG_10'].iloc[0]:.1f}")
                st.write(f"- Away PPG: {features_df['AWAY_PPG_10'].iloc[0]:.1f}")
                st.write(f"- Pace Diff: {features_df['HOME_PACE_DIFF'].iloc[0]:.1f}")
                st.write(f"- Line Value: {features_df['LINE_VALUE'].iloc[0]:.1f}")
                st.write(f"- Momentum: {features_df['HOME_MOMENTUM'].iloc[0]:.3f}")
        
        with col3:
            st.markdown("**Prediction Details:**")
            st.write(f"- Probability Over: {prediction['prob_over']:.3f}")
            st.write(f"- Recommendation: {prediction['bet']}")
            st.write(f"- Confidence: {prediction['confidence']:.1%}")
            st.write(f"- Expected Value: ${roi_info['expected_value']:.2f}")
            st.write(f"- ROI: {roi_info['roi_percent']:.1f}%")
    
    # Historical comparison (if available)
    if hasattr(st.session_state, 'historical_data'):
        with st.expander("📈 Historical Comparison", expanded=False):
            # Add historical comparison chart here
            pass

# ========================================
# 🚀 RUN THE APP
# ========================================
if __name__ == "__main__":
    main()