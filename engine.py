# ========================================
# 🏀 NBA LIVE BETTING ENGINE (STREAMLIT SAFE)
# ========================================

import requests
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
from datetime import datetime
from typing import Dict, List
import joblib

# ========================================
# PATHS
# ========================================
DATA_DIR = "data"
MODEL_DIR = "models"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ========================================
# ESPN API
# ========================================
ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
)

# ========================================
# TEAM MAP
# ========================================
TEAM_MAP = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA", "Washington Wizards": "WAS"
}

def norm_team(name: str) -> str:
    return TEAM_MAP.get(name, name[:3].upper())

# ========================================
# LIVE GAMES
# ========================================
def get_live_games() -> List[Dict]:
    r = requests.get(ESPN_SCOREBOARD_URL, timeout=15)
    data = r.json()

    games = []
    for event in data.get("events", []):
        comp = event["competitions"][0]
        status = comp["status"]["type"]

        if status["state"] != "in":
            continue

        teams = comp["competitors"]
        home = next(t for t in teams if t["homeAway"] == "home")
        away = next(t for t in teams if t["homeAway"] == "away")

        games.append({
            "home_team": home["team"]["displayName"],
            "away_team": away["team"]["displayName"],
            "home_score": int(home.get("score", 0)),
            "away_score": int(away.get("score", 0)),
            "period": comp["status"].get("period", 1),
            "clock": comp["status"].get("displayClock", "00:00"),
        })

    return games

# ========================================
# MODEL
# ========================================
def load_model():
    model = xgb.Booster()
    model.load_model(os.path.join(MODEL_DIR, "profit_model.json"))
    features = joblib.load(os.path.join(MODEL_DIR, "model_features.pkl"))
    return model, features

# ========================================
# FEATURES (SIMPLE / SAFE)
# ========================================
def make_features():
    return pd.DataFrame([{
        "HOME_PPG_10": 115.1,
        "AWAY_PPG_10": 115.1,
        "HOME_PACE_10": 100.3,
        "AWAY_PACE_10": 100.3,
        "LEAGUE_PACE_30D": 100.3,
        "SIMULATED_LINE": 226.5,
        "RESIDUAL": 0.0
    }])

# ========================================
# PREDICTION
# ========================================
def predict(model, features, expected):
    dm = xgb.DMatrix(features[expected])
    p = float(model.predict(dm)[0])
    return {
        "prob_over": p,
        "bet": "OVER" if p > 0.55 else "UNDER",
        "confidence": abs(p - 0.5) * 2
    }

# ========================================
# RUN ONCE (STREAMLIT ENTRY)
# ========================================
def run_once():
    model, expected = load_model()
    games = get_live_games()

    results = []
    for g in games:
        feats = make_features()
        pred = predict(model, feats, expected)
        results.append({
            "game": g,
            "prediction": pred
        })

    return results
