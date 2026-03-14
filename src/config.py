"""
config.py — Central Configuration for the Multibagger Stock Prediction System
==============================================================================
This is the single place where ALL settings live.

WHY A CONFIG FILE?
  Instead of scattering magic numbers and file paths throughout the code,
  we store them here. If you want to change, say, the training date range
  or add new stocks, you only edit THIS file — nothing else needs to change.

STRUCTURE:
  CONFIG is a Python dictionary (key → value) organized into sections:
    - PATHS     : Where to read/write files
    - STOCKS    : Which stocks to analyze
    - DATA      : Date ranges and data params
    - FEATURES  : Which indicators to include
    - MODEL     : ML training hyperparameters
    - TARGET    : Multibagger definition
    - DASHBOARD : UI display settings
"""

import os

# ── Root directory ────────────────────────────────────────────────────────────
# os.path.dirname(__file__)  →  the folder containing this file  (src/)
# os.path.join(..., '..')    →  one level up                     (project root)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CONFIG = {

    # ── File Paths ────────────────────────────────────────────────────────────
    # All paths are relative to the project root, resolved at runtime.
    "PATHS": {
        "raw_data":     os.path.join(ROOT_DIR, "data", "raw"),
        "processed":    os.path.join(ROOT_DIR, "data", "processed"),
        "predictions":  os.path.join(ROOT_DIR, "data", "predictions"),
        "models":       os.path.join(ROOT_DIR, "models"),
        "features_csv": os.path.join(ROOT_DIR, "data", "processed", "features.csv"),
        "predictions_csv": os.path.join(ROOT_DIR, "data", "predictions", "predictions.csv"),
    },

    # ── Stock Universe ────────────────────────────────────────────────────────
    # yfinance uses Yahoo Finance ticker symbols.
    # Indian NSE stocks have the suffix ".NS"  (e.g., RELIANCE.NS)
    # Indian BSE stocks have the suffix ".BO"  (e.g., RELIANCE.BO)
    # US stocks have no suffix                 (e.g., AAPL, MSFT)
    #
    # We start with a curated list of ~60 well-known Indian stocks across
    # sectors: IT, Banking, FMCG, Pharma, Auto, Infra, Consumer.
    "STOCKS": {
        "universe": [
            # ── Large-Cap IT ──────────────────────────────────
            "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
            "LTIM.NS", "MPHASIS.NS", "COFORGE.NS", "PERSISTENT.NS",

            # ── Large-Cap Banking & Financial ─────────────────
            "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS",
            "KOTAKBANK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS",

            # ── FMCG ──────────────────────────────────────────
            "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS",
            "MARICO.NS", "GODREJCP.NS", "COLPAL.NS",

            # ── Pharma & Healthcare ───────────────────────────
            "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS",
            "APOLLOHOSP.NS", "MANKIND.NS",

            # ── Auto & Auto-Ancillary ─────────────────────────
            "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS",
            "HEROMOTOCO.NS", "EICHERMOT.NS", "BOSCHLTD.NS",

            # ── Consumer & Retail ─────────────────────────────
            "TITAN.NS", "TRENT.NS", "DMART.NS", "NYKAA.NS",
            "ZOMATO.NS", "PAYTM.NS",

            # ── Infrastructure & Energy ───────────────────────
            "RELIANCE.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS",
            "ONGC.NS", "BPCL.NS", "ADANIENT.NS", "ADANIPORTS.NS",

            # ── Industrials & Conglomerates ───────────────────
            "LT.NS", "SIEMENS.NS", "ABB.NS", "HAVELLS.NS",
            "POLYCAB.NS", "ASIANPAINT.NS", "BERGERPAINTS.NS",

            # ── Mid-Cap Gems (potential multibaggers) ─────────
            "DIXON.NS", "LAURUS.NS", "DEEPAKNITR.NS", "ASTRAL.NS",
            "PIIND.NS", "CLEAN.NS",
        ],

        # Sector mapping — used for display and sector-level analysis
        "sectors": {
            "IT": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
                   "LTIM.NS", "MPHASIS.NS", "COFORGE.NS", "PERSISTENT.NS"],
            "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS",
                        "KOTAKBANK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS"],
            "FMCG": ["HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS",
                     "MARICO.NS", "GODREJCP.NS", "COLPAL.NS"],
            "Pharma": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS",
                       "APOLLOHOSP.NS", "MANKIND.NS"],
            "Auto": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS",
                     "HEROMOTOCO.NS", "EICHERMOT.NS", "BOSCHLTD.NS"],
            "Consumer": ["TITAN.NS", "TRENT.NS", "DMART.NS", "NYKAA.NS",
                         "ZOMATO.NS", "PAYTM.NS"],
            "Energy": ["RELIANCE.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS",
                       "ONGC.NS", "BPCL.NS"],
            "Industrials": ["LT.NS", "SIEMENS.NS", "ABB.NS", "HAVELLS.NS",
                            "POLYCAB.NS", "ASIANPAINT.NS"],
        },
    },

    # ── Data Settings ─────────────────────────────────────────────────────────
    "DATA": {
        # How many years of historical data to download per stock
        "years_of_history": 10,

        # yfinance interval — "1d" = daily bars (open, high, low, close, volume)
        # Other options: "1wk" (weekly), "1mo" (monthly)
        "interval": "1d",

        # From which date to start downloading (calculated dynamically)
        # We use 10 years back from 2024-01-01 so labels can look 3 years forward
        "start_date": "2013-01-01",
        "end_date":   "2024-01-01",
    },

    # ── Feature Engineering ───────────────────────────────────────────────────
    "FEATURES": {
        # Moving average window sizes (in trading days)
        # 20 days ≈ 1 month, 50 days ≈ 2.5 months, 200 days ≈ 10 months
        "ma_windows": [20, 50, 200],

        # RSI window — standard is 14 days
        "rsi_window": 14,

        # MACD settings: fast EMA, slow EMA, signal line EMA
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,

        # Bollinger Bands window and standard deviation multiplier
        "bb_window": 20,
        "bb_std": 2,

        # Volatility — rolling std dev of daily returns
        "volatility_windows": [10, 30],

        # Momentum — price return over N days
        "momentum_windows": [21, 63, 126],   # ~1 month, 3 months, 6 months

        # Volume spike — ratio of today's volume vs. N-day average
        "volume_avg_window": 30,
    },

    # ── Target Variable (What We're Predicting) ───────────────────────────────
    "TARGET": {
        # A stock is labeled MULTIBAGGER (1) if its price
        # increases by this multiplier over the forward_years horizon
        "return_multiplier": 5.0,     # 5× = 400% gain

        # How many years ahead to look for the price target
        "forward_years": 3,

        # In trading days: 3 years × 252 trading days/year
        "forward_days": 756,
    },

    # ── Machine Learning ──────────────────────────────────────────────────────
    "MODEL": {
        # Number of time-series cross-validation folds
        # TimeSeriesSplit ensures we never train on future data
        "cv_folds": 5,

        # Random Forest hyperparameters
        "rf": {
            "n_estimators": 300,       # Number of trees in the forest
            "max_depth": 8,            # Max depth per tree (controls overfitting)
            "min_samples_leaf": 20,    # Min samples per leaf (smoothing)
            "class_weight": "balanced",# Auto-adjust for rare multibagger class
            "random_state": 42,
            "n_jobs": -1,              # Use all CPU cores
        },

        # XGBoost hyperparameters
        "xgb": {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 10,   # Handles class imbalance (negatives >> positives)
            "random_state": 42,
            "n_jobs": -1,
        },

        # Test set size as a fraction (used in train/test split)
        "test_size": 0.2,
    },

    # ── Dashboard Settings ────────────────────────────────────────────────────
    "DASHBOARD": {
        # How many top stocks to show in the leaderboard
        "top_n_leaderboard": 20,

        # Score thresholds for color-coded labels
        "score_high": 0.70,     # Score ≥ 0.70 → 🔥 High Potential
        "score_medium": 0.45,   # Score ≥ 0.45 → ⚡ Moderate Potential
                                # Score < 0.45 → 🔵 Low Potential

        # Chart settings
        "chart_height": 500,
        "candlestick_days": 365,    # Show last N days of price data in chart
    },
}
