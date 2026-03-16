"""
predictor.py — Prediction Engine
=================================
WHAT THIS FILE DOES:
  Loads the trained ML model and scores EVERY stock with a
  "multibagger probability" — a number between 0 and 1 indicating
  how likely the model thinks that stock will deliver 5× returns
  over the next 3 years.

  Input  : Latest price data (downloaded fresh from Yahoo Finance)
           + trained model / imputer / scaler from models/
  Output : data/predictions/predictions.csv — ranked stock leaderboard

FLOW:
  1. Download the LATEST 1 year of price data for each stock
  2. Compute all the same features used in training
     (same functions from preprocessor.py — consistency is critical)
  3. Align features to match training columns exactly
  4. Impute + scale using the SAME imputer/scaler fitted during training
  5. Run model.predict_proba() → get probability of class 1 for each stock
  6. Attach company names, sector, current price from fundamentals
  7. Assign score category (High / Moderate / Low)
  8. Save ranked leaderboard to predictions.csv

WHY DO WE USE ONLY THE LATEST DATA FOR PREDICTION?
  During training, each row = one stock on one historical date.
  For prediction (inference), we only care about TODAY — so we
  compute features on the most recent available trading data
  (typically the last row / most recent window) for each stock.
  This gives us one row per stock → one score per stock.

IMPORTANT: SAME PREPROCESSING PIPELINE
  The model was trained on features processed in a specific way.
  We MUST apply identical transformations at prediction time.
  That's why we:
    - Reuse the same indicator functions from preprocessor.py
    - Load the same fitted imputer and scaler from training
    - Align column order to match training feature_names exactly
"""

import os
import sys
import logging
import warnings
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config      import CONFIG
from src.preprocessor import (
    clean_price_data,
    add_moving_averages,
    add_rsi,
    add_macd,
    add_bollinger_bands,
    add_momentum,
    add_volatility,
    add_volume_features,
    add_price_features,
    attach_fundamentals,
)
from src.data_collector import load_fundamentals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODELS_DIR    = CONFIG["PATHS"]["models"]
PRED_DIR      = CONFIG["PATHS"]["predictions"]
PRED_CSV      = CONFIG["PATHS"]["predictions_csv"]


# =============================================================================
#  SECTION 1 — LOAD MODEL ARTEFACTS
# =============================================================================

def load_model_artefacts() -> tuple:
    """
    Load everything saved during model training:
      - The best model        (random_forest.pkl or xgboost.pkl)
      - The fitted imputer    (imputer.pkl)
      - The fitted scaler     (scaler.pkl)
      - The feature name list (feature_names.pkl)
      - The metadata JSON     (model_metadata.json)

    WHY DO WE NEED THE SAME IMPUTER AND SCALER?
      During training, the imputer learned the MEDIAN of each column
      from the training data. The scaler learned the MEAN and STD
      of each column from the training data.

      At prediction time we must use those SAME statistics — not
      recompute from new data — because the model was trained on
      data that was processed with those exact numbers.

      Using a different scaler or imputer would shift the feature
      values and the model would receive inputs outside the range
      it was optimised for → garbage predictions.

    Returns:
      model, imputer, scaler, feature_names, metadata_dict
    """
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            "model_metadata.json not found.\n"
            "Run: python src/model_trainer.py  first."
        )

    with open(meta_path) as f:
        meta = json.load(f)

    best = meta["best_model"]  # e.g. "random_forest"
    model   = joblib.load(os.path.join(MODELS_DIR, f"{best}.pkl"))
    imputer = joblib.load(os.path.join(MODELS_DIR, "imputer.pkl"))
    scaler  = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))

    log.info(f"  Loaded model     : {best}")
    log.info(f"  RF  AUC-ROC      : {meta.get('rf_auc_roc', 'N/A')}")
    log.info(f"  XGB AUC-ROC      : {meta.get('xgb_auc_roc', 'N/A')}")
    log.info(f"  Feature columns  : {len(feature_names)}")

    return model, imputer, scaler, feature_names, meta


# =============================================================================
#  SECTION 2 — DOWNLOAD LATEST DATA
# =============================================================================

def download_latest_price(symbol: str, lookback_days: int = 800) -> pd.DataFrame | None:
    """
    Download the most recent `lookback_days` of daily OHLCV data for a stock.

    WHY 800 DAYS?
      Our preprocessor.py clean_price_data() function requires a minimum
      of 500 trading days (to ensure MA-200 and other long indicators
      have enough warmup). 800 calendar days ≈ 550 trading days.

    The data is downloaded fresh from Yahoo Finance every time
    run_predictions() is called. This means scores reflect the
    current state of the market, not historical snapshots.
    """
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=lookback_days)

    try:
        df = yf.download(
            symbol,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if df.empty:
            return None

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df["Symbol"] = symbol
        df.index.name = "Date"
        return df

    except Exception as e:
        log.debug(f"  Download failed for {symbol}: {e}")
        return None


# =============================================================================
#  SECTION 3 — FEATURE COMPUTATION FOR ONE STOCK
# =============================================================================

def compute_latest_features(symbol: str, fund_df: pd.DataFrame) -> pd.Series | None:
    """
    Compute all technical + fundamental features for a stock
    and return the MOST RECENT ROW as a single pandas Series.

    This Series becomes one row in our prediction matrix —
    one vector of features → one probability score.

    STEPS:
      1. Download latest price data (400 days)
      2. Run the same cleaning and feature functions as preprocessor.py
         (CRITICAL: identical pipeline = valid predictions)
      3. Attach fundamentals (current ratios, growth metrics)
      4. Return the LAST ROW (= today's feature vector)

    NOTE ON "LAST ROW":
      After computing all rolling indicators, the last row of the
      DataFrame has features computed on the most recent 200-day
      window. This is the "current state" of the stock that we
      feed into the model to get its present-day prediction.
    """
    # ── Step 1: Download ──────────────────────────────────────────────────────
    df = download_latest_price(symbol)
    if df is None:
        return None

    # ── Step 2: Clean ─────────────────────────────────────────────────────────
    df = clean_price_data(df, symbol)
    if df is None:
        return None

    # ── Step 3: Technical indicators (same as preprocessor.py) ───────────────
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_momentum(df)
    df = add_volatility(df)
    df = add_volume_features(df)
    df = add_price_features(df)

    # ── Step 4: Fundamentals ─────────────────────────────────────────────────
    df = attach_fundamentals(df, symbol, fund_df)

    # ── Step 5: Pick the most recent row ──────────────────────────────────────
    last_row = df.iloc[-1].copy()
    last_row["symbol"]        = symbol
    last_row["current_price"] = df["Close"].iloc[-1]
    last_row["as_of_date"]    = df.index[-1].strftime("%Y-%m-%d")

    return last_row


# =============================================================================
#  SECTION 4 — SCORE ALL STOCKS
# =============================================================================

def score_stocks(symbols: list[str] = None) -> pd.DataFrame:
    """
    Compute feature vectors for all stocks and score them with the model.

    FLOW:
      1. Load model artefacts
      2. Load fundamentals for company name / sector info
      3. For each stock:
           a. Compute latest feature vector
           b. Collect into a batch DataFrame
      4. Align feature columns to training order (critical!)
      5. Impute → Scale → Predict probability
      6. Attach metadata (company name, sector, CMP)
      7. Assign score label + rank

    HOW predict_proba WORKS:
      model.predict_proba(X) returns a 2-column array:
        col 0 = probability of class 0 (NOT multibagger)
        col 1 = probability of class 1 (IS multibagger)

      We take column 1: higher = model thinks this stock has more
      potential to deliver 5× returns over 3 years.

    Parameters:
      symbols : Optional override list. Defaults to CONFIG universe.
    """
    symbols = symbols or CONFIG["STOCKS"]["universe"]

    log.info("=" * 60)
    log.info("  Multibagger — Prediction Engine")
    log.info("=" * 60)

    # ── Load model artefacts ──────────────────────────────────────────────────
    model, imputer, scaler, feature_names, meta = load_model_artefacts()

    # ── Load fundamentals (for company names / sector) ────────────────────────
    fund_df = load_fundamentals()
    if fund_df is None:
        log.warning("  fundamentals.csv not found — company names will be missing.")
        fund_df = pd.DataFrame(columns=["Symbol", "shortName", "sector"])

    # ── Compute features for each stock ───────────────────────────────────────
    rows      = []
    meta_rows = []   # store non-feature info separately

    for symbol in tqdm(symbols, desc="🔮 Scoring", unit="stock", ncols=80):
        row = compute_latest_features(symbol, fund_df)
        if row is None:
            continue

        # Separate metadata from features
        meta_info = {
            "symbol":        row.get("symbol", symbol),
            "current_price": row.get("current_price", np.nan),
            "as_of_date":    row.get("as_of_date", ""),
        }
        # Drop non-feature columns before feeding to model
        feature_row = row.drop(
            labels=["symbol", "current_price", "as_of_date",
                    "Open", "High", "Low", "Close", "Volume"],
            errors="ignore"
        )
        rows.append(feature_row)
        meta_rows.append(meta_info)

    if not rows:
        log.error("  No stock data could be fetched. Check internet connection.")
        return pd.DataFrame()

    # ── Build feature matrix ──────────────────────────────────────────────────
    X_raw = pd.DataFrame(rows).reset_index(drop=True)

    # Align columns to EXACTLY match training feature order
    # Missing columns → fill with NaN (imputer will handle)
    # Extra columns   → drop (shouldn't exist, but safety net)
    X_aligned = pd.DataFrame(columns=feature_names)
    for col in feature_names:
        if col in X_raw.columns:
            X_aligned[col] = X_raw[col].values
        else:
            X_aligned[col] = np.nan

    X_aligned = X_aligned.astype(float)

    # ── Impute → Scale ────────────────────────────────────────────────────────
    X_imp    = imputer.transform(X_aligned)
    X_scaled = scaler.transform(X_imp)

    # ── Predict probabilities ─────────────────────────────────────────────────
    probabilities = model.predict_proba(X_scaled)[:, 1]

    # ── Assemble results DataFrame ────────────────────────────────────────────
    meta_df = pd.DataFrame(meta_rows)
    results = meta_df.copy()
    results["multibagger_score"] = np.round(probabilities, 4)

    # Attach company info from fundamentals
    if not fund_df.empty and "Symbol" in fund_df.columns:
        fund_subset = fund_df[["Symbol", "shortName", "sector",
                                "marketCap", "trailingPE"]].copy()
        fund_subset = fund_subset.rename(columns={
            "Symbol":    "symbol",
            "shortName": "company_name",
            "sector":    "sector",
            "marketCap": "market_cap",
            "trailingPE":"pe_ratio",
        })
        results = results.merge(fund_subset, on="symbol", how="left")

    # ── Score category labels ─────────────────────────────────────────────────
    high_thresh   = CONFIG["DASHBOARD"]["score_high"]    # 0.70
    medium_thresh = CONFIG["DASHBOARD"]["score_medium"]  # 0.45

    def categorise(score: float) -> str:
        if score >= high_thresh:
            return "🔥 High Potential"
        elif score >= medium_thresh:
            return "⚡ Moderate Potential"
        else:
            return "🔵 Low Potential"

    results["category"] = results["multibagger_score"].apply(categorise)

    # ── Rank by score ─────────────────────────────────────────────────────────
    results = results.sort_values("multibagger_score", ascending=False)
    results = results.reset_index(drop=True)
    results.index += 1          # rank starts at 1
    results.index.name = "rank"

    return results


# =============================================================================
#  SECTION 5 — SAVE & DISPLAY PREDICTIONS
# =============================================================================

def save_predictions(results: pd.DataFrame) -> str:
    """
    Save the ranked predictions to data/predictions/predictions.csv.
    Also prints a pretty leaderboard to the terminal.
    """
    os.makedirs(PRED_DIR, exist_ok=True)
    results.to_csv(PRED_CSV)
    log.info(f"\n  ✅ Predictions saved → {PRED_CSV}")
    return PRED_CSV


def print_leaderboard(results: pd.DataFrame, top_n: int = 20):
    """Print the top N stocks by multibagger score to the terminal."""
    top = results.head(top_n)

    print("\n" + "=" * 72)
    print(f"  🏆  MULTIBAGGER LEADERBOARD  —  Top {top_n} Stocks")
    print("=" * 72)
    print(f"  {'Rank':<5} {'Symbol':<15} {'Company':<25} {'Score':>7}  {'Category'}")
    print("-" * 72)

    for rank, row in top.iterrows():
        symbol  = str(row.get("symbol",       "—"))
        company = str(row.get("company_name", "—"))[:24]
        score   = row["multibagger_score"]
        cat     = row["category"]
        bar     = "█" * int(score * 20)   # visual bar (max 20 chars)
        print(f"  {rank:<5} {symbol:<15} {company:<25} {score:>6.2%}  {cat}")

    print("=" * 72)


# =============================================================================
#  SECTION 6 — SINGLE STOCK PREDICTION (for dashboard)
# =============================================================================

def predict_single_stock(symbol: str) -> dict | None:
    """
    Score a SINGLE stock. Used by the Streamlit dashboard when a user
    searches for a specific stock.

    Returns a dict with:
      symbol, score, category, current_price, as_of_date,
      feature_values (for display in the dashboard)
    """
    try:
        model, imputer, scaler, feature_names, meta = load_model_artefacts()
        fund_df = load_fundamentals()
        if fund_df is None:
            fund_df = pd.DataFrame()

        row = compute_latest_features(symbol, fund_df)
        if row is None:
            return None

        current_price = row.get("current_price", np.nan)
        as_of_date    = row.get("as_of_date", "")

        feature_row = row.drop(
            labels=["symbol", "current_price", "as_of_date",
                    "Open", "High", "Low", "Close", "Volume"],
            errors="ignore"
        )

        # Align, impute, scale
        X_aligned = pd.DataFrame([feature_row], columns=feature_names if True else [])
        X_aligned = pd.DataFrame(columns=feature_names)
        for col in feature_names:
            X_aligned.loc[0, col] = feature_row.get(col, np.nan)
        X_aligned = X_aligned.astype(float)

        X_imp    = imputer.transform(X_aligned)
        X_scaled = scaler.transform(X_imp)

        score = float(model.predict_proba(X_scaled)[0, 1])

        high_thresh   = CONFIG["DASHBOARD"]["score_high"]
        medium_thresh = CONFIG["DASHBOARD"]["score_medium"]
        if score >= high_thresh:
            category = "🔥 High Potential"
        elif score >= medium_thresh:
            category = "⚡ Moderate Potential"
        else:
            category = "🔵 Low Potential"

        # Pull key feature values for display
        key_features = {
            "RSI (14)"         : round(float(feature_row.get("rsi_14", np.nan)),        2),
            "MACD Histogram"   : round(float(feature_row.get("macd_histogram", np.nan)), 4),
            "MA50/MA200 Ratio" : round(float(feature_row.get("ma50_to_ma200", np.nan)),  3),
            "Momentum (3m)"    : round(float(feature_row.get("momentum_63d", np.nan)),   3),
            "Volume Spike"     : round(float(feature_row.get("volume_spike", np.nan)),   2),
            "P/E Ratio"        : round(float(feature_row.get("pe_ratio", np.nan)),        1),
            "P/B Ratio"        : round(float(feature_row.get("pb_ratio", np.nan)),        2),
            "ROE"              : round(float(feature_row.get("roe", np.nan)),             3),
            "Revenue Growth"   : round(float(feature_row.get("revenue_growth", np.nan)), 3),
            "Debt/Equity"      : round(float(feature_row.get("debt_to_equity", np.nan)), 2),
        }

        return {
            "symbol":        symbol,
            "score":         round(score, 4),
            "score_pct":     f"{score:.1%}",
            "category":      category,
            "current_price": current_price,
            "as_of_date":    as_of_date,
            "key_features":  key_features,
        }

    except Exception as e:
        log.error(f"predict_single_stock({symbol}) failed: {e}")
        return None


# =============================================================================
#  SECTION 7 — MASTER ORCHESTRATOR
# =============================================================================

def run_predictions(symbols: list[str] = None) -> pd.DataFrame:
    """
    Full prediction pipeline for all stocks.
    Downloads latest data, scores everything, saves CSV, prints leaderboard.
    """
    results = score_stocks(symbols)

    if results.empty:
        log.error("No predictions generated.")
        return results

    save_predictions(results)
    print_leaderboard(results, top_n=CONFIG["DASHBOARD"]["top_n_leaderboard"])

    high    = (results["category"] == "🔥 High Potential").sum()
    medium  = (results["category"] == "⚡ Moderate Potential").sum()
    low     = (results["category"] == "🔵 Low Potential").sum()

    log.info(f"\n  Summary:")
    log.info(f"    🔥 High Potential    : {high} stocks")
    log.info(f"    ⚡ Moderate Potential : {medium} stocks")
    log.info(f"    🔵 Low Potential     : {low} stocks")

    return results


def load_predictions() -> pd.DataFrame | None:
    """
    Load already-generated predictions from CSV.
    Used by the Streamlit dashboard — no need to rerun scoring on every page load.
    """
    if not os.path.exists(PRED_CSV):
        log.warning("  predictions.csv not found. Run run_predictions() first.")
        return None
    return pd.read_csv(PRED_CSV, index_col="rank")


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Run the prediction engine:
      python src/predictor.py

    Downloads today's data for all 64 stocks, scores them,
    prints the leaderboard, and saves predictions.csv.
    """
    results = run_predictions()

    if not results.empty:
        print(f"\n✅ Prediction complete!")
        print(f"   Stocks scored   : {len(results)}")
        print(f"   Saved to        : {PRED_CSV}")
        top1 = results.iloc[0]
        print(f"\n   🥇 Top pick : {top1['symbol']}  —  score {top1['multibagger_score']:.2%}")
        print(f"              {top1['category']}")
