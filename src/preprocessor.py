"""
preprocessor.py — Feature Engineering Pipeline
================================================
WHAT THIS FILE DOES:
  Transforms raw OHLCV + fundamental data → a clean ML-ready feature matrix.

  Input  : data/raw/*.csv  (one CSV per stock, from data_collector.py)
  Output : data/processed/features.csv  (one row per stock-day, 40+ features)

HOW THE PIPELINE WORKS (step by step):
  1. Load raw price data for each stock
  2. Clean: drop NaN rows, verify minimum data length
  3. Compute TECHNICAL INDICATORS  (from price & volume)
  4. Attach FUNDAMENTAL FEATURES   (from fundamentals.csv)
  5. Create the TARGET LABEL       (is this a future multibagger? 1 or 0)
  6. Stack all stocks into one DataFrame and save → features.csv

WHAT IS FEATURE ENGINEERING?
  Raw stock data is just: date, open, high, low, close, volume.
  A machine learning model can't learn from raw prices alone —
  prices are non-stationary (they drift upward over time) and
  carry no information about momentum, volatility, or trend.

  Feature engineering transforms raw data into SIGNALS:
    "RSI = 28"  →  stock is oversold (potential bounce)
    "MA50 > MA200"  →  golden cross (bullish trend)
    "revenue_growth = 35%"  →  company growing fast
  These signals are what the model actually learns from.
"""

import os
import sys
import logging
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import CONFIG
from src.data_collector import load_price_data, load_fundamentals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
#  SECTION 1 — DATA CLEANING
# =============================================================================

def clean_price_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame | None:
    """
    Clean raw OHLCV data before feature computation.

    Steps:
      1. Ensure the index is a DatetimeIndex (required for time-series ops)
      2. Sort by date ascending (oldest first)
      3. Keep only the 4 columns we need: Open, High, Low, Close, Volume
      4. Drop rows where Close is NaN (can happen on holidays/bad data)
      5. Forward-fill any remaining small gaps (weekend carry-overs)
      6. Reject stocks with < 500 rows (need enough history for indicators)

    WHY 500 ROWS MINIMUM?
      Our longest indicator window is MA-200 (200-day moving average).
      We need at least 200 days of warmup PLUS data for label computation.
      500 rows ≈ 2 years of trading days — a safe floor.
    """
    try:
        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        df = df.sort_index()

        # Normalise column names (yfinance can return "Adj Close" or "Close")
        df.columns = [str(c).strip() for c in df.columns]
        rename_map = {c: c.title().replace(" ", "_") for c in df.columns}
        df = df.rename(columns=rename_map)

        # Keep only OHLCV columns we need
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep].copy()

        # Drop rows where Close is missing
        df = df.dropna(subset=["Close"])

        # Forward-fill volume gaps (volume = 0 on some exchanges for holidays)
        df["Volume"] = df["Volume"].replace(0, np.nan).ffill()

        if len(df) < 500:
            log.warning(f"  ⚠  {symbol}: only {len(df)} rows — skipping (need ≥500)")
            return None

        return df

    except Exception as e:
        log.error(f"  ✗  clean_price_data failed for {symbol}: {e}")
        return None


# =============================================================================
#  SECTION 2 — TECHNICAL INDICATORS
# =============================================================================

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Simple Moving Averages (SMA) for multiple windows.

    SMA(n) = average of the last n closing prices.

    WHY MOVING AVERAGES MATTER:
      They smooth out daily noise and reveal the underlying TREND.
        MA20  = short-term trend  (~1 month)
        MA50  = medium-term trend (~3 months)
        MA200 = long-term trend   (~10 months)

    KEY PATTERNS:
      Price > MA200          → stock is in a long-term uptrend   (bullish)
      MA50 crosses above MA200 → "Golden Cross" — very bullish signal
      MA50 crosses below MA200 → "Death Cross" — bearish signal

    DERIVED FEATURES:
      price_to_ma50  : how far above/below the 50-day average is the price?
                       > 1.0 = trading above average (strength)
                       < 1.0 = trading below average (weakness)
      ma50_to_ma200  : ratio of short-term to long-term trend
                       > 1.0 = uptrend (potentially bullish)
    """
    windows = CONFIG["FEATURES"]["ma_windows"]  # [20, 50, 200]

    for w in windows:
        df[f"ma_{w}"] = df["Close"].rolling(window=w).mean()

    # Ratio features (more informative than raw MA values)
    if "ma_50" in df.columns and "ma_200" in df.columns:
        df["price_to_ma50"]  = df["Close"] / df["ma_50"]
        df["price_to_ma200"] = df["Close"] / df["ma_200"]
        df["ma50_to_ma200"]  = df["ma_50"]  / df["ma_200"]

    return df


def add_rsi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RSI — Relative Strength Index (0 to 100).

    RSI measures the SPEED and MAGNITUDE of recent price changes.
    Developed by J. Welles Wilder in 1978; still one of the most-used indicators.

    FORMULA:
      RS  = Average Gain over N days / Average Loss over N days
      RSI = 100 - (100 / (1 + RS))

    INTERPRETATION:
      RSI > 70  →  Overbought  (stock may be due for a pullback)
      RSI < 30  →  Oversold    (stock may be due for a bounce)
      RSI = 50  →  Neutral     (no strong momentum)

    FOR MULTIBAGGER DETECTION:
      Many multibaggers show RSI breaking above 60 and staying there
      during their early accumulation phase. Sustained RSI > 55 over months
      is a sign of steady institutional buying.
    """
    window = CONFIG["FEATURES"]["rsi_window"]  # 14 days (standard)
    close  = df["Close"]
    delta  = close.diff()                      # daily price change

    gain = delta.clip(lower=0)                 # keep only positive changes
    loss = (-delta).clip(lower=0)              # keep only negative changes (as positive)

    # Exponential weighted average (Wilder's smoothing = span = 2*N - 1)
    avg_gain = gain.ewm(span=window, adjust=False).mean()
    avg_loss = loss.ewm(span=window, adjust=False).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Extra derived signals
    df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)   # binary flag
    df["rsi_oversold"]   = (df["rsi_14"] < 30).astype(int)   # binary flag

    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MACD — Moving Average Convergence Divergence.

    MACD is a MOMENTUM indicator that shows the relationship
    between two exponential moving averages.

    COMPONENTS:
      MACD Line   = EMA(12) − EMA(26)    (fast EMA minus slow EMA)
      Signal Line = EMA(9) of MACD Line  (smoother version)
      Histogram   = MACD Line − Signal   (difference between the two)

    HOW TO READ IT:
      MACD > 0          → short-term trend above long-term (bullish)
      MACD < 0          → short-term trend below long-term (bearish)
      MACD crosses      → direction change (most important signal)
      above Signal Line → BUY signal (bullish crossover)
      below Signal Line → SELL signal (bearish crossover)

    FOR OUR MODEL:
      We compute the histogram (MACD − signal) which captures both
      the direction AND the strength of the momentum signal.
    """
    fast   = CONFIG["FEATURES"]["macd_fast"]    # 12
    slow   = CONFIG["FEATURES"]["macd_slow"]    # 26
    signal = CONFIG["FEATURES"]["macd_signal"]  # 9

    ema_fast   = df["Close"].ewm(span=fast,   adjust=False).mean()
    ema_slow   = df["Close"].ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line= macd_line.ewm(span=signal, adjust=False).mean()

    df["macd"]           = macd_line
    df["macd_signal"]    = signal_line
    df["macd_histogram"] = macd_line - signal_line   # positive = bullish, negative = bearish
    df["macd_crossover"] = (macd_line > signal_line).astype(int)  # 1 when bullish

    return df


def add_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Bollinger Bands — a VOLATILITY indicator.

    WHAT ARE BOLLINGER BANDS?
      Three lines plotted around the price:
        Middle Band = 20-day SMA                     (trend centre)
        Upper Band  = SMA + (2 × standard deviation) (resistance zone)
        Lower Band  = SMA − (2 × standard deviation) (support zone)

      The bands EXPAND when volatility is high (big price swings)
      and CONTRACT when volatility is low (consolidation phase).

    KEY SIGNALS:
      Price touches upper band → overbought / strong uptrend
      Price touches lower band → oversold / potential reversal
      Bollinger Squeeze (bands very close) → big move incoming

    %B INDICATOR:
      %B = (Price − Lower) / (Upper − Lower)
        %B = 1.0 → price at upper band
        %B = 0.5 → price at middle band
        %B = 0.0 → price at lower band
        %B > 1.0 → price ABOVE upper band (breakout!)
    """
    window = CONFIG["FEATURES"]["bb_window"]  # 20
    std    = CONFIG["FEATURES"]["bb_std"]     # 2

    rolling_mean = df["Close"].rolling(window=window).mean()
    rolling_std  = df["Close"].rolling(window=window).std()

    df["bb_upper"]  = rolling_mean + (std * rolling_std)
    df["bb_lower"]  = rolling_mean - (std * rolling_std)
    df["bb_middle"] = rolling_mean
    df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / rolling_mean  # normalised width

    # %B position (where is price within the bands?)
    band_range = df["bb_upper"] - df["bb_lower"]
    df["bb_pct_b"] = (df["Close"] - df["bb_lower"]) / band_range.replace(0, np.nan)

    return df


def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MOMENTUM — the rate of price change over multiple lookback periods.

    Momentum = (Current Price / Price N days ago) − 1

    Example: momentum_21 = 0.08 means the stock is up 8% in the last month.

    WHY MOMENTUM MATTERS FOR MULTIBAGGERS:
      Studies show "momentum" is one of the strongest predictors in finance.
      Stocks that have been going up tend to KEEP going up (trend persistence).
      Early-stage multibaggers often show consistent positive momentum
      across all three timeframes: 1 month, 3 months, and 6 months.

    WINDOWS:
      21  trading days ≈ 1 calendar month
      63  trading days ≈ 3 calendar months (1 quarter)
      126 trading days ≈ 6 calendar months
    """
    windows = CONFIG["FEATURES"]["momentum_windows"]  # [21, 63, 126]

    for w in windows:
        df[f"momentum_{w}d"] = df["Close"].pct_change(periods=w)

    # Momentum consistency: are all 3 timeframes positive? (1 = yes, 0 = no)
    mom_cols = [f"momentum_{w}d" for w in windows]
    df["momentum_all_positive"] = (df[mom_cols] > 0).all(axis=1).astype(int)

    return df


def add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute VOLATILITY — the standard deviation of daily returns.

    Daily return = (Close_today / Close_yesterday) − 1

    Volatility captures HOW MUCH the stock price swings day to day.
    We compute it over two windows: 10 days and 30 days.

    HIGH volatility:
      - Price swings wildly
      - Higher risk, but also higher potential reward
      - Early-stage growth stocks often have higher volatility

    LOW volatility:
      - Stable, predictable moves
      - Typical of large-cap mature companies

    ANNUALISED VOLATILITY:
      Daily vol × sqrt(252)
      This converts daily std to annual percentage — comparable across stocks.

    FOR OUR MODEL:
      We include both short (10-day) and mid-term (30-day) volatility
      to give the model a sense of current vs. recent historical risk.
    """
    windows = CONFIG["FEATURES"]["volatility_windows"]  # [10, 30]

    daily_returns = df["Close"].pct_change()

    for w in windows:
        df[f"volatility_{w}d"] = daily_returns.rolling(window=w).std() * np.sqrt(252)

    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute VOLUME-BASED FEATURES.

    Volume tells us how much INTEREST (buying/selling activity) a stock has.

    KEY VOLUME SIGNALS:
      volume_spike   : today's volume vs. 30-day average
                       Ratio > 2.0 means unusually high activity
                       → could signal institutional buying or news event

      volume_trend   : 5-day avg vs. 20-day avg volume
                       Rising volume trend often confirms price trends

      price_volume   : Close × Volume  = "dollar volume" (liquidity measure)
                       High dollar volume = easy to buy/sell in large quantities

    WHY DOES VOLUME MATTER FOR MULTIBAGGERS?
      When a stock is about to make a big move, smart money (institutions,
      mutual funds) starts accumulating shares quietly. This shows up as
      VOLUME INCREASING while the price is still flat or slightly rising —
      a classic early-stage multibagger signal called "accumulation."
    """
    avg_window = CONFIG["FEATURES"]["volume_avg_window"]  # 30

    df["volume_ma30"]   = df["Volume"].rolling(window=avg_window).mean()
    df["volume_spike"]  = df["Volume"] / df["volume_ma30"].replace(0, np.nan)
    df["volume_ma5"]    = df["Volume"].rolling(window=5).mean()
    df["volume_trend"]  = df["volume_ma5"] / df["volume_ma30"].replace(0, np.nan)

    # Log volume (reduces skew from extreme days)
    df["log_volume"]    = np.log1p(df["Volume"])

    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute raw PRICE-DERIVED FEATURES.

      daily_return     : % change from yesterday's close
      weekly_return    : % change over last 5 trading days
      high_low_range   : (High − Low) / Close  — intraday volatility
      gap_open         : (Open − PrevClose) / PrevClose — overnight gap
      52w_high_ratio   : Close / 52-week high  (how far from peak?)
      52w_low_ratio    : Close / 52-week low   (how far from trough?)
    """
    df["daily_return"]    = df["Close"].pct_change()
    df["weekly_return"]   = df["Close"].pct_change(periods=5)
    df["high_low_range"]  = (df["High"] - df["Low"]) / df["Close"]
    df["gap_open"]        = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)

    # 52-week rolling high and low
    df["high_52w"]        = df["Close"].rolling(window=252).max()
    df["low_52w"]         = df["Close"].rolling(window=252).min()
    df["high_52w_ratio"]  = df["Close"] / df["high_52w"]   # 1.0 = at all-time high
    df["low_52w_ratio"]   = df["Close"] / df["low_52w"]    # high = far from bottom

    return df


# =============================================================================
#  SECTION 3 — FUNDAMENTAL FEATURES
# =============================================================================

def attach_fundamentals(df: pd.DataFrame, symbol: str, fund_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach fundamental (company financial) data to the price DataFrame.

    Since fundamentals don't change daily (they're quarterly / annual snapshots),
    we broadcast the same fundamental values across ALL rows of a stock.

    FEATURES ATTACHED:
      pe_ratio          : Valuation (Price ÷ EPS)
      pb_ratio          : Price ÷ Book Value
      market_cap_log    : Log of market cap (reduces scale differences)
      market_cap_cat    : Small (0) / Mid (1) / Large (2) cap category
      revenue_growth    : YoY revenue growth rate
      earnings_growth   : YoY earnings (profit) growth rate
      roe               : Return on Equity
      roa               : Return on Assets
      debt_to_equity    : Financial leverage ratio
      current_ratio     : Short-term liquidity
      dividend_yield    : Dividend yield (%)
      beta              : Market sensitivity
    """
    row = fund_df[fund_df["Symbol"] == symbol]

    if row.empty:
        # No fundamental data — fill with NaN (model will handle it)
        log.warning(f"  ⚠  No fundamentals for {symbol} — using NaN")
        df["pe_ratio"]        = np.nan
        df["pb_ratio"]        = np.nan
        df["market_cap_log"]  = np.nan
        df["market_cap_cat"]  = np.nan
        df["revenue_growth"]  = np.nan
        df["earnings_growth"] = np.nan
        df["roe"]             = np.nan
        df["roa"]             = np.nan
        df["debt_to_equity"]  = np.nan
        df["current_ratio"]   = np.nan
        df["dividend_yield"]  = np.nan
        df["beta"]            = np.nan
        return df

    r = row.iloc[0]

    df["pe_ratio"]        = r.get("trailingPE",    np.nan)
    df["pb_ratio"]        = r.get("priceToBook",   np.nan)
    df["revenue_growth"]  = r.get("revenueGrowth", np.nan)
    df["earnings_growth"] = r.get("earningsGrowth",np.nan)
    df["roe"]             = r.get("returnOnEquity",np.nan)
    df["roa"]             = r.get("returnOnAssets",np.nan)
    df["debt_to_equity"]  = r.get("debtToEquity",  np.nan)
    df["current_ratio"]   = r.get("currentRatio",  np.nan)
    df["dividend_yield"]  = r.get("dividendYield", np.nan)
    df["beta"]            = r.get("beta",           np.nan)

    # Market cap — log-transform to reduce scale skew (billions vs trillions)
    mkt_cap = r.get("marketCap", np.nan)
    df["market_cap_log"] = np.log1p(mkt_cap) if mkt_cap and mkt_cap > 0 else np.nan

    # Market cap category: 0=Small (<5000 Cr), 1=Mid (5000–20000 Cr), 2=Large (>20000 Cr)
    # In INR: 1 Cr = 10M rupees; Yahoo gives market cap in absolute rupees
    if mkt_cap and mkt_cap > 0:
        mkt_cap_cr = mkt_cap / 1e7  # convert to crores
        if mkt_cap_cr < 5_000:
            df["market_cap_cat"] = 0
        elif mkt_cap_cr < 20_000:
            df["market_cap_cat"] = 1
        else:
            df["market_cap_cat"] = 2
    else:
        df["market_cap_cat"] = np.nan

    return df


# =============================================================================
#  SECTION 4 — TARGET LABEL CREATION
# =============================================================================

def create_target_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the MULTIBAGGER TARGET LABEL.

    This is the most crucial step in the entire pipeline.

    DEFINITION:
      For each row (a stock on a particular date), we look FORWARD
      by `forward_days` trading days and ask:
        "Did the stock price increase by `return_multiplier`× or more?"

      If YES → label = 1 (multibagger)
      If NO  → label = 0 (not a multibagger)

    EXAMPLE:
      Date: 2018-01-15, Close: ₹1,000
      forward_days = 756 (3 years)
      Date+756: 2021-01-15, Close: ₹5,500

      Return = 5500/1000 = 5.5×  ≥  5.0× (threshold)
      → label = 1 ✅ (this was a multibagger signal!)

    WHY THIS APPROACH?
      We are training the model to detect EARLY SIGNALS of future multibaggers.
      The model sees today's technical + fundamental features and learns
      which combinations preceded large 3-year price gains historically.

    IMPORTANT — DATA LEAKAGE RISK:
      We must NEVER let the model see the future price in its features.
      The label uses future price, but the features use ONLY past data.
      TimeSeriesSplit (used in training) further protects against this.

    NOTE ON LAST ROWS:
      Rows within the last `forward_days` of the dataset cannot have a label
      (we don't know their future yet). These get label = NaN and are dropped.
    """
    forward_days      = CONFIG["TARGET"]["forward_days"]       # 756
    return_multiplier = CONFIG["TARGET"]["return_multiplier"]  # 5.0

    # Future price = close price N days in the future
    df["future_close"]    = df["Close"].shift(-forward_days)

    # Total return from today to future
    df["future_return"]   = df["future_close"] / df["Close"]

    # Binary label: 1 if return ≥ 5×, else 0
    df["is_multibagger"]  = (df["future_return"] >= return_multiplier).astype(float)

    # Rows without a future price → NaN label (will be dropped at end)
    df.loc[df["future_close"].isna(), "is_multibagger"] = np.nan

    # Drop helper columns (model doesn't need them)
    df = df.drop(columns=["future_close", "future_return"], errors="ignore")

    return df


# =============================================================================
#  SECTION 5 — MASTER PIPELINE
# =============================================================================

def process_one_stock(symbol: str, fund_df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Full feature engineering pipeline for ONE stock symbol.

    Steps:
      1. Load raw price data from CSV
      2. Clean it
      3. Add all technical indicators
      4. Attach fundamental features
      5. Create multibagger target label
      6. Drop rows with NaN labels or too many missing values

    Returns a DataFrame ready to be stacked with other stocks.
    """
    # ── Step 1: Load ──────────────────────────────────────────────────────────
    df = load_price_data(symbol)
    if df is None:
        return None

    # ── Step 2: Clean ─────────────────────────────────────────────────────────
    df = clean_price_data(df, symbol)
    if df is None:
        return None

    # ── Step 3: Technical Indicators ──────────────────────────────────────────
    df = add_moving_averages(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_momentum(df)
    df = add_volatility(df)
    df = add_volume_features(df)
    df = add_price_features(df)

    # ── Step 4: Fundamentals ──────────────────────────────────────────────────
    df = attach_fundamentals(df, symbol, fund_df)

    # ── Step 5: Target Label ──────────────────────────────────────────────────
    df = create_target_label(df)

    # ── Step 6: Clean Up ──────────────────────────────────────────────────────
    df["symbol"] = symbol

    # Drop rows with no label (last 3 years of data — future unknown)
    df = df.dropna(subset=["is_multibagger"])

    # Drop rows where too many features are missing (>30% NaN in a row)
    max_nan_ratio = 0.30
    threshold     = int(len(df.columns) * max_nan_ratio)
    df = df.dropna(thresh=len(df.columns) - threshold)

    if len(df) < 100:
        log.warning(f"  ⚠  {symbol}: only {len(df)} labeled rows after cleaning — skipping")
        return None

    log.info(f"  ✓  {symbol:20s}  {len(df):>5} rows  "
             f"  multibaggers={int(df['is_multibagger'].sum()):>4}  "
             f"  ({df['is_multibagger'].mean()*100:.1f}%)")

    return df


def run_preprocessing() -> pd.DataFrame:
    """
    Master orchestrator: processes ALL stocks and builds the final feature matrix.

    Flow:
      1. Load the fundamentals table once
      2. Process each stock with process_one_stock()
      3. Stack all individual DataFrames → one big features.csv
      4. Print dataset summary statistics

    OUTPUT FORMAT (features.csv):
      - One row per (stock, date) pair
      - ~40 feature columns (technical + fundamental)
      - One target column: is_multibagger (0 or 1)
      - Symbol column for reference
    """
    log.info("=" * 60)
    log.info("  Multibagger — Feature Engineering Pipeline")
    log.info("=" * 60)

    # Load fundamentals once (shared across all stocks)
    fund_df = load_fundamentals()
    if fund_df is None:
        log.error("  fundamentals.csv not found. Run data_collector.py first.")
        return pd.DataFrame()

    symbols    = CONFIG["STOCKS"]["universe"]
    all_frames = []

    for symbol in symbols:
        df = process_one_stock(symbol, fund_df)
        if df is not None:
            all_frames.append(df)

    if not all_frames:
        log.error("  No data processed. Check data/raw/ directory.")
        return pd.DataFrame()

    # Stack all stocks into one DataFrame
    features = pd.concat(all_frames, axis=0)
    features = features.reset_index()  # makes Date a regular column

    # ── Save to CSV ───────────────────────────────────────────────────────────
    out_path = CONFIG["PATHS"]["features_csv"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    features.to_csv(out_path, index=False)

    # ── Summary Stats ─────────────────────────────────────────────────────────
    total       = len(features)
    multibagger = int(features["is_multibagger"].sum())
    pct         = multibagger / total * 100
    n_stocks    = features["symbol"].nunique()
    n_features  = len([c for c in features.columns
                       if c not in ["Date", "symbol", "is_multibagger",
                                    "Open", "High", "Low", "Close", "Volume"]])

    log.info("\n" + "=" * 60)
    log.info(f"  ✅  Stocks processed  : {n_stocks}")
    log.info(f"  ✅  Total rows        : {total:,}")
    log.info(f"  ✅  Multibagger rows  : {multibagger:,}  ({pct:.2f}%)")
    log.info(f"  ✅  Not-multibagger   : {total - multibagger:,}  ({100-pct:.2f}%)")
    log.info(f"  ✅  Feature columns   : {n_features}")
    log.info(f"  ✅  Saved to          : {out_path}")
    log.info("=" * 60)

    return features


# =============================================================================
#  SECTION 6 — LOADER (used by model_trainer.py)
# =============================================================================

# Columns that are raw price data — not features, not the target
RAW_COLS    = ["Open", "High", "Low", "Close", "Volume"]
META_COLS   = ["Date", "symbol"]
TARGET_COL  = "is_multibagger"


def load_features() -> tuple[pd.DataFrame, pd.Series] | tuple[None, None]:
    """
    Load the engineered feature matrix from disk.

    Returns:
      X : DataFrame of feature columns  (all except meta + target)
      y : Series of target labels        (is_multibagger: 0 or 1)

    Used by model_trainer.py and predictor.py.
    """
    path = CONFIG["PATHS"]["features_csv"]
    if not os.path.exists(path):
        log.error("  features.csv not found. Run run_preprocessing() first.")
        return None, None

    df = pd.read_csv(path, parse_dates=["Date"])

    drop_cols = META_COLS + RAW_COLS + [TARGET_COL]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[TARGET_COL]

    # Drop any remaining fully-NaN columns
    X = X.dropna(axis=1, how="all")

    log.info(f"Loaded features: X={X.shape}, y={y.shape}, "
             f"positives={int(y.sum())} ({y.mean()*100:.2f}%)")

    return X, y


def get_feature_names() -> list[str]:
    """Return the list of feature column names (useful for plots)."""
    X, _ = load_features()
    return list(X.columns) if X is not None else []


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Run the full preprocessing pipeline:
      python src/preprocessor.py

    This reads from data/raw/ and writes to data/processed/features.csv
    """
    features = run_preprocessing()

    if not features.empty:
        print("\n📊 Feature Matrix Preview:")
        print(f"   Shape  : {features.shape}")
        print(f"   Columns: {list(features.columns[:8])} ...")
        print(f"\n   Label distribution:")
        print(f"   {features['is_multibagger'].value_counts().to_dict()}")
        print(f"\n   Saved to: {CONFIG['PATHS']['features_csv']}")
