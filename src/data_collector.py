"""
data_collector.py — Stock Data Collection Pipeline
====================================================
WHAT THIS FILE DOES:
  Downloads historical stock market data from Yahoo Finance using the
  `yfinance` library, then saves it to CSV files in data/raw/.

TWO TYPES OF DATA WE COLLECT:
  1. Price Data (OHLCV)
     - Open, High, Low, Close prices (daily)
     - Volume (number of shares traded each day)
     - Used to compute technical indicators (RSI, MACD, etc.)

  2. Fundamental Data
     - Financial metrics about the *company* (not just the price)
     - P/E ratio, Market Cap, EPS, Revenue, Book Value, ROE, etc.
     - Used to compute growth metrics and valuation features

HOW IT WORKS:
  1. Read stock symbols from CONFIG
  2. For each stock, call yfinance.download() → gets OHLCV data
  3. For each stock, call yf.Ticker().info → gets fundamental metadata
  4. Save both to CSV files in data/raw/
  5. Show a progress bar (tqdm) so you can see how far along we are
"""

import os
import time
import logging
import warnings

import pandas as pd
import yfinance as yf
from tqdm import tqdm

# Suppress noisy yfinance deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Import our central config ─────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import CONFIG

# ── Set up logging ────────────────────────────────────────────────────────────
# logging.basicConfig sets up a simple logger that prints timestamped messages.
# We use INFO level — shows important events, hides debug noise.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
#  SECTION 1 — PRICE DATA (OHLCV)
# =============================================================================

def download_price_data(symbol: str, start: str, end: str, interval: str) -> pd.DataFrame | None:
    """
    Download daily OHLCV (Open, High, Low, Close, Volume) data for one stock.

    PARAMETERS:
      symbol   : e.g. "RELIANCE.NS" (Yahoo Finance ticker format)
      start    : "2013-01-01"  — start date (inclusive)
      end      : "2024-01-01"  — end date (exclusive)
      interval : "1d" = daily bars

    RETURNS:
      A pandas DataFrame with columns:
        Date | Open | High | Low | Close | Volume | Dividends | Stock Splits

    WHAT IS OHLCV?
      Every trading day, a stock has 4 key prices:
        Open   = price at market open (9:30 AM IST)
        High   = highest price reached during the day
        Low    = lowest price reached during the day
        Close  = final price at market close (3:30 PM IST)
      Volume = total shares traded that day (measures activity/interest)
    """
    try:
        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,   # Adjusts for splits & dividends automatically
            progress=False,     # Suppress per-stock progress bar (we have our own)
        )

        if df.empty:
            log.warning(f"  ⚠  No price data returned for {symbol}")
            return None

        # yfinance sometimes returns MultiIndex columns — flatten them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Add symbol column so we know which stock this row belongs to
        df["Symbol"] = symbol
        df.index.name = "Date"

        return df

    except Exception as e:
        log.error(f"  ✗  Failed to download price data for {symbol}: {e}")
        return None


# =============================================================================
#  SECTION 2 — FUNDAMENTAL DATA
# =============================================================================

def download_fundamental_data(symbol: str) -> dict | None:
    """
    Download key fundamental (company financial) metrics for one stock.

    yfinance's Ticker().info returns a large Python dictionary (~150 keys).
    We extract only the fields we care about for our ML model.

    WHAT ARE FUNDAMENTALS?
      While price data tells you WHAT a stock is doing,
      fundamentals tell you WHY — the underlying business health.

      Key metrics we collect:
        trailingPE    : P/E ratio = Price ÷ Earnings-per-share
                        High P/E = market expects high growth
                        Low P/E  = undervalued or slow growth

        priceToBook   : P/B ratio = Price ÷ Book Value per share
                        Book value = Assets − Liabilities (net worth of company)

        marketCap     : Total value of all shares = Price × Shares Outstanding
                        Small-cap (<₹5,000 Cr) vs Large-cap (>₹20,000 Cr)

        trailingEps   : Earnings Per Share (last 12 months)
                        Higher EPS = company is more profitable per share

        revenueGrowth : Year-over-year revenue growth rate (0.15 = 15%)

        earningsGrowth: Year-over-year earnings growth rate

        returnOnEquity: ROE = Net Income ÷ Shareholders' Equity
                        Measures how efficiently the company uses investor money

        debtToEquity  : Total Debt ÷ Equity
                        Low = financially healthy, High = risky leverage

        currentRatio  : Current Assets ÷ Current Liabilities
                        > 1.5 = healthy short-term liquidity

        dividendYield : Annual dividend ÷ Stock price
                        Mature companies pay dividends; growth stocks often don't
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Keys we want to extract (with None as fallback if missing)
        fields = [
            "shortName",        # Company name (e.g. "Reliance Industries")
            "sector",           # e.g. "Energy", "Technology"
            "industry",         # e.g. "Oil & Gas Refining"
            "marketCap",        # Total market capitalisation (in rupees/dollars)
            "trailingPE",       # P/E ratio (trailing 12 months)
            "forwardPE",        # P/E ratio (next 12 months forecast)
            "priceToBook",      # P/B ratio
            "trailingEps",      # Earnings per share (trailing 12 months)
            "revenueGrowth",    # Revenue growth YoY (as decimal, e.g. 0.18 = 18%)
            "earningsGrowth",   # Earnings growth YoY
            "returnOnEquity",   # ROE
            "returnOnAssets",   # ROA
            "debtToEquity",     # D/E ratio
            "currentRatio",     # Liquidity ratio
            "dividendYield",    # Annual dividend yield
            "beta",             # Market sensitivity (1.0 = moves with market)
            "52WeekHigh",       # 52-week high price
            "52WeekLow",        # 52-week low price
            "averageVolume",    # Average daily volume
            "sharesOutstanding",# Total shares in circulation
        ]

        fundamentals = {"Symbol": symbol}
        for field in fields:
            fundamentals[field] = info.get(field, None)  # None if not available

        return fundamentals

    except Exception as e:
        log.error(f"  ✗  Failed to download fundamentals for {symbol}: {e}")
        return None


# =============================================================================
#  SECTION 3 — SAVE TO CSV
# =============================================================================

def save_price_data(df: pd.DataFrame, symbol: str, output_dir: str) -> str:
    """
    Save OHLCV DataFrame to a CSV file.

    File naming convention: RELIANCE.NS → RELIANCE_NS.csv
    (Replace '.' with '_' to avoid issues with file extensions)
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = symbol.replace(".", "_") + ".csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath)
    return filepath


def save_fundamentals(fundamentals_list: list[dict], output_dir: str) -> str:
    """
    Save ALL stocks' fundamental data into ONE combined CSV.

    We collect fundamentals for every stock and save them together
    in a single file: data/raw/fundamentals.csv
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "fundamentals.csv")
    df = pd.DataFrame(fundamentals_list)
    df.to_csv(filepath, index=False)
    return filepath


# =============================================================================
#  SECTION 4 — MAIN ORCHESTRATOR
# =============================================================================

def run_collection(symbols: list[str] = None) -> dict:
    """
    Master function that downloads OHLCV + fundamentals for all stocks.

    FLOW:
      1. Load settings from CONFIG
      2. Loop over every stock symbol with a progress bar
      3. Download price data → save as individual CSV
      4. Download fundamentals → collect in a list
      5. Save all fundamentals as one combined CSV
      6. Return a summary dict (success/fail counts)

    PARAMETERS:
      symbols : Optional list to override CONFIG. Useful for testing with
                a small subset, e.g. ["TCS.NS", "INFY.NS"]
    """
    # Load config values
    raw_dir    = CONFIG["PATHS"]["raw_data"]
    start      = CONFIG["DATA"]["start_date"]
    end        = CONFIG["DATA"]["end_date"]
    interval   = CONFIG["DATA"]["interval"]
    all_symbols = symbols or CONFIG["STOCKS"]["universe"]

    log.info("=" * 60)
    log.info("  Multibagger — Data Collection Pipeline")
    log.info("=" * 60)
    log.info(f"  Stocks     : {len(all_symbols)}")
    log.info(f"  Date range : {start}  →  {end}")
    log.info(f"  Interval   : {interval} (daily bars)")
    log.info(f"  Output dir : {raw_dir}")
    log.info("=" * 60)

    success_price = 0
    failed_price  = 0
    fundamentals_list = []

    # ── tqdm wraps the loop to show a live progress bar ───────────────────────
    # Format: [████████████░░░░░░░] 35/64 | TCS.NS | 00:12<00:20
    for symbol in tqdm(all_symbols, desc="📥 Downloading", unit="stock", ncols=80):

        # ── Step A: Price Data ─────────────────────────────────────────────────
        df_price = download_price_data(symbol, start, end, interval)

        if df_price is not None and not df_price.empty:
            filepath = save_price_data(df_price, symbol, raw_dir)
            success_price += 1
            log.info(f"  ✓ {symbol:20s}  {len(df_price):>5} rows  →  {os.path.basename(filepath)}")
        else:
            failed_price += 1

        # ── Step B: Fundamental Data ───────────────────────────────────────────
        fund = download_fundamental_data(symbol)
        if fund:
            fundamentals_list.append(fund)

        # ── Politeness delay ───────────────────────────────────────────────────
        # Yahoo Finance rate-limits aggressive scrapers.
        # A 0.5-second pause between requests keeps us well within limits.
        time.sleep(0.5)

    # ── Save combined fundamentals file ───────────────────────────────────────
    if fundamentals_list:
        fund_path = save_fundamentals(fundamentals_list, raw_dir)
        log.info(f"\n  ✓ Fundamentals saved  →  {fund_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info(f"  ✅  Price data  :  {success_price} succeeded,  {failed_price} failed")
    log.info(f"  ✅  Fundamentals:  {len(fundamentals_list)} stocks collected")
    log.info("=" * 60)

    return {
        "success": success_price,
        "failed":  failed_price,
        "fundamentals": len(fundamentals_list),
    }


# =============================================================================
#  SECTION 5 — UTILITIES
# =============================================================================

def load_price_data(symbol: str) -> pd.DataFrame | None:
    """
    Load already-downloaded OHLCV data from CSV for a given symbol.

    Used by preprocessor.py and predictor.py — they don't re-download,
    they read from the files we already saved.
    """
    raw_dir  = CONFIG["PATHS"]["raw_data"]
    filename = symbol.replace(".", "_") + ".csv"
    filepath = os.path.join(raw_dir, filename)

    if not os.path.exists(filepath):
        log.warning(f"  File not found: {filepath}")
        return None

    df = pd.read_csv(filepath, index_col="Date", parse_dates=True)
    return df


def load_all_price_data() -> dict[str, pd.DataFrame]:
    """
    Load OHLCV data for ALL stocks from disk into a dictionary.

    Returns:
      { "TCS.NS": DataFrame, "INFY.NS": DataFrame, ... }

    Useful when the preprocessor needs all stocks at once.
    """
    symbols = CONFIG["STOCKS"]["universe"]
    all_data = {}

    for symbol in symbols:
        df = load_price_data(symbol)
        if df is not None:
            all_data[symbol] = df

    log.info(f"Loaded price data for {len(all_data)}/{len(symbols)} stocks.")
    return all_data


def load_fundamentals() -> pd.DataFrame | None:
    """
    Load the combined fundamentals CSV from disk.

    Returns a DataFrame with one row per stock and columns for
    P/E, market cap, revenue growth, ROE, etc.
    """
    raw_dir  = CONFIG["PATHS"]["raw_data"]
    filepath = os.path.join(raw_dir, "fundamentals.csv")

    if not os.path.exists(filepath):
        log.warning("  fundamentals.csv not found. Run run_collection() first.")
        return None

    return pd.read_csv(filepath)


def get_collection_summary() -> dict:
    """
    Check how many stocks have already been downloaded (useful for resuming).
    Returns counts of price files and whether fundamentals.csv exists.
    """
    raw_dir = CONFIG["PATHS"]["raw_data"]
    if not os.path.exists(raw_dir):
        return {"price_files": 0, "fundamentals": False}

    csv_files = [f for f in os.listdir(raw_dir)
                 if f.endswith(".csv") and f != "fundamentals.csv"]

    return {
        "price_files":   len(csv_files),
        "fundamentals":  os.path.exists(os.path.join(raw_dir, "fundamentals.csv")),
        "raw_dir":       raw_dir,
    }


# =============================================================================
#  ENTRY POINT — Run directly: python src/data_collector.py
# =============================================================================

if __name__ == "__main__":
    """
    When you run this file directly:
      python src/data_collector.py

    It downloads data for ALL stocks in CONFIG["STOCKS"]["universe"].

    To test with just 2-3 stocks first, temporarily change this to:
      run_collection(symbols=["TCS.NS", "INFY.NS", "RELIANCE.NS"])
    """
    # First check what's already been downloaded
    summary = get_collection_summary()
    log.info(f"Already downloaded: {summary['price_files']} price files, "
             f"fundamentals={'Yes' if summary['fundamentals'] else 'No'}")

    # Run the full collection
    results = run_collection()

    print("\n📊 Collection Complete!")
    print(f"   Price files saved : {results['success']}")
    print(f"   Failed downloads  : {results['failed']}")
    print(f"   Fundamental rows  : {results['fundamentals']}")
    print(f"\n   Data saved in     : {CONFIG['PATHS']['raw_data']}")
