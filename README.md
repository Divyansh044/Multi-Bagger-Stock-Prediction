# 🚀 AI-Powered Multibagger Stock Prediction System

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest%20%7C%20XGBoost-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)
![Status](https://img.shields.io/badge/Status-Live-success.svg)

An advanced, end-to-end Machine Learning pipeline designed to predict **"Multibagger"** stocks. The system automatically scrapes historical market data and fundamentals, engineers over 40 financial indicators, trains a model robust to class imbalance, and provides a real-time Streamlit dashboard to score any stock on the market.

---

## 🎯 What is a "Multibagger"?

A multibagger is a stock that returns **5× to 10× (or more)** of its original investment over a designated holding period. In this project, our target definition is:
> **Target:** Did the stock price increase by **500% (5×) or more** over a **3-year (756 trading days)** forward-looking horizon?

This turns the stock market into a **Binary Classification Problem** (1 = Multibagger, 0 = Not a Multibagger), which we solve using supervised learning.

---

## 🧠 What the Model Considers (Features)

Instead of relying solely on price action or basic fundamentals, the system engineers a comprehensive mix of signals to give the AI a complete picture of a company's financial health, momentum, and valuation.

### 📈 Technical Indicators (Momentum & Trend)
*   **Moving Averages:** SMA 20, 50, 200 (Short, Mid, Long-term trends)
*   **Oscillators:** RSI (Relative Strength Index) and MACD (Moving Average Convergence Divergence)
*   **Volatility:** Bollinger Bands (Upper, Lower, Width, %B) and annualized historical volatility
*   **Volume Activity:** Volume spikes vs. 30-day average, volume trend
*   **Price Action:** Gap up/down frequency, daily/weekly returns, proximity to 52-week Highs/Lows

### 📊 Fundamental Indicators (Company Health & Valuation)
*   **Valuation Ratios:** P/E (Price-to-Earnings), P/B (Price-to-Book)
*   **Profitability:** ROE (Return on Equity), ROA (Return on Assets)
*   **Growth:** YoY Revenue Growth, YoY Earnings Growth
*   **Financial Risk:** Debt-to-Equity Ratio, Current Ratio
*   **Company Size:** Market Capitalization (Log-transformed & Categorized)
*   **Income:** Dividend Yield

> **Top Signals:** During training, the Random Forest model heavily prioritized company fundamentals: **Debt-to-Equity**, **Earnings Growth**, and **ROA** proved to be the most critical signals, outranking technical momentum indicators.

---

## 🏗️ System Architecture & Pipeline

The project is structured into modular, isolated phases:

1.  **Data Collector (`src/data_collector.py`)** 
    *   Interfaces with the Yahoo Finance API (`yfinance`).
    *   Downloads 10+ years of daily OHLCV data and core fundamentals for a predefined universe of stocks (configurable).
2.  **Preprocessor & Feature Engineer (`src/preprocessor.py`)** 
    *   Cleans raw data (handles missing values, drop delisted symbols).
    *   Computes all complex rolling and technical metrics using the `ta` library.
    *   Shifts data forward by 3 years to create the supervised `'is_multibagger'` target label without data leakage.
3.  **Model Trainer (`src/model_trainer.py`)** 
    *   Implements **Time-Series Split** validation (crucial for financial data to avoid predicting the past with future test sets).
    *   Trains both **Random Forest** and **XGBoost** utilizing `class_weight` and `scale_pos_weight` to address the extreme class imbalance (only ~3.3% of rows are true multibaggers).
    *   Evaluates via **AUC-ROC** and **AUC-PR** curves, saving the best models via `joblib`.
4.  **Prediction Engine (`src/predictor.py`)** 
    *   Downloads the *most recent* data (last 800 days) to calculate today's feature vector.
    *   Scales data using the exact state (mean/variance) of the training dataset.
    *   Outputs a real-time probability score (0.0% to 100.0%).
5.  **Interactive Dashboard (`app.py`)** 
    *   A web-based UI built on **Streamlit** and **Plotly** to visualize predictions.

---

## 🏆 Model Performance

On out-of-sample historical validation:

| Model | AUC-ROC | AUC-PR | Status |
| :--- | :--- | :--- | :--- |
| **Random Forest** | **0.9776** | **0.7458** | 🥇 Best Model |
| **XGBoost** | 0.9431 | 0.6529 | |

*(An AUC-ROC of >0.90 is considered exceptional in quantitative finance modeling)*

---

## 💻 Installation & Setup

**1. Clone the repository**
```bash
git clone https://github.com/Divyansh044/Multi-Bagger-Stock-Prediction.git
cd Multi-Bagger-Stock-Prediction
```

**2. Create a Virtual Environment & Install Dependencies**
```bash
python -m venv venv

# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

## 🚀 How to Run the Dashboard

The models are pre-trained and saved in the `models/` directory. To explore the system immediately, you can launch the Streamlit frontend:

```bash
streamlit run app.py
```

### Dashboard Features:
*   **🏆 Leaderboard View:** See the pre-computed top predictions from our standard stock universe, categorized into 🔥 High Potential, ⚡ Moderate, and 🔵 Low Potential.
*   **🔍 Custom Stock Deep Dive:** Type in *any* valid Yahoo Finance ticker (e.g., `AAPL`, `NVDA`, `ZENTEC.NS`, `RELIANCE.NS`). The system will:
    1. Download its current market data on the fly.
    2. Engineer the 40+ indicators in real-time.
    3. Generate an AI Probability Score.
    4. Display a breakdown of the key metrics driving the score alongside an interactive candlestick chart.

---

## 🛠️ Running the Full ML Pipeline (Advanced)

If you wish to re-train the model, update the stock listed universe inside `src/config.py`, or generate new leaderboard predictions, run the pipeline scripts in this exact order:

```bash
# 1. Download up-to-date historical data for your stock universe
python src/data_collector.py

# 2. Re-compute all features and generate the target labels
python src/preprocessor.py

# 3. Train the Random Forest / XGBoost models on the new feature matrix
python src/model_trainer.py

# 4. Score all stocks in your universe for the UI Leaderboard
python src/predictor.py
```

---
*Disclaimer: This software is for educational and research purposes only. The predictions made by this machine learning model do not constitute financial advice. Always do your own research before investing in the stock market.*
