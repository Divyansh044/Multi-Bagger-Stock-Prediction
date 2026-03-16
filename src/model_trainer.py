"""
model_trainer.py — Machine Learning Training Pipeline
======================================================
WHAT THIS FILE DOES:
  Trains two ML models (Random Forest and XGBoost) on the engineered
  feature matrix and evaluates their ability to predict multibaggers.

  Input  : data/processed/features.csv  (from preprocessor.py)
  Output : models/random_forest.pkl     (trained Random Forest)
           models/xgboost.pkl           (trained XGBoost)
           models/scaler.pkl            (fitted feature scaler)
           models/training_report.txt   (evaluation metrics)

THE MACHINE LEARNING PROBLEM:
  This is a BINARY CLASSIFICATION problem:
    Input  (X) : 40+ technical + fundamental features for a stock on a date
    Output (y) : 0 = not a multibagger,  1 = multibagger (5× in 3 years)

  We train on historical data (stock features from 2013–2021) and test
  on a held-out period (2021–2024) to simulate real-world prediction.

WHY TWO MODELS?
  • Random Forest : robust, interpretable, good baseline
  • XGBoost       : state-of-the-art for tabular data, typically more accurate
  Comparing both gives us confidence in which signals truly matter
  and lets us pick (or ensemble) the best performer.

PIPELINE OVERVIEW:
  1. Load features.csv
  2. Sort chronologically (critical for time-series integrity)
  3. Impute missing values (median strategy)
  4. Scale features (StandardScaler)
  5. Time-series train/test split
  6. Train Random Forest  → evaluate → save
  7. Train XGBoost        → evaluate → save
  8. Print full report + feature importances
"""

import os
import sys
import logging
import warnings
import json

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (no GUI window)
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.impute           import SimpleImputer
from sklearn.metrics          import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection  import TimeSeriesSplit
from xgboost                  import XGBClassifier

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config       import CONFIG
from src.preprocessor import load_features, RAW_COLS, META_COLS, TARGET_COL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODELS_DIR = CONFIG["PATHS"]["models"]


# =============================================================================
#  SECTION 1 — DATA PREPARATION
# =============================================================================

def prepare_data() -> tuple:
    """
    Load, validate, impute, and scale the feature matrix.

    Returns:
      X_train, X_test, y_train, y_test, feature_names, scaler

    WHY IMPUTATION?
      Some features (especially fundamentals like P/E ratio) are missing
      for certain stocks/dates. ML models crash on NaN values.
      We replace NaN with the MEDIAN of each column — median is preferred
      over mean because it's robust to outliers (a P/E of 5000 won't skew it).

    WHY SCALING?
      Features have wildly different scales:
        rsi_14        → 0 to 100
        market_cap_log → 20 to 30
        momentum_21d  → -0.5 to +1.0
        volume_spike  → 0 to 50

      Random Forest doesn't need scaling (it uses thresholds, not distances),
      but XGBoost also doesn't strictly need it. We scale anyway for:
        1. Numerical stability
        2. Future-proofing if we add SVM or logistic regression
        3. Makes feature coefficients comparable in magnitude

    WHY TIME-SERIES SPLIT AND NOT RANDOM SPLIT?
      Standard k-fold randomly shuffles all rows before splitting.
      But our data is chronological — row from 2023 comes AFTER row from 2019.

      If we randomly shuffle, the training set might include rows from 2023
      and the test set rows from 2015. The model would be "learning the future"
      to predict the past → data leakage → inflated, fake accuracy.

      TimeSeriesSplit always uses PAST for training, FUTURE for testing.
      This simulates real-world deployment: train on what happened,
      test on what comes after.

    TRAIN/TEST SPLIT:
      We use the last 20% of the time range as the test set.
      Given our dataset covers 2013–2021 (labeled data),
      roughly:  Train = 2013–2018   |   Test = 2018–2021
    """
    log.info("Loading features...")
    X, y = load_features()

    if X is None:
        raise FileNotFoundError("features.csv not found. Run preprocessor.py first.")

    # Also load the full df to get Date for chronological ordering
    df_full = pd.read_csv(CONFIG["PATHS"]["features_csv"], parse_dates=["Date"])

    # Sort by Date so time ordering is preserved
    sort_idx = df_full["Date"].argsort()
    X = X.iloc[sort_idx].reset_index(drop=True)
    y = y.iloc[sort_idx].reset_index(drop=True)

    feature_names = list(X.columns)
    log.info(f"  Features : {len(feature_names)}")
    log.info(f"  Samples  : {len(X):,}  (positives: {int(y.sum()):,}, {y.mean()*100:.2f}%)")

    # ── Impute missing values with column median ───────────────────────────────
    log.info("Imputing missing values (median strategy)...")
    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(X)
    X_imp   = pd.DataFrame(X_imp, columns=feature_names)

    # ── Scale features ────────────────────────────────────────────────────────
    log.info("Scaling features (StandardScaler)...")
    scaler  = StandardScaler()
    X_scaled= scaler.fit_transform(X_imp)
    X_scaled= pd.DataFrame(X_scaled, columns=feature_names)

    # ── Time-series train/test split ──────────────────────────────────────────
    test_size  = CONFIG["MODEL"]["test_size"]      # 0.20
    split_idx  = int(len(X_scaled) * (1 - test_size))

    X_train, X_test = X_scaled.iloc[:split_idx], X_scaled.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx],         y.iloc[split_idx:]

    log.info(f"  Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    log.info(f"  Train positives: {int(y_train.sum()):,} ({y_train.mean()*100:.2f}%)")
    log.info(f"  Test  positives: {int(y_test.sum()):,}  ({y_test.mean()*100:.2f}%)")

    # Save imputer + scaler for use in predictor.py
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(imputer, os.path.join(MODELS_DIR, "imputer.pkl"))
    joblib.dump(scaler,  os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))
    log.info(f"  Saved imputer + scaler → {MODELS_DIR}/")

    return X_train, X_test, y_train, y_test, feature_names, scaler, imputer


# =============================================================================
#  SECTION 2 — MODEL EVALUATION HELPER
# =============================================================================

def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """
    Compute and print a comprehensive evaluation report for one model.

    METRICS EXPLAINED:

    ACCURACY:
      (TP + TN) / Total
      Misleading for imbalanced data! If only 3.3% are multibaggers,
      a model that predicts "0" always gets 96.7% accuracy — but is useless.

    PRECISION (for class 1 = multibagger):
      TP / (TP + FP)
      Of all stocks the model flagged as multibaggers, what % actually were?
      High precision → fewer false alarms.

    RECALL (Sensitivity) (for class 1):
      TP / (TP + FN)
      Of all actual multibaggers, what % did the model catch?
      High recall → model misses fewer real multibaggers.

    F1 SCORE:
      Harmonic mean of Precision and Recall.
      Balances both — useful when you care about both false positives and negatives.

    AUC-ROC (Area Under the ROC Curve):
      Measures the model's ability to RANK positives above negatives.
      Range: 0.5 (random) to 1.0 (perfect).
      AUC > 0.70 = good for a rare-event problem like this.
      This is our PRIMARY METRIC — the most informative single number.

    AUC-PR (Area Under the Precision-Recall Curve):
      Better than AUC-ROC when classes are very imbalanced.
      Focuses on how well the model finds the rare positive class.
    """
    y_prob = model.predict_proba(X_test)[:, 1]   # probability of class 1
    y_pred = (y_prob >= 0.5).astype(int)         # threshold at 50%

    auc_roc = roc_auc_score(y_test, y_prob)
    auc_pr  = average_precision_score(y_test, y_prob)
    cm      = confusion_matrix(y_test, y_pred)
    report  = classification_report(y_test, y_pred, target_names=["Not Multi", "Multibagger"])

    log.info(f"\n{'='*55}")
    log.info(f"  📊  {name} — Evaluation Report")
    log.info(f"{'='*55}")
    log.info(f"  AUC-ROC      : {auc_roc:.4f}  (target: >0.70)")
    log.info(f"  AUC-PR       : {auc_pr:.4f}")
    log.info(f"\n{report}")
    log.info(f"  Confusion Matrix:\n{cm}")

    return {
        "name":    name,
        "auc_roc": round(auc_roc, 4),
        "auc_pr":  round(auc_pr,  4),
        "cm":      cm.tolist(),
        "y_prob":  y_prob,
    }


# =============================================================================
#  SECTION 3 — RANDOM FOREST
# =============================================================================

def train_random_forest(X_train, y_train, X_test, y_test, feature_names) -> dict:
    """
    Train a Random Forest classifier and evaluate it.

    WHAT IS A RANDOM FOREST?
      An ensemble of many Decision Trees, each trained on:
        1. A random SUBSET of training rows (bootstrap sampling)
        2. A random SUBSET of features at each split

      The final prediction is the MAJORITY VOTE (classification)
      or AVERAGE (regression) across all trees.

    WHY RANDOM FOREST WORKS WELL HERE:
      ✅ Handles mixed features well (technical indicators + fundamentals)
      ✅ Naturally handles non-linear relationships (unlike linear models)
      ✅ Built-in feature importance ranking
      ✅ Robust to outliers (extreme RSI values, unusual volume spikes)
      ✅ No need for feature scaling (but we do it for consistency)
      ✅ class_weight='balanced' handles the 3.3% vs 96.7% imbalance

    KEY HYPERPARAMETERS:
      n_estimators    : 300 trees — more trees = more stable, slower
      max_depth       : 8 — limits tree depth to prevent overfitting
      min_samples_leaf: 20 — needs 20+ samples to make a leaf — smoothing
      class_weight    : 'balanced' — auto-weights minority class higher
      n_jobs          : -1 — uses all 4 CPU cores

    OVERFITTING vs UNDERFITTING:
      Overfitting  → model memorises training data, fails on new data
                     Prevented by: max_depth, min_samples_leaf, n_estimators
      Underfitting → model too simple, misses patterns
                     Prevented by: enough trees, not too shallow
    """
    log.info("\n" + "="*55)
    log.info("  🌲  Training Random Forest...")
    log.info("="*55)

    params = CONFIG["MODEL"]["rf"]
    rf = RandomForestClassifier(**params)
    rf.fit(X_train, y_train)

    log.info(f"  Trees trained: {rf.n_estimators}")

    # Evaluate
    results = evaluate_model("Random Forest", rf, X_test, y_test)

    # Save model
    rf_path = os.path.join(MODELS_DIR, "random_forest.pkl")
    joblib.dump(rf, rf_path)
    log.info(f"  ✅ Saved → {rf_path}")

    # Feature importance (top 15)
    importances = pd.Series(rf.feature_importances_, index=feature_names)
    results["importances"] = importances.sort_values(ascending=False)
    results["model"] = rf

    return results


# =============================================================================
#  SECTION 4 — XGBOOST
# =============================================================================

def train_xgboost(X_train, y_train, X_test, y_test, feature_names) -> dict:
    """
    Train an XGBoost classifier and evaluate it.

    WHAT IS XGBOOST?
      XGBoost = eXtreme Gradient Boosting.

      Unlike Random Forest (trees grown in PARALLEL, independently),
      XGBoost grows trees SEQUENTIALLY — each new tree corrects the
      errors made by the previous one.

      Process:
        Tree 1 → makes predictions, calculates errors (residuals)
        Tree 2 → trained to predict those errors → reduces them
        Tree 3 → trained on remaining errors
        ...repeat for n_estimators trees

      The final prediction = sum of all trees' outputs.

    WHY XGBOOST IS OFTEN BETTER THAN RANDOM FOREST:
      ✅ Learns from mistakes iteratively (gradient descent on loss)
      ✅ Typically higher accuracy on tabular data
      ✅ scale_pos_weight handles severe class imbalance
      ✅ Regularisation (L1/L2) built-in → less overfitting
      ✅ Won most Kaggle tabular competitions from 2014–2020

    KEY HYPERPARAMETERS:
      n_estimators      : 400 boosting rounds (= 400 sequential trees)
      max_depth         : 6 — depth of each tree
      learning_rate     : 0.05 — shrinks each tree's contribution
                          Smaller = more conservative, needs more trees
      subsample         : 0.8 — use 80% of rows per tree (reduces overfitting)
      colsample_bytree  : 0.8 — use 80% of features per tree
      scale_pos_weight  : 10 — weight positives 10× more (handles 3% vs 97%)

    GRADIENT DESCENT INTUITION:
      Think of trying to hit a target by throwing a ball.
      Each throw (tree) corrects the direction based on where the last ball landed.
      After 400 throws, you're much more accurate than any single throw.
    """
    log.info("\n" + "="*55)
    log.info("  ⚡  Training XGBoost...")
    log.info("="*55)

    params = {**CONFIG["MODEL"]["xgb"]}
    params.pop("random_state", None)   # XGBClassifier uses 'seed' not 'random_state'

    xgb = XGBClassifier(
        **CONFIG["MODEL"]["xgb"],
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
    )
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Evaluate
    results = evaluate_model("XGBoost", xgb, X_test, y_test)

    # Save model
    xgb_path = os.path.join(MODELS_DIR, "xgboost.pkl")
    joblib.dump(xgb, xgb_path)
    log.info(f"  ✅ Saved → {xgb_path}")

    # Feature importance
    importances = pd.Series(xgb.feature_importances_, index=feature_names)
    results["importances"] = importances.sort_values(ascending=False)
    results["model"] = xgb

    return results


# =============================================================================
#  SECTION 5 — PLOTS
# =============================================================================

def save_plots(rf_results: dict, xgb_results: dict, X_test, y_test):
    """
    Generate and save evaluation plots to models/:
      - roc_curve.png       : ROC curves for both models
      - feature_importance.png : Top 20 features by importance (XGBoost)

    ROC CURVE:
      X-axis = False Positive Rate (FPR) = FP / (FP + TN)
               How often does the model falsely flag normal stocks?
      Y-axis = True Positive Rate (TPR) = Recall = TP / (TP + FN)
               How many real multibaggers does it catch?

      The curve traces all possible thresholds from 0% to 100%.
      Area Under Curve (AUC) = single number summary.
      The closer to the top-left corner, the better.
      Diagonal line = random guessing (AUC = 0.5).

    FEATURE IMPORTANCE:
      Shows which features the XGBoost model found most useful.
      Based on how often each feature was used to split data across all trees,
      weighted by how much each split improved prediction accuracy.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    plt.style.use("seaborn-v0_8-darkgrid")

    # ── ROC Curves ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Evaluation — Multibagger Prediction", fontsize=14, fontweight="bold")

    for ax, results, color in zip(
        axes,
        [rf_results,  xgb_results],
        ["steelblue", "darkorange"]
    ):
        fpr, tpr, _ = roc_curve(y_test, results["y_prob"])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{results["name"]}  (AUC={results["auc_roc"]:.3f})')
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.50)")
        ax.fill_between(fpr, tpr, alpha=0.1, color=color)
        ax.set_xlabel("False Positive Rate",  fontsize=11)
        ax.set_ylabel("True Positive Rate",   fontsize=11)
        ax.set_title(f'{results["name"]} — ROC Curve', fontsize=12)
        ax.legend(loc="lower right", fontsize=10)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

    plt.tight_layout()
    roc_path = os.path.join(MODELS_DIR, "roc_curves.png")
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  📈 Saved ROC curves  → {roc_path}")

    # ── Feature Importance (XGBoost top 20) ───────────────────────────────────
    top20 = xgb_results["importances"].head(20)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors  = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top20)))[::-1]
    bars    = ax.barh(top20.index[::-1], top20.values[::-1], color=colors[::-1])

    ax.set_xlabel("Feature Importance Score", fontsize=11)
    ax.set_title("Top 20 Features — XGBoost\n(Multibagger Prediction)", fontsize=13, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)

    for bar, val in zip(bars, top20.values[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    fi_path = os.path.join(MODELS_DIR, "feature_importance.png")
    plt.savefig(fi_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  📊 Saved feature importance → {fi_path}")


# =============================================================================
#  SECTION 6 — SAVE TRAINING REPORT
# =============================================================================

def save_report(rf_results: dict, xgb_results: dict):
    """
    Save a plain-text training report to models/training_report.txt
    and a JSON summary to models/model_metadata.json
    """
    report_path = os.path.join(MODELS_DIR, "training_report.txt")
    meta_path   = os.path.join(MODELS_DIR, "model_metadata.json")

    # Top 10 features comparison
    rf_top10  = rf_results["importances"].head(10)
    xgb_top10 = xgb_results["importances"].head(10)

    report = f"""
MULTIBAGGER PREDICTION — TRAINING REPORT
=========================================
Random Forest
  AUC-ROC : {rf_results['auc_roc']}
  AUC-PR  : {rf_results['auc_pr']}

XGBoost
  AUC-ROC : {xgb_results['auc_roc']}
  AUC-PR  : {xgb_results['auc_pr']}

Best Model : {'XGBoost' if xgb_results['auc_roc'] > rf_results['auc_roc'] else 'Random Forest'}

TOP 10 FEATURES (Random Forest):
{rf_top10.to_string()}

TOP 10 FEATURES (XGBoost):
{xgb_top10.to_string()}
"""
    with open(report_path, "w") as f:
        f.write(report)

    # JSON metadata (used by predictor.py to pick the best model)
    best_model = "xgboost" if xgb_results["auc_roc"] > rf_results["auc_roc"] else "random_forest"
    meta = {
        "best_model":     best_model,
        "rf_auc_roc":     rf_results["auc_roc"],
        "xgb_auc_roc":    xgb_results["auc_roc"],
        "rf_auc_pr":      rf_results["auc_pr"],
        "xgb_auc_pr":     xgb_results["auc_pr"],
        "top_features":   list(xgb_results["importances"].head(15).index),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"  📄 Saved report → {report_path}")
    log.info(f"  📄 Saved metadata → {meta_path}")
    return meta


# =============================================================================
#  SECTION 7 — MASTER ORCHESTRATOR
# =============================================================================

def run_training() -> dict:
    """
    Run the complete training pipeline end-to-end.

    Returns metadata dict with model paths and performance numbers.
    """
    log.info("=" * 55)
    log.info("  Multibagger — ML Training Pipeline")
    log.info("=" * 55)

    # Step 1: Prepare data
    X_train, X_test, y_train, y_test, feature_names, scaler, imputer = prepare_data()

    # Step 2: Train models
    rf_results  = train_random_forest(X_train, y_train, X_test, y_test, feature_names)
    xgb_results = train_xgboost(X_train, y_train, X_test, y_test, feature_names)

    # Step 3: Plots + report
    save_plots(rf_results, xgb_results, X_test, y_test)
    meta = save_report(rf_results, xgb_results)

    log.info("\n" + "=" * 55)
    log.info(f"  🏆  Best model : {meta['best_model'].upper()}")
    log.info(f"  AUC-ROC  RF  : {meta['rf_auc_roc']}")
    log.info(f"  AUC-ROC  XGB : {meta['xgb_auc_roc']}")
    log.info("=" * 55)

    return meta


# =============================================================================
#  LOADERS (used by predictor.py)
# =============================================================================

def load_best_model():
    """Load the best model based on training_report metadata."""
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError("model_metadata.json not found. Run run_training() first.")

    with open(meta_path) as f:
        meta = json.load(f)

    best = meta["best_model"]
    model_path = os.path.join(MODELS_DIR, f"{best}.pkl")
    model = joblib.load(model_path)
    log.info(f"Loaded best model: {best}  (AUC-ROC: {meta[f'{best}_auc_roc' if best == 'rf' else 'xgb_auc_roc']})")
    return model, meta


def load_preprocessors():
    """Load imputer, scaler, and feature names saved during training."""
    imputer       = joblib.load(os.path.join(MODELS_DIR, "imputer.pkl"))
    scaler        = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    return imputer, scaler, feature_names


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Run training:
      python src/model_trainer.py

    Trains both models, saves to models/, prints full evaluation report.
    Generates:
      models/random_forest.pkl
      models/xgboost.pkl
      models/scaler.pkl
      models/imputer.pkl
      models/feature_names.pkl
      models/roc_curves.png
      models/feature_importance.png
      models/training_report.txt
      models/model_metadata.json
    """
    meta = run_training()

    print("\n✅ Training Complete!")
    print(f"   Best model  : {meta['best_model'].upper()}")
    print(f"   RF  AUC-ROC : {meta['rf_auc_roc']}")
    print(f"   XGB AUC-ROC : {meta['xgb_auc_roc']}")
    print(f"\n   Top features:")
    for i, f in enumerate(meta["top_features"][:10], 1):
        print(f"     {i:2d}. {f}")
    print(f"\n   Saved to: {MODELS_DIR}")
