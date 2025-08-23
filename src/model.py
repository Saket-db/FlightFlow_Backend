# path: src/model.py

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer

"""
This module provides:
1) A classification model => predicts whether a flight will be delayed > 15 min (dep_delayed_15)
2) Quantile regression models => predict P50 and P90 of departure delay in minutes (risk bands)

Design goals (per our discussion):
- No leakage: DO NOT use ATD/ATA/ActualBlockMin/ArrivalDelayMin as features.
- Robust to missing values with SimpleImputer.
- Same feature schema for classifier and quantiles (easy maintenance).
- Friendly helpers to train, load, and predict.

Expected columns in the input dataframe (at least these):
- Categorical: TimeSlot, From, To, Aircraft, airline
- Numeric:    STD_MinOfDay, DayOfWeek, IsWeekend, SchedBlockMin, slot_load
- Targets:    dep_delayed_15 (for classifier), DepartureDelayMin (for quantiles)
"""

# -----------------------------
# Feature schema (keep in sync)
# -----------------------------
FEATURES_CAT = ["TimeSlot", "From", "To", "Aircraft", "airline"]  # keep 'route' out (often high-cardinality)
FEATURES_NUM = ["STD_MinOfDay", "DayOfWeek", "IsWeekend", "SchedBlockMin", "slot_load"]

MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR, exist_ok=True)

CLASSIFIER_OUT_PATH = f"{MODEL_DIR}/delay_classifier.pkl"
P50_OUT_PATH = f"{MODEL_DIR}/delay_p50.pkl"
P90_OUT_PATH = f"{MODEL_DIR}/delay_p90.pkl"


def _build_preprocessor(cat_cols, num_cols) -> ColumnTransformer:
    """Preprocessor with imputers for robustness."""
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    pre = ColumnTransformer(transformers=[
        ("cat", cat_pipe, [c for c in cat_cols if c in cat_cols]),
        ("num", num_pipe, [c for c in num_cols if c in num_cols]),
    ])
    return pre


# ============================================================
# 1) CLASSIFIER: dep_delayed_15 (delay > 15 minutes) => yes/no
# ============================================================
def train_delay_classifier(df: pd.DataFrame, out_path: str = CLASSIFIER_OUT_PATH):
    """FIXED: Better error handling and feature validation"""
    try:
        # Defensive copy
        data = df.copy()
        
        print(f"Input data shape: {data.shape}")
        print(f"Available columns: {list(data.columns)}")

        # Ensure target exists
        if "dep_delayed_15" not in data.columns:
            if "DepartureDelayMin" in data.columns:
                delay_vals = pd.to_numeric(data["DepartureDelayMin"], errors="coerce")
                data["dep_delayed_15"] = (delay_vals > 15).astype(int)
                print("Created dep_delayed_15 target from DepartureDelayMin")
            else:
                raise ValueError("Neither 'dep_delayed_15' nor 'DepartureDelayMin' found in dataframe.")

        # Check available features
        available_features = []
        for feat in FEATURES_CAT + FEATURES_NUM:
            if feat in data.columns:
                available_features.append(feat)
            else:
                print(f"Warning: Feature '{feat}' not found in data")
        
        if len(available_features) < 3:
            raise ValueError(f"Too few features available: {available_features}")
        
        print(f"Using features: {available_features}")

        # Clean data - remove rows with missing target
        data = data.dropna(subset=["dep_delayed_15"])
        print(f"After removing missing targets: {data.shape}")
        
        if len(data) < 100:
            raise ValueError("Insufficient data for training (< 100 rows)")

        X = data[available_features]
        y = data["dep_delayed_15"].astype(int)
        
        # Check target distribution
        target_counts = y.value_counts()
        print(f"Target distribution: {target_counts.to_dict()}")
        
        if len(target_counts) < 2:
            raise ValueError("Target has only one class - cannot train classifier")

        # Build preprocessor with available features
        cat_cols = [c for c in FEATURES_CAT if c in available_features]
        num_cols = [c for c in FEATURES_NUM if c in available_features]
        
        pre = _build_preprocessor(cat_cols, num_cols)
        clf = GradientBoostingClassifier(random_state=42, n_estimators=50)  # Reduced for speed
        pipe = Pipeline([("prep", pre), ("clf", clf)])

        # Train-test split with stratification if possible
        try:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except ValueError:
            # Fallback without stratification if classes are imbalanced
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Training classifier...")
        pipe.fit(Xtr, ytr)

        # Evaluate
        proba = pipe.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)

        print("=== Classifier Results ===")
        print("AUC:", round(roc_auc_score(yte, proba), 4))
        print("F1 :", round(f1_score(yte, pred), 4))

        # Save model
        joblib.dump(pipe, out_path)
        print(f"Model saved to: {out_path}")
        return pipe
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise

def load_model(path: str = CLASSIFIER_OUT_PATH):
    """FIXED: Safe model loading with error handling"""
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        return joblib.load(path)
    except Exception as e:
        print(f"Failed to load model from {path}: {str(e)}")
        raise

# ============================================================
# 2) QUANTILE REGRESSION: Predict P50/P90 DepartureDelayMin
# ============================================================
def _quantile_pipe(alpha: float, cat_cols, num_cols) -> Pipeline:
    pre = _build_preprocessor(cat_cols, num_cols)
    # GradientBoostingRegressor with quantile loss for uncertainty bands
    gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha, random_state=42)
    return Pipeline([("prep", pre), ("gbr", gbr)])


def train_delay_quantiles(
    df: pd.DataFrame,
    p50_path: str = P50_OUT_PATH,
    p90_path: str = P90_OUT_PATH
):
    """
    Train two regressors to predict the 50th and 90th percentile of DepartureDelayMin.
    We clamp targets to [-60, 240] to reduce outlier impact.
    """
    data = df.copy()

    if "DepartureDelayMin" not in data.columns:
        raise ValueError("'DepartureDelayMin' not found; needed for quantile regression.")

    features = [c for c in FEATURES_CAT + FEATURES_NUM if c in data.columns]
    if not features:
        raise ValueError("No expected feature columns found. Check your CSV and ingest step.")

    # Drop rows without target and clamp target range
    data = data.dropna(subset=["DepartureDelayMin"])
    data["DepartureDelayMin"] = data["DepartureDelayMin"].clip(lower=-60, upper=240)

    X = data[features]
    y = data["DepartureDelayMin"].astype(float)

    cat_cols = [c for c in FEATURES_CAT if c in X.columns]
    num_cols = [c for c in FEATURES_NUM if c in X.columns]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train P50
    pipe50 = _quantile_pipe(0.5, cat_cols, num_cols)
    pipe50.fit(Xtr, ytr)
    joblib.dump(pipe50, p50_path)

    # Train P90
    pipe90 = _quantile_pipe(0.9, cat_cols, num_cols)
    pipe90.fit(Xtr, ytr)
    joblib.dump(pipe90, p90_path)

    print("=== Quantile Regressors ===")
    print(f"Saved P50 -> {p50_path}")
    print(f"Saved P90 -> {p90_path}")
    return pipe50, pipe90


def load_delay_quantiles(
    p50_path: str = P50_OUT_PATH,
    p90_path: str = P90_OUT_PATH
):
    """Load previously trained P50 and P90 models."""
    return joblib.load(p50_path), joblib.load(p90_path)


# ------------------------------------------------------------
# Convenience: predict P50/P90 for a single flight (row slice)
# ------------------------------------------------------------
def predict_delay_quantiles_for_row(row: pd.Series, p50_model, p90_model) -> tuple[float, float]:
    """
    Given a single-row Series with required feature columns,
    return (p50_minutes, p90_minutes). Falls back to np.nan if missing features.
    """
    needed = [c for c in FEATURES_CAT + FEATURES_NUM]
    missing = [c for c in needed if c not in row.index]
    if missing:
        # must be called with row that already contains expected columns
        return (np.nan, np.nan)

    X = pd.DataFrame([row[needed].to_dict()])
    p50 = float(p50_model.predict(X)[0])
    p90 = float(p90_model.predict(X)[0])
    return p50, p90
