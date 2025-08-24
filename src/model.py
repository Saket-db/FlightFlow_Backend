# path: src/model.py

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, classification_report, mean_absolute_error
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer

FEATURES_CAT = ["TimeSlot", "From", "To", "Aircraft", "airline"]
FEATURES_NUM = ["STD_MinOfDay", "DayOfWeek", "IsWeekend", "SchedBlockMin", "slot_load"]

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

CLASSIFIER_OUT_PATH = f"{MODEL_DIR}/delay_classifier.pkl"
P50_OUT_PATH = f"{MODEL_DIR}/delay_p50.pkl"
P90_OUT_PATH = f"{MODEL_DIR}/delay_p90.pkl"


def _build_preprocessor(cat_cols, num_cols) -> ColumnTransformer:
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    pre = ColumnTransformer(transformers=[
        ("cat", cat_pipe, cat_cols),
        ("num", num_pipe, num_cols),
    ])
    return pre


def train_delay_classifier(df: pd.DataFrame, out_path: str = CLASSIFIER_OUT_PATH):
    """
    Train classifier and return (pipeline, metrics_dict) so the UI can display results.
    """
    data = df.copy()

    # Target
    if "dep_delayed_15" not in data.columns:
        if "DepartureDelayMin" in data.columns:
            delay_vals = pd.to_numeric(data["DepartureDelayMin"], errors="coerce")
            data["dep_delayed_15"] = (delay_vals > 15).astype(int)
        else:
            raise ValueError("Neither 'dep_delayed_15' nor 'DepartureDelayMin' found in dataframe.")

    # Available features
    features = [c for c in FEATURES_CAT + FEATURES_NUM if c in data.columns]
    if len(features) < 3:
        raise ValueError(f"Too few features available: {features}")

    data = data.dropna(subset=["dep_delayed_15"])
    if len(data) < 100:
        raise ValueError("Insufficient data for training (< 100 rows)")

    X = data[features]
    y = data["dep_delayed_15"].astype(int)

    # Preprocessor
    cat_cols = [c for c in FEATURES_CAT if c in X.columns]
    num_cols = [c for c in FEATURES_NUM if c in X.columns]
    pre = _build_preprocessor(cat_cols, num_cols)

    pipe = Pipeline([("prep", pre), ("clf", GradientBoostingClassifier(random_state=42, n_estimators=50))])

    # Split
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    pipe.fit(Xtr, ytr)

    # Evaluate
    proba = pipe.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(yte, proba)
    f1 = f1_score(yte, pred)
    report = classification_report(yte, pred, output_dict=True)

    # Save
    joblib.dump(pipe, out_path)

    metrics = {
        "auc": float(auc),
        "f1": float(f1),
        "classes": y.value_counts().to_dict(),
        "report": report,
        "features_used": features
    }
    return pipe, metrics


def load_model(path: str = CLASSIFIER_OUT_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def _quantile_pipe(alpha: float, cat_cols, num_cols) -> Pipeline:
    pre = _build_preprocessor(cat_cols, num_cols)
    gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha, random_state=42)
    return Pipeline([("prep", pre), ("gbr", gbr)])


def train_delay_quantiles(df: pd.DataFrame, p50_path: str = P50_OUT_PATH, p90_path: str = P90_OUT_PATH):
    """
    Train P50 and P90 quantile regressors and return (p50_model, p90_model, metrics_dict).
    """
    data = df.copy()
    if "DepartureDelayMin" not in data.columns:
        raise ValueError("'DepartureDelayMin' not found; needed for quantile regression.")

    features = [c for c in FEATURES_CAT + FEATURES_NUM if c in data.columns]
    if not features:
        raise ValueError("No expected feature columns found. Check your CSV and ingest step.")

    data = data.dropna(subset=["DepartureDelayMin"])
    data["DepartureDelayMin"] = data["DepartureDelayMin"].clip(lower=-60, upper=240)

    X = data[features]
    y = data["DepartureDelayMin"].astype(float)

    cat_cols = [c for c in FEATURES_CAT if c in X.columns]
    num_cols = [c for c in FEATURES_NUM if c in X.columns]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    pipe50 = _quantile_pipe(0.5, cat_cols, num_cols).fit(Xtr, ytr)
    pipe90 = _quantile_pipe(0.9, cat_cols, num_cols).fit(Xtr, ytr)

    # Evaluate (MAE for reference)
    pred50 = pipe50.predict(Xte)
    pred90 = pipe90.predict(Xte)
    mae50 = mean_absolute_error(yte, pred50)
    mae90 = mean_absolute_error(yte, pred90)

    # Save
    joblib.dump(pipe50, p50_path)
    joblib.dump(pipe90, p90_path)

    metrics = {
        "features_used": features,
        "mae_p50": float(mae50),
        "mae_p90": float(mae90),
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte))
    }
    return pipe50, pipe90, metrics


def load_delay_quantiles(p50_path: str = P50_OUT_PATH, p90_path: str = P90_OUT_PATH):
    return joblib.load(p50_path), joblib.load(p90_path)


def predict_delay_quantiles_for_row(row: pd.Series, p50_model, p90_model) -> tuple[float, float]:
    needed = [c for c in FEATURES_CAT + FEATURES_NUM]
    missing = [c for c in needed if c not in row.index]
    if missing:
        return (np.nan, np.nan)
    X = pd.DataFrame([row[needed].to_dict()])
    p50 = float(p50_model.predict(X)[0])
    p90 = float(p90_model.predict(X)[0])
    return p50, p90
