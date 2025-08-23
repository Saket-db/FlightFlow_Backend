# src/model.py
import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier

CAT = ["TimeSlot","From","To","Aircraft","airline"]   # route is high-cardinality; add if needed
NUM = ["STD_MinOfDay","DayOfWeek","IsWeekend","SchedBlockMin","slot_load"]

def train_delay_classifier(df: pd.DataFrame, out_path="data/model_dep_delay.pkl"):
    data = df.dropna(subset=["dep_delayed_15"]).copy()
    X = data[CAT + NUM]
    y = data["dep_delayed_15"].astype(int)

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
            ("num", "passthrough", NUM),
        ]
    )

    clf = GradientBoostingClassifier(random_state=42)
    pipe = Pipeline([("prep", pre), ("clf", clf)])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:,1]
    pred  = (proba >= 0.5).astype(int)

    print("AUC:", roc_auc_score(yte, proba))
    print("F1 :", f1_score(yte, pred))
    print(classification_report(yte, pred))

    joblib.dump(pipe, out_path)
    return pipe

def load_model(path="data/model_dep_delay.pkl"):
    return joblib.load(path)
