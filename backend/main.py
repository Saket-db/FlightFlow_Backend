# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import threading
from pathlib import Path
import sys
import os

# ------------------------------------------------------
# Paths & import setup
# ------------------------------------------------------
# project root (one level above backend/)
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DATA_PATH = ROOT_DIR / "data" / "processed" / "combined_clean_ml.csv"

# Reuse your project modules
from src.ingest import load_flights
from src.features import build_features
from src.analysis import slot_stats, green_windows, top_routes, top_airlines
from src.model import load_model, load_delay_quantiles, train_delay_classifier, train_delay_quantiles
from src.whatif import shift_by_minutes, queueing_burden, compute_queueing_stats
from src.queueing import RunwayConfig
from src.cascade import cascade_risk
from src.nlp import parse_intent


# ------------------------------------------------------
# App state
# ------------------------------------------------------
class AppState:
    def __init__(self):
        # Use the absolute path so it works no matter where uvicorn is launched
        self.raw = load_flights(str(DATA_PATH))  # Fixed: Use DATA_PATH variable
        self.df = build_features(self.raw)
        self.lock = threading.Lock()
        self.cfg = RunwayConfig()
        self.clf = None
        self.p50 = None
        self.p90 = None


state = AppState()

app = FastAPI(title="Airport Scheduling Copilot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------
# Schemas
# ------------------------------------------------------
class Health(BaseModel):
    ok: bool

class RunwayConfigIn(BaseModel):
    mode: Optional[str] = None        # single | segregated | parallel_independent
    weather: Optional[str] = None     # VMC | IMC | Rain | Storm

class RouteQuery(BaseModel):
    origin: str
    dest: str
    top_n: int = 10
    kind: str = "best"                # best | worst

class FlightPredictIn(BaseModel):
    flight: str

class ShiftRequest(BaseModel):
    flight: str
    minutes: int

class ChatIn(BaseModel):
    message: str


# ------------------------------------------------------
# Helpers
# ------------------------------------------------------
def ensure_models_loaded():
    if state.p50 is None or state.p90 is None:
        try:
            state.p50, state.p90 = load_delay_quantiles()
        except Exception:
            raise HTTPException(status_code=400, detail="Quantile models not found. Train models first.")
    return state.p50, state.p90

def required_features_block(df_block: pd.DataFrame) -> pd.DataFrame:
    req = ["TimeSlot","From","To","Aircraft","airline","STD_MinOfDay","DayOfWeek","IsWeekend","SchedBlockMin","slot_load"]
    x = df_block.copy()

    if "airline" not in x.columns and "Flight Number" in x.columns:
        x["airline"] = x["Flight Number"].astype(str).str.extract(r"^([A-Z]+)")

    if "slot_load" not in x.columns or x["slot_load"].isna().all():
        b = (pd.to_numeric(x.get("STD_MinOfDay"), errors="coerce").fillna(-1) // 15) * 15
        x["slot_15_bucket"] = b
        x["slot_load"] = x.groupby("slot_15_bucket")["Flight Number"].transform("count")

    missing = [c for c in req if c not in x.columns]
    if missing:
        return pd.DataFrame(columns=req)
    return x[req]


# ------------------------------------------------------
# Endpoints
# ------------------------------------------------------
@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Airport Scheduling Copilot API",
        "version": "1.0.0",
        "status": "running",
        "data_loaded": len(state.df) > 0 if hasattr(state, 'df') else False,
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "openapi": "/openapi.json",
            "dataset_meta": "/dataset/meta",
            "analysis": {
                "slots": "/analysis/slots",
                "green_windows": "/analysis/green",
                "top_routes": "/analysis/top_routes",
                "top_airlines": "/analysis/top_airlines",
                "cascade_risk": "/analysis/cascade"
            },
            "predictions": "/predict/quantiles",
            "whatif": "/whatif/shift",
            "chat": "/chat"
        }
    }

@app.get("/health", response_model=Health)
def health():
    """Health check endpoint"""
    return Health(ok=True)

@app.get("/dataset/meta")
def dataset_meta():
    """Get dataset metadata"""
    with state.lock:
        return {
            "rows": int(len(state.df)),
            "columns": list(state.df.columns),
            "from_top": state.df["From"].dropna().astype(str).str.upper().value_counts().head(10).to_dict(),
            "to_top": state.df["To"].dropna().astype(str).str.upper().value_counts().head(10).to_dict(),
        }

@app.get("/flights")
def get_flights():
    """Get list of available flights"""
    with state.lock:
        flights = state.df["Flight Number"].dropna().unique().tolist()
        return {"flights": flights}

@app.post("/dataset/reload")
def dataset_reload():
    """Reload dataset from file"""
    with state.lock:
        state.raw = load_flights(str(DATA_PATH))
        state.df = build_features(state.raw)
    return {"rows": int(len(state.df))}

@app.post("/config/runway")
def set_runway(cfg: RunwayConfigIn):
    """Configure runway settings"""
    with state.lock:
        if cfg.mode: 
            state.cfg.mode = cfg.mode
        if cfg.weather: 
            state.cfg.weather = cfg.weather
        info = state.cfg.get_operational_info()
        info["mu_per_min"] = round(state.cfg.mu(), 3)
        return info

@app.get("/analysis/slots")
def by_slots():
    """Get slot statistics analysis"""
    with state.lock:
        g = slot_stats(state.df)
        # Replace NaN with None so JSON serialization does not fail
        g = g.replace({np.nan: None})
    return g.to_dict(orient="records")

@app.get("/analysis/green")
def green(n: int = 20):
    """Get green window analysis"""
    with state.lock:
        g = green_windows(state.df, n=n)
    return g.to_dict(orient="records")

@app.get("/analysis/top_routes")
def routes(n: int = 10):
    """Get top routes analysis"""
    with state.lock:
        t = top_routes(state.df, n=n)
    return t.to_dict(orient="records")

@app.get("/analysis/top_airlines")
def airlines(n: int = 10):
    """Get top airlines analysis"""
    with state.lock:
        t = top_airlines(state.df, n=n)
    return t.to_dict(orient="records")

@app.get("/analysis/cascade")
def cascade(n: int = 10):
    """Get cascade risk analysis"""
    with state.lock:
        t = cascade_risk(state.df, top_n=n)
        # Normalize output field names for frontend expectations
        out = (
            t.rename(columns={
                "Flight Number": "flight_id",
                "From": "origin",
                "To": "destination",
                "SchedBlockMin": "sched_block_min",
                "slot_p90": "p90_dep_delay"
            })
        )
    return out.replace({np.nan: None}).to_dict(orient="records")

@app.post("/predict/quantiles")
def predict_quantiles(inp: FlightPredictIn):
    """Predict delay quantiles for a specific flight.
    If trained models are unavailable, fall back to empirical quantiles from similar historical data.
    Also returns a simple delay probability and confidence score based on sample size.
    """
    with state.lock:
        df = state.df
        sub = df[df["Flight Number"].astype(str) == inp.flight]
        if sub.empty:
            raise HTTPException(status_code=404, detail=f"Flight {inp.flight} not found.")

        # Build context cohort: same airline and same 15-min bucket
        row = sub.iloc[0]
        airline = str(row.get("airline", "")).upper()
        # derive bucket robustly
        if "slot_15" in df.columns and df["slot_15"].notna().any() and pd.notna(row.get("slot_15")):
            key_col = "slot_15"
            key_val = row["slot_15"]
            cohort = df[(df.get("airline").astype(str).str.upper() == airline) & (df[key_col] == key_val)]
        else:
            bucket = (pd.to_numeric(pd.Series([row.get("STD_MinOfDay")]), errors="coerce").fillna(-1) // 15 * 15).iloc[0]
            buckets = (pd.to_numeric(df.get("STD_MinOfDay"), errors="coerce").fillna(-1) // 15) * 15
            cohort = df[(df.get("airline").astype(str).str.upper() == airline) & (buckets == bucket)]

        # Clean cohort delays
        delays = pd.to_numeric(cohort.get("DepartureDelayMin"), errors="coerce").dropna()
        n = int(delays.shape[0])

        # Try model-based prediction; else empirical quantiles
        try:
            p50_model, p90_model = ensure_models_loaded()
            X = required_features_block(sub.iloc[[0]])
            if X.empty:
                raise HTTPException(status_code=400, detail="Missing required features for prediction.")
            pred_p50 = float(p50_model.predict(X)[0])
            pred_p90 = float(p90_model.predict(X)[0])
        except HTTPException:
            # Empirical fallback
            if n == 0:
                pred_p50 = float(pd.to_numeric(df.get("DepartureDelayMin"), errors="coerce").dropna().quantile(0.5))
                pred_p90 = float(pd.to_numeric(df.get("DepartureDelayMin"), errors="coerce").dropna().quantile(0.9))
            else:
                pred_p50 = float(delays.quantile(0.5))
                pred_p90 = float(delays.quantile(0.9))

        # Probability of delay (>0) in cohort; confidence from sample size
        if n > 0:
            prob = float((delays > 0).mean())
            # confidence: saturating function of sample size
            confidence = float(min(0.99, n / (n + 20)))
        else:
            prob = 0.5
            confidence = 0.5

    return {
        "flight": inp.flight,
        "p50": round(pred_p50, 2),
        "p90": round(pred_p90, 2),
        "delay_probability": round(prob, 3),
        "confidence": round(confidence, 3),
        "sample_size": n,
    }

@app.post("/route/rank")
def route_rank(q: RouteQuery):
    """Rank flights on a specific route"""
    with state.lock:
        df = state.df
        mask = (df["From"].astype(str).str.upper() == q.origin.upper()) & \
               (df["To"].astype(str).str.upper() == q.dest.upper())
        sub = df[mask]
        if sub.empty:
            raise HTTPException(status_code=404, detail="No flights for route.")
        agg = (sub.groupby(["Flight Number","From","To"])
                 .agg(flights=("Flight Number","count"),
                      avg_dep_delay=("DepartureDelayMin","mean"))
                 .reset_index())
        agg["avg_dep_delay"] = pd.to_numeric(agg["avg_dep_delay"], errors="coerce")
        agg = agg.dropna(subset=["avg_dep_delay"])
        ascending = True if q.kind == "best" else False
        out = agg.sort_values("avg_dep_delay", ascending=ascending).head(q.top_n)
    return out.to_dict(orient="records")

@app.post("/whatif/shift")
def whatif_shift(req: ShiftRequest):
    """Perform what-if analysis by shifting flight time"""
    with state.lock:
        if req.flight not in state.df["Flight Number"].astype(str).values:
            raise HTTPException(status_code=404, detail="Flight not found.")

        before = state.df.copy()
        after = shift_by_minutes(state.df, req.flight, req.minutes)

        # Use 1-minute buckets for sensitive what-if comparisons so small shifts change counts
        qb_before = queueing_burden(before, state.cfg, slot_minutes=1)
        qb_after = queueing_burden(after, state.cfg, slot_minutes=1)

        # Detailed queueing stats for frontend: totals + per-slot breakdown (sanitized)
        stats_before = compute_queueing_stats(before, state.cfg, slot_minutes=1)
        stats_after = compute_queueing_stats(after, state.cfg, slot_minutes=1)

        def sanitize_stats(s, top_n=10):
            # Make sure values are plain Python types and limit per-slot list
            per_slot = s.get("per_slot", []) or []
            sorted_slots = sorted(per_slot, key=lambda x: -float(x.get("total_wait", 0.0)))[:top_n]
            sanitized = []
            for slot in sorted_slots:
                try:
                    b = int(slot.get("bucket", -1))
                except Exception:
                    b = -1
                sanitized.append({
                    "bucket": b,
                    "count": int(slot.get("count", 0)),
                    "per_flight_wait": float(slot.get("per_flight_wait", 0.0)),
                    "total_wait": float(slot.get("total_wait", 0.0)),
                })
            return {
                "total_wait": float(s.get("total_wait", 0.0)),
                "total_flights": int(s.get("total_flights", 0)),
                "avg_wait_per_flight": float(s.get("avg_wait_per_flight", 0.0)),
                "top_slots": sanitized,
            }

        stats_before_clean = sanitize_stats(stats_before)
        stats_after_clean = sanitize_stats(stats_after)
    return {
        "flight": req.flight,
        "minutes": req.minutes,
        "queueing_burden_before": round(float(qb_before), 2),
        "queueing_burden_after": round(float(qb_after), 2),
        "delta": round(float(qb_after - qb_before), 2),
        "stats_before": stats_before_clean,
        "stats_after": stats_after_clean,
    }

# Optional training endpoints (protect in prod)
@app.post("/train/classifier")
def train_classifier():
    """Train delay classifier model"""
    with state.lock:
        try:
            _ = train_delay_classifier(state.df)
            return {"status": "ok", "message": "Classifier trained and saved"}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

@app.post("/train/quantiles")
def train_quantiles():
    """Train quantile regression models"""
    with state.lock:
        try:
            p50, p90 = train_delay_quantiles(state.df)
            state.p50, state.p90 = p50, p90
            return {"status": "ok", "message": "Quantile models trained and saved"}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

# Simple chatbot passthrough that returns markdown
@app.post("/chat")
def chat(inp: ChatIn):
    """Chat interface for natural language queries"""
    with state.lock:
        reply = enhanced_chatbot_handler(inp.message, state.df, state.cfg)
    return {"markdown": reply}


# ------------------------------------------------------
# Chatbot handler without Streamlit
# ------------------------------------------------------
def enhanced_chatbot_handler(q: str, df: pd.DataFrame, cfg: RunwayConfig) -> str:
    """Enhanced chatbot handler that processes natural language queries"""
    try:
        intent = parse_intent(q)
        if intent["intent"] == "set_mode":
            cfg.mode = intent["mode"]
            return f"Runway mode set to **{cfg.mode}**. μ={cfg.mu():.2f}/min"
        if intent["intent"] == "set_weather":
            cfg.weather = intent["weather"]
            return f"Weather set to **{cfg.weather}**. μ={cfg.mu():.2f}/min"
        if intent["intent"] == "busiest":
            g = slot_stats(df).sort_values("flights", ascending=False).head(10)
            if g.empty:
                return "No data."
            t = g[["slot_label","flights","avg_dep_delay","p50_dep_delay","p90_dep_delay"]]
            return t.to_markdown(index=False)
        if intent["intent"] == "best":
            g = green_windows(df, n=10)
            if g.empty:
                return "No green windows."
            t = g[["slot_label","flights","p90_dep_delay","p50_dep_delay","avg_dep_delay"]]
            return t.to_markdown(index=False)
        if intent["intent"] == "cascade":
            t = cascade_risk(df, top_n=10)
            if t.empty:
                return "No cascade risk data."
            keep = (
                t.rename(columns={"Flight Number":"Flight"})
                 .loc[:,["Flight","From","To","SchedBlockMin","slot_load","slot_p90","cascade_score"]]
            )
            return keep.to_markdown(index=False)
        if intent["intent"] == "predict":
            flight = intent["flight"]
            sub = df[df["Flight Number"].astype(str) == flight]
            if sub.empty:
                return f"Flight {flight} not found."
            try:
                p50, p90 = load_delay_quantiles()
            except Exception:
                return "Quantile models not found."
            X = required_features_block(sub.iloc[[0]])
            if X.empty:
                return "Missing required features."
            d50 = float(p50.predict(X)[0]); d90 = float(p90.predict(X)[0])
            return pd.DataFrame([{"Flight": flight, "P50 (min)": f"{d50:.1f}", "P90 (min)": f"{d90:.1f}"}]).to_markdown(index=False)
        if intent["intent"] == "route_rank":
            kind, o, d, top_n = intent["kind"], intent["origin"], intent["dest"], intent["top_n"]
            mask = (df["From"].astype(str).str.upper() == o) & (df["To"].astype(str).str.upper() == d)
            sub = df[mask]
            if sub.empty:
                return f"No flights on {o}->{d}."
            agg = (sub.groupby(["Flight Number","From","To"])
                     .agg(flights=("Flight Number","count"),
                          avg_dep_delay=("DepartureDelayMin","mean"))
                     .reset_index())
            agg["avg_dep_delay"] = pd.to_numeric(agg["avg_dep_delay"], errors="coerce")
            agg = agg.dropna(subset=["avg_dep_delay"])
            ascending = True if kind == "best" else False
            out = agg.sort_values("avg_dep_delay", ascending=ascending).head(top_n)
            out = out.rename(columns={"avg_dep_delay":"Avg Dep Delay (min)","Flight Number":"Flight"})
            return out.to_markdown(index=False)
        return ("I can help with: busiest, best, cascade, "
                "predict delay for <flight>, best/worst flights on <ORIG->DEST>, "
                "set runway mode to <mode>, set weather to <weather>")
    except Exception as e:
        return f"Chatbot error: {e}"


# ------------------------------------------------------
# Run server (optional, for development)
# ------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)