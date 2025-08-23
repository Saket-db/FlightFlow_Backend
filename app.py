# app.py (relevant parts)
import streamlit as st
import pandas as pd
import plotly.express as px
from src.ingest import load_flights
from src.features import build_features
from src.analysis import slot_stats, top_routes, top_airlines
from src.model import train_delay_classifier, load_model
from src.whatif import shift_by_minutes, queueing_burden

st.set_page_config(page_title="Airport Scheduling Copilot", layout="wide")
st.title("Airport Scheduling Copilot — BOM (Demo)")

@st.cache_data
def _load_and_featurize():
    raw = load_flights("data/processed/combined_clean_ml.csv")
    feat = build_features(raw)
    return raw, feat

raw, df = _load_and_featurize()
st.caption(f"Flights loaded: {len(df)}")
