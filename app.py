# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from src.ingest import load_flights
from src.features import build_features
from src.analysis import slot_stats, top_routes, top_airlines
from src.model import train_delay_classifier, load_model
from src.whatif import shift_by_minutes, queueing_burden

st.set_page_config(page_title="Airport Scheduling Copilot — BOM (Demo)", layout="wide")
st.title("Airport Scheduling Copilot — BOM (Demo)")

@st.cache_data
def _load_and_featurize():
    raw = load_flights("data/processed/combined_clean_ml.csv")
    feat = build_features(raw)
    return raw, feat

# ------------ Load data ------------
raw, df = _load_and_featurize()
st.caption(f"Flights loaded: {len(df)}")

# Quick sanity: show columns in sidebar
with st.sidebar:
    st.subheader("Debug")
    st.write("Columns:", list(df.columns))
    st.write("Sample:", df.head(3))

# ------------ Tabs ------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Delay Lab", "What-If Studio", "Model"])

# ---- Overview ----
with tab1:
    try:
        st.subheader("Sample rows")
        keep_cols = [c for c in ["Flight Number","From","To","STD","ATD","DepartureDelayMin","TimeSlot"] if c in df.columns]
        st.dataframe(df[keep_cols].head(20), use_container_width=True)

        st.subheader("Top Routes by Avg Departure Delay")
        st.dataframe(top_routes(df, n=10), use_container_width=True)

        st.subheader("Top Airlines by Avg Departure Delay")
        st.dataframe(top_airlines(df, n=10), use_container_width=True)
    except Exception as e:
        st.error(f"Overview error: {e}")

# ---- Delay Lab ----
with tab2:
    try:
        st.subheader("P90 Departure Delay by 15-min Slot")
        g = slot_stats(df)
        # Make a simple line chart for robustness (no fancy density heatmap)
        x_col = g.columns[0]  # slot_15 or slot_15_bucket
        fig = px.line(g, x=x_col, y="p90_dep_delay", markers=True)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Busiest Slots (by flights)")
        fig2 = px.bar(g.sort_values("flights", ascending=False).head(20), x=x_col, y="flights")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Best (Green) Windows (p90 ≤ 15 min)")
        st.dataframe(g[g["is_green"]==True].head(20), use_container_width=True)
    except Exception as e:
        st.error(f"Delay Lab error: {e}")

# ---- What-If Studio ----
with tab3:
    try:
        st.subheader("Shift a Flight by ± minutes (Minute-of-Day)")
        flights = df["Flight Number"].dropna().astype(str).unique().tolist()
        if not flights:
            st.info("No flights found.")
        else:
            fsel = st.selectbox("Choose Flight Number", flights)
            delta = st.number_input("Shift minutes (negative to move earlier)", value=5, step=5)
            if st.button("Simulate Shift"):
                before = df.copy()
                after  = shift_by_minutes(df, fsel, int(delta))
                qb_before = queueing_burden(before)
                qb_after  = queueing_burden(after)
                st.metric("Queueing burden Δ (min)", round(qb_after - qb_before, 2))
                st.success("Simulation complete. Scroll up to re-run with a different delta.")
    except Exception as e:
        st.error(f"What-If error: {e}")

# ---- Model ----
with tab4:
    try:
        st.subheader("Train/Load Delay Classifier (delayed > 15 min)")
        colA, colB = st.columns(2)
        with colA:
            if st.button("Train model now"):
                with st.spinner("Training model…"):
                    model = train_delay_classifier(df, out_path="data/model_dep_delay.pkl")
                st.success("Model trained and saved to data/model_dep_delay.pkl")
        with colB:
            if st.button("Load existing model"):
                model = load_model("data/model_dep_delay.pkl")
                st.success("Model loaded.")
        st.caption("Note: Features exclude leakage columns like ATD/ATA/ActualBlockMin/ArrivalDelayMin.")
    except Exception as e:
        st.error(f"Model tab error: {e}")
