# path: app.py
import streamlit as st
import pandas as pd
import plotly.express as px

from src.ingest import load_flights
from src.features import build_features
from src.analysis import slot_stats, top_routes, top_airlines, green_windows
from src.model import train_delay_classifier, load_model, train_delay_quantiles, load_delay_quantiles
from src.whatif import shift_by_minutes, queueing_burden
from src.queueing import RunwayConfig
from src.cascade import cascade_risk
from src.nlp import parse_intent


def enhanced_chatbot_handler(q: str, df: pd.DataFrame, cfg):
    """FIXED: Robust chatbot handler with proper error handling"""
    from src.nlp import parse_intent
    from src.cascade import cascade_risk
    from src.whatif import shift_by_minutes, queueing_burden
    
    try:
        intent = parse_intent(q)
        print(f"Parsed intent: {intent}")  # Debug
        
        if intent["intent"] == "set_mode":
            cfg.mode = intent["mode"]
            return f"✅ Runway mode set to **{cfg.mode}**. Effective μ={cfg.mu():.2f}/min"

        elif intent["intent"] == "set_weather":
            cfg.weather = intent["weather"]
            return f"✅ Weather set to **{cfg.weather}**. Effective μ={cfg.mu():.2f}/min"

        elif intent["intent"] == "busiest":
            try:
                g = slot_stats(df).sort_values("flights", ascending=False).head(10)
                if g.empty:
                    return "❌ No flight data available for analysis."
                
                result = "🔥 **Top Busiest Slots:**\n"
                for _, row in g.head(5).iterrows():
                    slot_name = row.get("slot_label", "Unknown")
                    flights = int(row["flights"])
                    delay = row["p90_dep_delay"]
                    result += f"• {slot_name}: {flights} flights, P90 delay: {delay:.1f} min\n"
                return result
                
            except Exception as e:
                return f"❌ Error analyzing busy slots: {str(e)}"

        elif intent["intent"] == "best":
            try:
                g = green_windows(df, n=10, threshold=30)  # More lenient threshold
                if g.empty:
                    return "❌ No suitable time windows found. All slots have high delays."
                
                result = "✅ **Best Time Windows:**\n"
                for _, row in g.head(5).iterrows():
                    slot_name = row.get("slot_label", "Unknown")
                    flights = int(row["flights"])
                    delay = row["p90_dep_delay"]
                    result += f"• {slot_name}: {flights} flights, P90 delay: {delay:.1f} min\n"
                return result
                
            except Exception as e:
                return f"❌ Error finding best windows: {str(e)}"

        elif intent["intent"] == "cascade":
            try:
                top = cascade_risk(df, top_n=10)
                if top.empty:
                    return "❌ No cascade risk data available."
                
                result = "⚠️ **High Cascade Risk Flights:**\n"
                for _, row in top.head(5).iterrows():
                    flight = row["Flight Number"]
                    route = f"{row['From']}-{row['To']}"
                    score = row["cascade_score"]
                    result += f"• {flight} ({route}): Risk score {score:.2f}\n"
                return result
                
            except Exception as e:
                return f"❌ Error analyzing cascade risk: {str(e)}"

        elif intent["intent"] == "shift_by":
            try:
                flight = intent["flight"]
                mins = intent["mins"]
                
                # Check if flight exists
                if flight not in df["Flight Number"].astype(str).values:
                    return f"Flight {flight} not found in schedule."

                before = df.copy()
                after = shift_by_minutes(df, flight, mins)
                qb_before = queueing_burden(before, cfg)
                qb_after = queueing_burden(after, cfg)
                delta = qb_after - qb_before
                
                impact = "Improved" if delta < 0 else "Increased" if delta > 0 else "No change"
                return f"{impact} queueing burden by {delta:.1f} min for shifting {flight} by {mins} min"
                
            except Exception as e:
                return f"Error simulating shift: {str(e)}"

        elif intent["intent"] == "predict":
            try:
                flight = intent["flight"]
                flight_data = df[df["Flight Number"].astype(str) == flight]
                
                if flight_data.empty:
                    return f"Flight {flight} not found."
                
                # Try to load models
                try:
                    p50, p90 = load_delay_quantiles()
                except:
                    return "Prediction models not available. Train models first."
                
                # Make prediction
                row = flight_data.iloc[0]
                required_cols = ["TimeSlot", "From", "To", "Aircraft", "airline",
                               "STD_MinOfDay", "DayOfWeek", "IsWeekend", "SchedBlockMin", "slot_load"]
                
                missing_cols = [col for col in required_cols if col not in flight_data.columns]
                if missing_cols:
                    return f"Missing data columns for prediction: {missing_cols}"
                
                X = flight_data[required_cols].iloc[[0]]
                d50 = float(p50.predict(X)[0])
                d90 = float(p90.predict(X)[0])
                
                return f"**Delay Prediction for {flight}:**\n• P50: {d50:.1f} min\n• P90: {d90:.1f} min"
                
            except Exception as e:
                return f" Error predicting delay: {str(e)}"

        else:
            return ("ℹ I can help with:\n"
                   "• 'busiest' - Find peak times\n"
                   "• 'best' - Find optimal slots\n" 
                   "• 'cascade' - Analyze risk\n"
                   "• 'shift AI2509 by 10 min' - Simulate changes\n"
                   "• 'predict delay for AI2509' - Forecast delays\n"
                   "• 'set runway mode to segregated'\n"
                   "• 'set weather to rain'")
            
    except Exception as e:
        return f" Chatbot error: {str(e)}"

st.set_page_config(page_title="Airport Scheduling Copilot — BOM (Demo)", layout="wide")
st.title("Airport Scheduling Copilot — BOM (Demo)")

@st.cache_data
def _load_and_featurize():
    raw = load_flights("data/processed/combined_clean_ml.csv")
    feat = build_features(raw)
    return raw, feat

raw, df = _load_and_featurize()
st.caption(f"Flights loaded: {len(df)}")

# state
if "runway_cfg" not in st.session_state:
    st.session_state.runway_cfg = RunwayConfig()
if "chat" not in st.session_state:
    st.session_state.chat = []

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Delay Lab", "What-If Studio", "Model", "💬 Chatbot"])

# -------- Overview --------
with tab1:
    try:
        st.subheader("TimeSlot coverage")
        if "TimeSlot" in df.columns:
            st.bar_chart(df["TimeSlot"].value_counts())
        else:
            st.info("No TimeSlot column found; showing first 20 rows below.")

        st.subheader("Quick preview by TimeSlot")
        if "TimeSlot" in df.columns:
            slots = df["TimeSlot"].dropna().unique().tolist()
            slot_pick = st.selectbox("Select window", slots)
            keep_cols = [c for c in ["Flight Number","From","To","STD","ATD","DepartureDelayMin","TimeSlot"] if c in df.columns]
            st.dataframe(df[df["TimeSlot"]==slot_pick][keep_cols].head(20), use_container_width=True)
        else:
            keep_cols = [c for c in ["Flight Number","From","To","STD","ATD","DepartureDelayMin"] if c in df.columns]
            st.dataframe(df[keep_cols].head(20), use_container_width=True)

        st.subheader("Top Routes by Avg Departure Delay")
        st.dataframe(top_routes(df, n=10), use_container_width=True)

        st.subheader("Top Airlines by Avg Departure Delay")
        st.dataframe(top_airlines(df, n=10), use_container_width=True)
    except Exception as e:
        st.error(f"Overview error: {e}")

# -------- Delay Lab --------
with tab2:
    try:
        st.subheader("P90 Departure Delay by 15-min Slot")
        g = slot_stats(df)
        x_col = "slot_label" if "slot_label" in g.columns else g.columns[0]
        fig = px.line(g, x=x_col, y="p90_dep_delay", markers=True)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Busiest Slots (by flights)")
        fig2 = px.bar(g.sort_values("flights", ascending=False).head(20), x=x_col, y="flights")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Best (Green) Windows (p90 ≤ 15 min)")
        st.dataframe(green_windows(df, n=20), use_container_width=True)
    except Exception as e:
        st.error(f"Delay Lab error: {e}")

# -------- What-If Studio --------
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
                qb_before = queueing_burden(before, st.session_state.runway_cfg)
                qb_after  = queueing_burden(after,  st.session_state.runway_cfg)
                st.metric("Total queueing burden (min) — before", round(qb_before, 2))
                st.metric("Total queueing burden (min) — after",  round(qb_after, 2))
                st.metric("Δ (after − before)", round(qb_after - qb_before, 2))
                st.success("Simulation complete.")
    except Exception as e:
        st.error(f"What-If error: {e}")

# -------- Model --------
with tab4:
    try:
        st.subheader("Train / Load Models")
        colA, colB, colC = st.columns(3)
        with colA:
            if st.button("Train classifier"):
                with st.spinner("Training classifier…"):
                    _ = train_delay_classifier(df)
                st.success("Classifier trained ✔")
        with colB:
            if st.button("Train quantiles (P50/P90)"):
                with st.spinner("Training quantile regressors…"):
                    _ = train_delay_quantiles(df)
                st.success("Quantile models trained ✔")
        with colC:
            if st.button("Load classifier"):
                _ = load_model()
                st.success("Classifier loaded ✔")
        st.caption("Classifier → delayed >15min. Quantiles → P50/P90 delay. No ATD/ATA leakage used.")
    except Exception as e:
        st.error(f"Model tab error: {e}")

# -------- Chatbot --------
with tab5:
    st.subheader("AI Ops Copilot 🤖")
    st.caption("Natural language interface for flight operations analysis")
    
    # Quick action buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🔥 Busiest"):
            st.session_state.chat.append({"role": "user", "content": "busiest"})
            response = enhanced_chatbot_handler("busiest", df, st.session_state.runway_cfg)
            st.session_state.chat.append({"role": "assistant", "content": response})
            st.rerun()
    
    with col2:
        if st.button("✅ Best Slots"):
            st.session_state.chat.append({"role": "user", "content": "best"})
            response = enhanced_chatbot_handler("best", df, st.session_state.runway_cfg)
            st.session_state.chat.append({"role": "assistant", "content": response})
            st.rerun()
    
    with col3:
        if st.button("⚠️ Cascade Risk"):
            st.session_state.chat.append({"role": "user", "content": "cascade"})
            response = enhanced_chatbot_handler("cascade", df, st.session_state.runway_cfg)
            st.session_state.chat.append({"role": "assistant", "content": response})
            st.rerun()
    
    with col4:
        # Runway config display
        cfg = st.session_state.runway_cfg
        st.metric("Current Config", f"{cfg.mode.title()}, {cfg.weather}", f"μ={cfg.mu():.2f}/min")

    st.divider()
    
    # Chat history
    chat_container = st.container(height=400)
    with chat_container:
        if not st.session_state.chat:
            st.info("👋 Hello! I can help you analyze flight schedules. Try asking about 'busiest times' or 'best slots'.")
        
        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask me about flight schedules... (e.g., 'busiest', 'predict delay for AI2509')"):
        # Add user message
        st.session_state.chat.append({"role": "user", "content": prompt})
        
        # Generate and add assistant response
        with st.spinner("Analyzing..."):
            response = enhanced_chatbot_handler(prompt, df, st.session_state.runway_cfg)
        
        st.session_state.chat.append({"role": "assistant", "content": response})
        st.rerun()

    # Clear chat button
    if st.button("🗑️ Clear Chat", type="secondary"):
        st.session_state.chat = []
        st.rerun()
    
    # Example queries
    with st.expander("💡 Example Queries"):
        st.code("""
# Traffic Analysis
"busiest"
"best"
"cascade"

# Specific Flight Operations  
"shift AI2509 by 15 min"
"shift UA101 by -10 min"
"predict delay for AI2509"

# Configuration
"set runway mode to segregated"
"set weather to rain"
        """)