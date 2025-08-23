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
    """
    Robust chatbot handler with route-ranking, predictions, what-if,
    busiest/best windows, cascade risk, and runway config.
    Returns a markdown-friendly string for Streamlit chat.
    """
    from src.nlp import parse_intent
    from src.analysis import slot_stats, green_windows
    from src.cascade import cascade_risk
    from src.whatif import shift_by_minutes, queueing_burden
    from src.model import load_delay_quantiles

    try:
        intent = parse_intent(q)

        # --- runway config ---
        if intent["intent"] == "set_mode":
            cfg.mode = intent["mode"]
            return f"✅ Runway mode set to **{cfg.mode}**. Effective μ={cfg.mu():.2f}/min"

        if intent["intent"] == "set_weather":
            cfg.weather = intent["weather"]
            return f"✅ Weather set to **{cfg.weather}**. Effective μ={cfg.mu():.2f}/min"

        # --- analytics: busiest ---
        if intent["intent"] == "busiest":
            g = slot_stats(df).sort_values("flights", ascending=False).head(10)
            if g.empty:
                return "❌ No flight data available for analysis."
            lines = ["🔥 **Top Busiest Slots:**"]
            for _, r in g.iterrows():
                slot_name = r.get("slot_label", str(r.iloc[0]))
                lines.append(f"• {slot_name}: {int(r['flights'])} flights, P90 delay {r['p90_dep_delay']:.1f} min")
            return "\n".join(lines)

        # --- analytics: best windows ---
        if intent["intent"] == "best":
            g = green_windows(df, n=10)
            if g.empty:
                return "❌ No green windows (P90 ≤ 15 min) found."
            lines = ["✅ **Best Time Windows:**"]
            for _, r in g.iterrows():
                slot_name = r.get("slot_label", str(r.iloc[0]))
                lines.append(f"• {slot_name}: {int(r['flights'])} flights, P90 delay {r['p90_dep_delay']:.1f} min")
            return "\n".join(lines)

        # --- cascade risk ---
        if intent["intent"] == "cascade":
            top = cascade_risk(df, top_n=10)
            if top.empty:
                return "❌ No cascade risk data available."
            lines = ["⚠️ **High Cascade Risk Flights:**"]
            for _, r in top.iterrows():
                lines.append(f"• {r['Flight Number']} ({r['From']}→{r['To']}): Risk {r['cascade_score']:.2f}")
            return "\n".join(lines)

        # --- what-if: shift-by ---
        if intent["intent"] == "shift_by":
            flight = intent["flight"]
            mins = intent["mins"]
            if flight not in df["Flight Number"].astype(str).values:
                return f"❌ Flight **{flight}** not found in schedule."

            before = df.copy()
            after = shift_by_minutes(df, flight, mins)
            # recompute burden using counts (queueing_burden does this)
            qb_before = queueing_burden(before, cfg)
            qb_after  = queueing_burden(after,  cfg)
            delta = qb_after - qb_before
            impact = "Improved" if delta < 0 else "Increased" if delta > 0 else "No change"
            return f"{impact} queueing burden by **{delta:.1f} min** for shifting **{flight}** by **{mins} min** (mode={cfg.mode}, weather={cfg.weather})."

        # --- predictions: P50/P90 delay ---
        if intent["intent"] == "predict":
            flight = intent["flight"]
            row_df = df[df["Flight Number"].astype(str) == flight]
            if row_df.empty:
                return f"❌ Flight **{flight}** not found."

            # ensure required feature cols exist (recompute slot_load if missing)
            if "slot_load" not in row_df.columns or row_df["slot_load"].isna().all():
                tmp = df.copy()
                tmp["slot_15_bucket"] = (pd.to_numeric(tmp.get("STD_MinOfDay"), errors="coerce").fillna(-1) // 15) * 15
                tmp["slot_load"] = tmp.groupby("slot_15_bucket")["Flight Number"].transform("count")
                row_df = tmp[tmp["Flight Number"].astype(str) == flight]

            required = ["TimeSlot","From","To","Aircraft","airline",
                        "STD_MinOfDay","DayOfWeek","IsWeekend","SchedBlockMin","slot_load"]
            missing = [c for c in required if c not in row_df.columns]
            if missing:
                return f"❌ Missing columns for prediction: {missing}"

            try:
                p50, p90 = load_delay_quantiles()
            except Exception:
                return "❌ Quantile models not found. Train them in the *Model* tab first."

            X = row_df[required].iloc[[0]]
            d50 = float(p50.predict(X)[0]); d90 = float(p90.predict(X)[0])
            return (f"**Delay Prediction for {flight}:**\n"
                    f"• P50: **{d50:.1f} min**\n"
                    f"• P90: **{d90:.1f} min**")

        # --- NEW: best / worst flights on a route ---
        if intent["intent"] == "route_rank":
            kind  = intent["kind"]      # 'best' | 'worst'
            o, d  = intent["origin"], intent["dest"]
            top_n = intent["top_n"]

            # prefer the 'route' column if present, else use From/To
            if "route" in df.columns:
                route_key = f"{o}->{d}"
                sub = df[df["route"].astype(str).str.upper() == route_key]
            else:
                sub = df[(df["From"].astype(str).str.upper() == o) &
                         (df["To"].astype(str).str.upper() == d)]

            if sub.empty:
                return f"❌ No flights found on route **{o}->{d}**."

            # aggregate by flight number
            agg = (sub.groupby(["Flight Number","From","To"])
                     .agg(flights=("Flight Number","count"),
                          avg_dep_delay=("DepartureDelayMin","mean"))
                     .reset_index())
            agg["avg_dep_delay"] = pd.to_numeric(agg["avg_dep_delay"], errors="coerce")
            agg = agg.dropna(subset=["avg_dep_delay"])

            if agg.empty:
                return f"❌ No delay data for route **{o}->{d}**."

            ascending = True if kind == "best" else False
            agg = agg.sort_values("avg_dep_delay", ascending=ascending).head(top_n)

            header = f"**{kind.title()} flights on {o}->{d} (top {len(agg)})**"
            lines = [header]
            for _, r in agg.iterrows():
                lines.append(f"• {r['Flight Number']} ({r['From']}→{r['To']}): "
                             f"avg dep delay **{r['avg_dep_delay']:.1f} min** over {int(r['flights'])} flights")
            return "\n".join(lines)

        # --- fallback help ---
        return ("ℹ I can help with:\n"
                "• 'busiest' — peak times\n"
                "• 'best' — optimal windows\n"
                "• 'cascade' — high-impact flights\n"
                "• 'shift AI2509 by 10 min' — simulate queueing impact\n"
                "• 'predict delay for AI2509' — P50/P90 forecast\n"
                "• 'best flights on BOM->DEL', 'worst flights from DEL to BLR top 5'\n"
                "• 'set runway mode to segregated' / 'set weather to rain'")

    except Exception as e:
        return f"❌ Chatbot error: {str(e)}"

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

    # --- Quick action buttons ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Busiest"):
            st.session_state.chat.append({"role": "user", "content": "busiest"})
            response = enhanced_chatbot_handler("busiest", df, st.session_state.runway_cfg)
            st.session_state.chat.append({"role": "assistant", "content": response})
            st.rerun()

    with col2:
        if st.button("Best Slots"):
            st.session_state.chat.append({"role": "user", "content": "best"})
            response = enhanced_chatbot_handler("best", df, st.session_state.runway_cfg)
            st.session_state.chat.append({"role": "assistant", "content": response})
            st.rerun()

    with col3:
        if st.button("Cascade Risk"):
            st.session_state.chat.append({"role": "user", "content": "cascade"})
            response = enhanced_chatbot_handler("cascade", df, st.session_state.runway_cfg)
            st.session_state.chat.append({"role": "assistant", "content": response})
            st.rerun()

    with col4:
        # Runway config display
        cfg = st.session_state.runway_cfg
        try:
            st.metric("Current Config", f"{cfg.mode.title()}, {cfg.weather}", f"μ={cfg.mu():.2f}/min")
        except AttributeError:
            st.error("RunwayConfig.mu() not found. Please update src/queueing.py with the latest version.")

    st.divider()

    # --- Route ranking quick tool (best/worst on a route) ---
    if {"From", "To"} <= set(df.columns):
        st.markdown("**Route Ranking (Quick Tool)**")
        rcol1, rcol2, rcol3, rcol4 = st.columns([1,1,1,1])
        with rcol1:
            origins = sorted(df["From"].dropna().astype(str).str.upper().unique().tolist())
            origin_sel = st.selectbox("From", origins, index=0)
        with rcol2:
            dests = sorted(df["To"].dropna().astype(str).str.upper().unique().tolist())
            dest_sel = st.selectbox("To", dests, index=min(1, len(dests)-1))
        with rcol3:
            top_n = st.number_input("Top N", min_value=1, max_value=50, value=10, step=1)
        with rcol4:
            colb1, colb2 = st.columns(2)
            with colb1:
                if st.button("Best on Route"):
                    prompt = f"best flights on {origin_sel}->{dest_sel} top {int(top_n)}"
                    st.session_state.chat.append({"role": "user", "content": prompt})
                    resp = enhanced_chatbot_handler(prompt, df, st.session_state.runway_cfg)
                    st.session_state.chat.append({"role": "assistant", "content": resp})
                    st.rerun()
            with colb2:
                if st.button("Worst on Route"):
                    prompt = f"worst flights on {origin_sel}->{dest_sel} top {int(top_n)}"
                    st.session_state.chat.append({"role": "user", "content": prompt})
                    resp = enhanced_chatbot_handler(prompt, df, st.session_state.runway_cfg)
                    st.session_state.chat.append({"role": "assistant", "content": resp})
                    st.rerun()

    st.divider()

    # --- Chat history ---
    if not st.session_state.chat:
        st.info("Hello! I can help you analyze flight schedules. Try asking about 'busiest', 'best', or 'best flights on BOM->DEL'.")
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # --- Chat input ---
    prompt = st.chat_input("Ask me about flight schedules... (e.g., 'busiest', 'predict delay for AI2509', 'best flights on BOM->DEL')")
    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.spinner("Analyzing..."):
            response = enhanced_chatbot_handler(prompt, df, st.session_state.runway_cfg)
        st.session_state.chat.append({"role": "assistant", "content": response})
        st.rerun()

    # --- Clear chat ---
    if st.button("Clear Chat", type="secondary"):
        st.session_state.chat = []
        st.rerun()

    # --- Example queries ---
    with st.expander("💡 Example Queries"):
        st.code(
            '# Traffic Analysis\n'
            'busiest\n'
            'best\n'
            'cascade\n\n'
            '# Specific Flight Ops\n'
            'shift AI2509 by 15 min\n'
            'predict delay for AI2509\n\n'
            '# Route-ranking\n'
            'best flights on BOM->DEL top 5\n'
            'worst flights from DEL to BLR\n\n'
            '# Configuration\n'
            'set runway mode to segregated\n'
            'set weather to rain\n'
        )
