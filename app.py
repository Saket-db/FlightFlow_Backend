import streamlit as st
import pandas as pd
import plotly.express as px

from src.ingest import load_flights
from src.features import build_features
from src.analysis import slot_stats, top_routes, top_airlines, green_windows
from src.model import (
    train_delay_classifier,
    load_model,
    train_delay_quantiles,
    load_delay_quantiles,
    CLASSIFIER_OUT_PATH,
)
from src.whatif import shift_by_minutes, queueing_burden
from src.queueing import RunwayConfig
from src.cascade import cascade_risk
from src.nlp import parse_intent

# === Delay Lab helpers ===
# === Delay Lab helpers ===
def ensure_slotload(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee slot_load exists as flights per 15-min bucket."""
    if "slot_load" in df.columns and not df["slot_load"].isna().all():
        return df
    x = df.copy()
    bucket = (pd.to_numeric(x.get("STD_MinOfDay"), errors="coerce").fillna(-1) // 15) * 15
    x["slot_15_bucket"] = bucket
    x["slot_load"] = x.groupby("slot_15_bucket")["Flight Number"].transform("count")
    return x

def airline_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Per-airline summary used in multiple charts."""
    z = df.copy()
    if "airline" not in z.columns:
        z["airline"] = z["Flight Number"].astype(str).str.extract(r"^([A-Z]+)")
    out = (
        z.groupby("airline", dropna=True)
         .agg(
             flights=("Flight Number", "count"),
             avg_dep_delay=("DepartureDelayMin", "mean"),
             p90_dep_delay=("DepartureDelayMin", lambda s: pd.to_numeric(s, errors="coerce").quantile(0.90))
         )
         .reset_index()
         .sort_values("avg_dep_delay", ascending=False)
    )
    return out

# REPLACE your existing pick_route() with this version
def pick_route(df: pd.DataFrame, key_prefix: str = "") -> tuple[str|None, str|None]:
    """Return a selected (From, To) pair with unique widget keys via key_prefix."""
    origins = sorted(df["From"].dropna().astype(str).str.upper().unique().tolist())
    dests   = sorted(df["To"].dropna().astype(str).str.upper().unique().tolist())
    col_a, col_b = st.columns(2)
    with col_a:
        o = st.selectbox(
            "From (route filter)", origins,
            key=f"{key_prefix}from_pick"
        ) if origins else None
    with col_b:
        d = st.selectbox(
            "To (route filter)", dests,
            key=f"{key_prefix}to_pick"
        ) if dests else None
    return o, d


def safe_models_in_session():
    """Return (p50_model, p90_model) if available either in session or disk, else (None, None)."""
    p50 = st.session_state.get("p50_model")
    p90 = st.session_state.get("p90_model")
    if p50 is not None and p90 is not None:
        return p50, p90
    try:
        from src.model import load_delay_quantiles
        p50, p90 = load_delay_quantiles()
        st.session_state.p50_model = p50
        st.session_state.p90_model = p90
        return p50, p90
    except Exception:
        return None, None

def features_for_rowblock(df_block: pd.DataFrame) -> pd.DataFrame:
    """Return the exact feature frame required by the quantile models for a given block of rows."""
    required = ["TimeSlot","From","To","Aircraft","airline","STD_MinOfDay","DayOfWeek","IsWeekend","SchedBlockMin","slot_load"]
    x = df_block.copy()
    # derive airline if missing
    if "airline" not in x.columns and "Flight Number" in x.columns:
        x["airline"] = x["Flight Number"].astype(str).str.extract(r"^([A-Z]+)")
    # ensure slot_load
    if "slot_load" not in x.columns or x["slot_load"].isna().all():
        x = ensure_slotload(x)
    missing = [c for c in required if c not in x.columns]
    if missing:
        return pd.DataFrame(columns=required)
    return x[required]




def enhanced_chatbot_handler(q: str, df: pd.DataFrame, cfg):
    """
    Robust chatbot handler with route-ranking, predictions, what-if,
    busiest/best windows, cascade risk, and runway config.
    Returns a markdown-friendly string for Streamlit chat (no emojis).
    """
    from src.nlp import parse_intent
    from src.analysis import slot_stats, green_windows
    from src.cascade import cascade_risk
    from src.whatif import shift_by_minutes, queueing_burden
    from src.model import load_delay_quantiles
    import numpy as np

    def df_to_markdown(d: pd.DataFrame) -> str:
        # Safe markdown table generator (no dependency on tabulate)
        cols = [str(c) for c in d.columns]
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = ["| " + " | ".join("" if (isinstance(x, float) and np.isnan(x)) else str(x) for x in row) + " |"
                for row in d.itertuples(index=False, name=None)]
        return "\n".join([header, sep] + rows)

    try:
        intent = parse_intent(q)

        # --- runway config ---
        if intent["intent"] == "set_mode":
            cfg.mode = intent["mode"]
            return f"Runway mode set to **{cfg.mode}**. Effective μ={cfg.mu():.2f}/min"

        if intent["intent"] == "set_weather":
            cfg.weather = intent["weather"]
            return f"Weather set to **{cfg.weather}**. Effective μ={cfg.mu():.2f}/min"

        # --- analytics: busiest ---
        if intent["intent"] == "busiest":
            g = slot_stats(df).sort_values("flights", ascending=False).head(10)
            if g.empty:
                return "No flight data available for analysis."
            show = g[["slot_label","flights","avg_dep_delay","p50_dep_delay","p90_dep_delay"]].rename(
                columns={"slot_label":"Slot","flights":"Flights",
                         "avg_dep_delay":"Avg Delay (min)",
                         "p50_dep_delay":"P50 Delay (min)",
                         "p90_dep_delay":"P90 Delay (min)"}
            )
            return df_to_markdown(show)

        # --- analytics: best windows ---
        if intent["intent"] == "best":
            g = green_windows(df, n=10)
            if g.empty:
                return "No green windows found."
            show = g[["slot_label","flights","p90_dep_delay","p50_dep_delay","avg_dep_delay"]].rename(
                columns={"slot_label":"Slot","flights":"Flights",
                         "p90_dep_delay":"P90 Delay (min)",
                         "p50_dep_delay":"P50 Delay (min)",
                         "avg_dep_delay":"Avg Delay (min)"}
            )
            return df_to_markdown(show)

        # --- cascade risk ---
        if intent["intent"] == "cascade":
            top = cascade_risk(df, top_n=10)
            if top.empty:
                return "No cascade risk data available."
            show = top.rename(columns={
                "Flight Number":"Flight",
                "SchedBlockMin":"Sched Block (min)",
                "slot_load":"Slot Load",
                "slot_p90":"Slot P90 (min)",
                "cascade_score":"Risk"
            })
            keep = ["Flight","From","To","Sched Block (min)","Slot Load","Slot P90 (min)","Risk"]
            return df_to_markdown(show[keep])

        # --- what-if: shift-by ---
        if intent["intent"] == "shift_by":
            flight = intent["flight"]
            mins = intent["mins"]
            if flight not in df["Flight Number"].astype(str).values:
                return f"Flight **{flight}** not found in schedule."

            before = df.copy()
            after = shift_by_minutes(df, flight, mins)
            qb_before = queueing_burden(before, cfg)
            qb_after  = queueing_burden(after,  cfg)
            delta = qb_after - qb_before
            impact = "Improved" if delta < 0 else "Increased" if delta > 0 else "No change"
            return (f"{impact} queueing burden by **{delta:.1f} min** for shifting **{flight}** "
                    f"by **{mins} min** (mode={cfg.mode}, weather={cfg.weather}).")

        # --- predictions: P50/P90 delay ---
        if intent["intent"] == "predict":
            flight = intent["flight"]
            row_df = df[df["Flight Number"].astype(str) == flight]
            if row_df.empty:
                return f"Flight **{flight}** not found."

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
                return f"Missing columns for prediction: {missing}"

            try:
                p50, p90 = load_delay_quantiles()
            except Exception:
                return "Quantile models not found. Train them in the Model tab first."

            X = row_df[required].iloc[[0]]
            d50 = float(p50.predict(X)[0]); d90 = float(p90.predict(X)[0])
            md = pd.DataFrame(
                [{"Flight": flight, "P50 (min)": f"{d50:.1f}", "P90 (min)": f"{d90:.1f}"}]
            )
            return df_to_markdown(md)

        # --- best / worst flights on a route ---
        if intent["intent"] == "route_rank":
            kind  = intent["kind"]      # 'best' | 'worst'
            o, d  = intent["origin"], intent["dest"]
            top_n = intent["top_n"]

            if "route" in df.columns:
                route_key = f"{o}->{d}"
                sub = df[df["route"].astype(str).str.upper() == route_key]
            else:
                sub = df[(df["From"].astype(str).str.upper() == o) &
                         (df["To"].astype(str).str.upper() == d)]

            if sub.empty:
                return f"No flights found on route **{o}->{d}**."

            agg = (sub.groupby(["Flight Number","From","To"])
                     .agg(flights=("Flight Number","count"),
                          avg_dep_delay=("DepartureDelayMin","mean"))
                     .reset_index())
            agg["avg_dep_delay"] = pd.to_numeric(agg["avg_dep_delay"], errors="coerce")
            agg = agg.dropna(subset=["avg_dep_delay"])

            if agg.empty:
                return f"No delay data for route **{o}->{d}**."

            ascending = True if kind == "best" else False
            agg = agg.sort_values("avg_dep_delay", ascending=ascending).head(top_n)

            show = agg.rename(columns={
                "Flight Number":"Flight", "avg_dep_delay":"Avg Dep Delay (min)", "flights":"Obs"
            })
            return df_to_markdown(show[["Flight","From","To","Avg Dep Delay (min)","Obs"]])

        # --- fallback help ---
        return ("I can help with:\n"
                "- busiest — peak times\n"
                "- best — optimal windows\n"
                "- cascade — high-impact flights\n"
                "- shift AI2509 by 10 min — simulate queueing impact\n"
                "- predict delay for AI2509 — P50/P90 forecast\n"
                "- best flights on BOM->DEL top 5, worst flights from DEL to BLR\n"
                "- set runway mode to segregated / set weather to rain")

    except Exception as e:
        return f"Chatbot error: {str(e)}"


st.set_page_config(page_title="Airport Scheduling Optimizer", layout="wide")
st.title("Airport Scheduling Optimizer")

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

tab1, tab2, tab4, tab5 = st.tabs(["Overview", "Delay Lab", "Model", "Chatbot"])

# -------- Overview --------
with tab1:
    st.subheader("Dataset Debug Info")
    st.write("Unique From:", df["From"].astype(str).str.strip().str.upper().unique()[:10])
    st.write("Unique To:", df["To"].astype(str).str.strip().str.upper().unique()[:10])
    st.write("From counts:", df["From"].astype(str).str.strip().str.upper().value_counts().head(5))
    st.write("To counts:", df["To"].astype(str).str.strip().str.upper().value_counts().head(5))

    try:
        st.subheader("TimeSlot coverage")
        if "TimeSlot" in df.columns:
            st.bar_chart(df["TimeSlot"].astype(str).value_counts())
        else:
            st.info("No TimeSlot column found; showing first 20 rows below.")

        st.subheader("Quick preview by TimeSlot")
        if "TimeSlot" in df.columns:
            slots = sorted(set(df["TimeSlot"].dropna().astype(str).tolist()))
            slot_pick = st.selectbox("Select window", slots, key="ov_timeslot_pick")
            keep_cols = [c for c in ["Flight Number","From","To","STD","ATD","DepartureDelayMin","TimeSlot"] if c in df.columns]
            st.dataframe(df[df["TimeSlot"].astype(str)==slot_pick][keep_cols].head(20), use_container_width=True)
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
# -------- Delay Lab --------
# -------- Delay Lab --------
with tab2:
    try:
        st.subheader("Delay Lab")
        df_dl = ensure_slotload(df)

        sub1, sub2, sub3, sub4 = st.tabs([
            "By Slot",
            "Airlines",
            "Routes & Flights",
            "Predictions & What-If"
        ])

        # ---------- By Slot ----------
        with sub1:
            st.markdown("**P90 and volume by 15-minute slot**")
            g = slot_stats(df_dl)
            x_col = "slot_label" if "slot_label" in g.columns else g.columns[0]

            fig = px.line(g, x=x_col, y="p90_dep_delay", markers=True,
                          title="P90 Departure Delay by Slot")
            st.plotly_chart(fig, use_container_width=True, key="dl_slot_p90_line")

            fig2 = px.bar(
                g.sort_values("flights", ascending=False).head(20),
                x=x_col, y="flights", title="Busiest Slots (Top 20)"
            )
            st.plotly_chart(fig2, use_container_width=True, key="dl_slot_busiest_bar")

            st.markdown("**Best (Green) Windows**")
            st.dataframe(green_windows(df_dl, n=20), use_container_width=True)

        # ---------- Airlines ----------
        with sub2:
            a = airline_agg(df_dl)
            if a.empty:
                st.info("No airline data available.")
            else:
                col_a1, col_a2 = st.columns(2)
                with col_a1:
                    fig_a1 = px.bar(a.head(15), x="airline", y="avg_dep_delay",
                                    title="Average Departure Delay by Airline (Top 15)")
                    st.plotly_chart(fig_a1, use_container_width=True, key="dl_airline_avg_bar")
                with col_a2:
                    fig_a2 = px.bar(a.head(15), x="airline", y="p90_dep_delay",
                                    title="P90 Departure Delay by Airline (Top 15)")
                    st.plotly_chart(fig_a2, use_container_width=True, key="dl_airline_p90_bar")

                st.markdown("**Delay Distribution by Airline**")
                samp = df_dl.sample(min(len(df_dl), 4000), random_state=42) if len(df_dl) > 4000 else df_dl
                fig_box = px.box(
                    samp.dropna(subset=["DepartureDelayMin"]),
                    x="airline", y="DepartureDelayMin",
                    points="suspectedoutliers",
                    title="Departure Delay Distribution (Boxplot)"
                )
                st.plotly_chart(fig_box, use_container_width=True, key="dl_airline_boxplot")

                if "STD_MinOfDay" in df_dl.columns:
                    tmp = df_dl.copy()
                    tmp["hour"] = (pd.to_numeric(tmp["STD_MinOfDay"], errors="coerce") // 60).astype("Int64")
                    pivot = (tmp.dropna(subset=["hour"])
                               .groupby(["airline","hour"])["DepartureDelayMin"]
                               .mean().reset_index())
                    fig_hm = px.density_heatmap(
                        pivot, x="hour", y="airline", z="DepartureDelayMin",
                        histfunc="avg", nbinsx=24, title="Average Departure Delay Heatmap"
                    )
                    st.plotly_chart(fig_hm, use_container_width=True, key="dl_airline_heatmap")

        # ---------- Routes & Flights ----------
        with sub3:
            r_o, r_d = pick_route(df_dl, key_prefix="routes_")
            if r_o and r_d:
                sub_route = df_dl[
                    (df_dl["From"].astype(str).str.upper() == r_o) &
                    (df_dl["To"].astype(str).str.upper() == r_d)
                ]
                if sub_route.empty:
                    st.warning("No rows for the selected route.")
                else:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Flights", int(sub_route["Flight Number"].count()))
                    with c2:
                        st.metric("Avg dep delay (min)",
                                  f"{pd.to_numeric(sub_route['DepartureDelayMin'], errors='coerce').mean():.2f}")
                    with c3:
                        st.metric("P90 dep delay (min)",
                                  f"{pd.to_numeric(sub_route['DepartureDelayMin'], errors='coerce').quantile(0.90):.2f}")

                    tbl = (sub_route.groupby("Flight Number")
                                     .agg(flights=("Flight Number","count"),
                                          avg_dep_delay=("DepartureDelayMin","mean"),
                                          p90=("DepartureDelayMin",
                                               lambda s: pd.to_numeric(s, errors="coerce").quantile(0.90)))
                                     .reset_index()
                                     .sort_values(["avg_dep_delay","flights"], ascending=[False, False]))
                    st.markdown("**Flights on selected route**")
                    st.dataframe(tbl.rename(columns={"avg_dep_delay":"Avg Dep Delay (min)",
                                                     "p90":"P90 (min)"}),
                                 use_container_width=True)

                    ar = (sub_route.groupby("airline")
                                    .agg(avg_dep_delay=("DepartureDelayMin","mean"),
                                         flights=("Flight Number","count"))
                                    .reset_index()
                                    .sort_values("avg_dep_delay", ascending=False))
                    fig_rb = px.bar(ar, x="airline", y="avg_dep_delay", hover_data=["flights"],
                                    title=f"Avg Departure Delay by Airline on {r_o}->{r_d}")
                    st.plotly_chart(fig_rb, use_container_width=True, key="dl_route_airline_bar")
            else:
                st.info("Pick both origin and destination to view route analytics.")

        # ---------- Predictions & What-If ----------
        with sub4:
            st.markdown("**Model-based projections and quick what-if**")

            r_o2, r_d2 = pick_route(df_dl, key_prefix="pred_")
            p50_model, p90_model = safe_models_in_session()
            if r_o2 and r_d2:
                sub2 = df_dl[
                    (df_dl["From"].astype(str).str.upper() == r_o2) &
                    (df_dl["To"].astype(str).str.upper() == r_d2)
                ]
                if sub2.empty:
                    st.warning("No rows for the selected route.")
                else:
                    if p50_model is None or p90_model is None:
                        st.info("Quantile models not found. Train them in the Model tab to enable predictions.")
                    else:
                        Xr = features_for_rowblock(sub2)
                        if Xr.empty:
                            st.warning("Required features missing for prediction on this route.")
                        else:
                            pred50 = p50_model.predict(Xr)
                            pred90 = p90_model.predict(Xr)
                            attach = sub2.copy()
                            attach["pred_p50"] = pred50
                            attach["pred_p90"] = pred90

                            flight_scores = (attach.groupby("Flight Number")
                                                    .agg(obs=("Flight Number","count"),
                                                         p50=("pred_p50","mean"),
                                                         p90=("pred_p90","mean"))
                                                    .reset_index()
                                                    .sort_values(["p90","p50"], ascending=[False, False]))
                            st.markdown("**Predicted delay scores by flight (on selected route)**")
                            st.dataframe(flight_scores.rename(columns={"obs":"Obs",
                                                                       "p50":"P50 (min)",
                                                                       "p90":"P90 (min)"}),
                                         use_container_width=True)

                            airline_scores = (attach.groupby("airline")
                                                      .agg(obs=("Flight Number","count"),
                                                           p50=("pred_p50","mean"),
                                                           p90=("pred_p90","mean"))
                                                      .reset_index()
                                                      .sort_values(["p90","p50"], ascending=[False, False]))
                            st.markdown("**Predicted delay scores by airline (on selected route)**")
                            st.dataframe(airline_scores.rename(columns={"obs":"Obs",
                                                                        "p50":"P50 (min)",
                                                                        "p90":"P90 (min)"}),
                                         use_container_width=True)

            st.divider()

            st.markdown("**Quick What-If: shift one flight in time**")
            flights_all = df_dl["Flight Number"].dropna().astype(str).unique().tolist()
            cwa, cwb = st.columns([2,1])
            with cwa:
                f_pick = st.selectbox("Pick flight", flights_all, key="dl_pred_flight_pick") if flights_all else None
            with cwb:
                delta = st.number_input("Shift minutes (− earlier / + later)", value=5, step=5,
                                        key="dl_pred_shift_delta")
            if st.button("Simulate shift", key="dl_pred_shift_btn") and f_pick:
                before = df_dl.copy()
                after  = shift_by_minutes(df_dl, f_pick, int(delta))
                qb_before = queueing_burden(before, st.session_state.runway_cfg)
                qb_after  = queueing_burden(after,  st.session_state.runway_cfg)
                st.metric("Queueing burden before (min)", round(qb_before, 2))
                st.metric("Queueing burden after (min)",  round(qb_after, 2))
                st.metric("Δ (after − before)", round(qb_after - qb_before, 2))

            st.divider()

            st.markdown("**Simple future projection by weekday**")
            dow = st.selectbox("Assume DayOfWeek for projection (0=Mon … 6=Sun)",
                               list(range(7)), index=0, key="dl_future_dow_tabs")
            p50_model, p90_model = safe_models_in_session()
            if p50_model is None or p90_model is None:
                st.info("Train quantile models in the Model tab to enable the projection.")
            else:
                df_future = df_dl.copy()
                df_future["DayOfWeek"] = dow
                df_future["IsWeekend"] = df_future["DayOfWeek"].isin([5,6]).astype("Int64")
                Xf = features_for_rowblock(df_future)
                if Xf.empty:
                    st.warning("Required features missing for projection.")
                else:
                    pred50 = p50_model.predict(Xf)
                    pred90 = p90_model.predict(Xf)
                    attach = df_future.copy()
                    attach["pred_p50"] = pred50
                    attach["pred_p90"] = pred90
                    if "slot_15" in attach.columns and attach["slot_15"].notna().any():
                        grp_key = "slot_15"
                    else:
                        attach["slot_15_bucket"] = (pd.to_numeric(attach.get("STD_MinOfDay"),
                                                                  errors="coerce").fillna(-1) // 15) * 15
                        grp_key = "slot_15_bucket"
                    proj = (attach.groupby(grp_key)
                                  .agg(pred_p50=("pred_p50","mean"),
                                       pred_p90=("pred_p90","mean"))
                                  .reset_index())
                    if grp_key == "slot_15":
                        proj["slot_label"] = proj["slot_15"].astype(str)
                    else:
                        mins = pd.to_numeric(proj[grp_key], errors="coerce").fillna(0).astype(int)
                        proj["slot_label"] = (mins//60).astype(str).str.zfill(2) + ":" + (mins%60).astype(str).str.zfill(2)
                    fig_proj = px.line(proj.sort_values(grp_key), x="slot_label",
                                       y=["pred_p50","pred_p90"],
                                       title=f"Projected P50/P90 by Slot for DayOfWeek={dow}")
                    st.plotly_chart(fig_proj, use_container_width=True, key="dl_future_projection_line")

    except Exception as e:
        st.error(f"Delay Lab error: {e}")

# -------- What-If Studio --------
#
    try:
        st.subheader("Shift a Flight by ± minutes (Minute-of-Day)")
        flights = df["Flight Number"].dropna().astype(str).unique().tolist()
        if not flights:
            st.info("No flights found.")
        else:
            fsel  = st.selectbox("Choose Flight Number", flights, key="sel_shift_flight")
            delta = st.number_input("Shift minutes (negative to move earlier)", value=5, step=5, key="num_shift_minutes")
            if st.button("Simulate Shift", key="btn_simulate_shift"):
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

        if "clf_model" not in st.session_state: st.session_state.clf_model = None
        if "p50_model" not in st.session_state: st.session_state.p50_model = None
        if "p90_model" not in st.session_state: st.session_state.p90_model = None

        with colA:
            if st.button("Train classifier", key="btn_train_classifier"):
                with st.spinner("Training classifier..."):
                    clf, metrics = train_delay_classifier(df)
                    st.session_state.clf_model = clf
                m1, m2 = st.columns(2)
                with m1: st.metric("AUC", f"{metrics['auc']:.3f}")
                with m2: st.metric("F1", f"{metrics['f1']:.3f}")
                st.write("Class balance (target counts):", metrics["classes"])
                st.write("Features used:", ", ".join(metrics["features_used"]))
                st.write("Classification report:")
                st.dataframe(pd.DataFrame(metrics["report"]).T, use_container_width=True)

        with colB:
            if st.button("Train quantiles (P50/P90)", key="btn_train_quantiles"):
                with st.spinner("Training quantile regressors..."):
                    p50, p90, qmetrics = train_delay_quantiles(df)
                    st.session_state.p50_model = p50
                    st.session_state.p90_model = p90
                q1, q2 = st.columns(2)
                with q1: st.metric("MAE P50 (min)", f"{qmetrics['mae_p50']:.2f}")
                with q2: st.metric("MAE P90 (min)", f"{qmetrics['mae_p90']:.2f}")
                st.write("Features used:", ", ".join(qmetrics["features_used"]))
                st.caption(f"Train size: {qmetrics['n_train']}, Test size: {qmetrics['n_test']}")

        with colC:
            if st.button("Load classifier", key="btn_load_classifier"):
                import os, time
                try:
                    clf = load_model()
                except FileNotFoundError as e:
                    st.error(f"{e}. Train the classifier first, or place a file at: {os.path.abspath(CLASSIFIER_OUT_PATH)}")
                except Exception as e:
                    st.error(f"Failed to load classifier: {str(e)}")
                else:
                    st.session_state.clf_model = clf
                    st.success(f"Classifier loaded from {os.path.abspath(CLASSIFIER_OUT_PATH)}")
                    try:
                        stat = os.stat(CLASSIFIER_OUT_PATH)
                        st.caption(f"File size: {stat.st_size/1e6:.2f} MB • Modified: {time.ctime(stat.st_mtime)}")
                    except Exception:
                        pass

        st.caption("Classifier → delayed >15min. Quantiles → P50/P90 delay. No ATD/ATA leakage used.")

        st.divider()
        st.subheader("Quick test (optional)")

        if st.session_state.clf_model is not None:
            flights = df["Flight Number"].dropna().astype(str).unique().tolist()
            pick = st.selectbox("Pick a flight to score (classifier)", flights, key="sel_clf_flight") if flights else None
            if pick:
                row = df[df["Flight Number"].astype(str) == pick]
                required = ["TimeSlot","From","To","Aircraft","airline","STD_MinOfDay","DayOfWeek","IsWeekend","SchedBlockMin","slot_load"]
                missing = [c for c in required if c not in row.columns]
                if missing:
                    st.warning(f"Missing columns for scoring: {missing}")
                else:
                    proba = float(st.session_state.clf_model.predict_proba(row[required].iloc[[0]])[:,1])
                    st.metric("P(delay > 15 min)", f"{proba:.3f}")

        if st.session_state.p50_model is not None and st.session_state.p90_model is not None:
            flights_q = df["Flight Number"].dropna().astype(str).unique().tolist()
            pick_q = st.selectbox("Pick a flight to predict (quantiles)", flights_q, key="sel_quant_flight") if flights_q else None
            if pick_q:
                row = df[df["Flight Number"].astype(str) == pick_q]
                required = ["TimeSlot","From","To","Aircraft","airline","STD_MinOfDay","DayOfWeek","IsWeekend","SchedBlockMin","slot_load"]
                missing = [c for c in required if c not in row.columns]
                if missing:
                    st.warning(f"Missing columns for quantile prediction: {missing}")
                else:
                    d50 = float(st.session_state.p50_model.predict(row[required].iloc[[0]])[0])
                    d90 = float(st.session_state.p90_model.predict(row[required].iloc[[0]])[0])
                    p_df = pd.DataFrame([{"Flight": pick_q, "P50 (min)": f"{d50:.1f}", "P90 (min)": f"{d90:.1f}"}])
                    st.dataframe(p_df, use_container_width=True)

    except Exception as e:
        st.error(f"Model tab error: {e}")

# -------- Chatbot --------
with tab5:
    st.subheader("AI Ops Copilot")
    st.caption("Natural language interface for flight operations analysis")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Busiest", key="btn_chat_busiest"):
            st.session_state.chat.append({"role": "user", "content": "busiest"})
            response = enhanced_chatbot_handler("busiest", df, st.session_state.runway_cfg)
            st.session_state.chat.append({"role": "assistant", "content": response})
            st.rerun()

    with col2:
        if st.button("Best Slots", key="btn_chat_best"):
            st.session_state.chat.append({"role": "user", "content": "best"})
            response = enhanced_chatbot_handler("best", df, st.session_state.runway_cfg)
            st.session_state.chat.append({"role": "assistant", "content": response})
            st.rerun()

    with col3:
        if st.button("Cascade Risk", key="btn_chat_cascade"):
            st.session_state.chat.append({"role": "user", "content": "cascade"})
            response = enhanced_chatbot_handler("cascade", df, st.session_state.runway_cfg)
            st.session_state.chat.append({"role": "assistant", "content": response})
            st.rerun()

    with col4:
        cfg = st.session_state.runway_cfg
        try:
            st.metric("Current Config", f"{cfg.mode.title()}, {cfg.weather}", f"μ={cfg.mu():.2f}/min")
        except AttributeError:
            st.error("RunwayConfig.mu() not found. Please update src/queueing.py with the latest version.")

    st.divider()

    if {"From", "To"} <= set(df.columns):
        st.markdown("**Route Ranking (Quick Tool)**")
        rcol1, rcol2, rcol3, rcol4 = st.columns([1,1,1,1])
        with rcol1:
            origins = sorted(df["From"].dropna().astype(str).str.upper().unique().tolist())
            origin_sel = st.selectbox("From", origins, index=0, key="sel_route_from")
        with rcol2:
            dests = sorted(df["To"].dropna().astype(str).str.upper().unique().tolist())
            dest_sel = st.selectbox("To", dests, index=min(1, len(dests)-1), key="sel_route_to")
        with rcol3:
            top_n = st.number_input("Top N", min_value=1, max_value=50, value=10, step=1, key="num_route_topn")
        with rcol4:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Best on Route", key="btn_best_route"):
                    prompt = f"best flights on {origin_sel}->{dest_sel} top {int(top_n)}"
                    st.session_state.chat.append({"role": "user", "content": prompt})
                    resp = enhanced_chatbot_handler(prompt, df, st.session_state.runway_cfg)
                    st.session_state.chat.append({"role": "assistant", "content": resp})
                    st.rerun()
            with c2:
                if st.button("Worst on Route", key="btn_worst_route"):
                    prompt = f"worst flights on {origin_sel}->{dest_sel} top {int(top_n)}"
                    st.session_state.chat.append({"role": "user", "content": prompt})
                    resp = enhanced_chatbot_handler(prompt, df, st.session_state.runway_cfg)
                    st.session_state.chat.append({"role": "assistant", "content": resp})
                    st.rerun()

    st.divider()

    # Chat history (render markdown so tables display nicely)
    if not st.session_state.chat:
        st.info("Hello! Ask about 'busiest', 'best', or 'best flights on BOM->DEL'.")
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    prompt = st.chat_input(
        "Ask me about flight schedules... (e.g., 'busiest', 'predict delay for AI2509', 'best flights on BOM->DEL')",
        key="chat_input_main"
    )
    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.spinner("Analyzing..."):
            response = enhanced_chatbot_handler(prompt, df, st.session_state.runway_cfg)
        st.session_state.chat.append({"role": "assistant", "content": response})
        st.rerun()


        # Example queries
        with st.expander("Example Queries"):
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
