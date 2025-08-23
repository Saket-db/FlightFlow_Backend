# path: src/features.py
import pandas as pd
import numpy as np
import re

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Idempotent feature builder aligned with our pipeline.
    - Ensures binary target dep_delayed_15
    - Derives hour / weekday when possible
    - Adds peak flag
    - Guarantees slot buckets and slot_load (15-min congestion proxy)
    - Ensures airline and route helper columns exist
    - Performs safe numeric coercions and NA fills for robustness
    """
    out = df.copy()

    # --------- Basic helpers: airline / route ----------
    if "airline" not in out.columns and "Flight Number" in out.columns:
        out["airline"] = out["Flight Number"].astype(str).str.extract(r"^([A-Z]+)")

    if "route" not in out.columns and {"From", "To"} <= set(out.columns):
        out["route"] = out["From"].astype(str).str.replace("\xa0", " ", regex=False) + "->" + \
                       out["To"].astype(str).str.replace("\xa0", " ", regex=False)

    # --------- Coerce key numerics safely ----------
    to_numeric_cols = ["STD_MinOfDay", "DayOfWeek", "IsWeekend", "SchedBlockMin",
                       "slot_load", "DepartureDelayMin"]
    for c in to_numeric_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # --------- Ensure target: dep_delayed_15 ----------
    if "dep_delayed_15" not in out.columns:
        if "DepartureDelayMin" in out.columns:
            out["dep_delayed_15"] = (out["DepartureDelayMin"] > 15).astype("Int64")
        else:
            # create placeholder target if totally missing; model will error if used for training
            out["dep_delayed_15"] = pd.Series([pd.NA] * len(out), dtype="Int64")

    # --------- Hour / Weekday from STD if available ----------
    if "STD" in out.columns and pd.api.types.is_datetime64_any_dtype(out["STD"]):
        if "DayOfWeek" not in out.columns or out["DayOfWeek"].isna().all():
            out["DayOfWeek"] = out["STD"].dt.weekday
        if "hour" not in out.columns or out["hour"].isna().all():
            out["hour"] = out["STD"].dt.hour
    else:
        # fallback from minutes-of-day
        if "hour" not in out.columns and "STD_MinOfDay" in out.columns:
            out["hour"] = (out["STD_MinOfDay"] // 60).astype("Int64")

    # --------- Peak flag ----------
    if "hour" in out.columns:
        out["is_peak"] = out["hour"].between(6, 9) | out["hour"].between(17, 22)
    elif "TimeSlot" in out.columns:
        # coarse fallback using textual slot
        out["is_peak"] = out["TimeSlot"].astype(str).str.contains(
            r"(6AM|7AM|8AM|9AM|5PM|6PM|7PM|8PM|9PM|10PM)", flags=re.I, regex=True
        )
    else:
        out["is_peak"] = False

    # --------- Slot buckets (15-min) & slot_load ----------
    # Prefer real timestamp bucket if STD exists
    if "STD" in out.columns and pd.api.types.is_datetime64_any_dtype(out["STD"]):
        if "slot_15" not in out.columns or out["slot_15"].isna().all():
            out["slot_15"] = out["STD"].dt.floor("15min")
        # slot_load from timestamp bucket
        if "slot_load" not in out.columns or out["slot_load"].isna().all():
            out["slot_load"] = out.groupby("slot_15")["Flight Number"].transform("count")
    else:
        # Fallback to numeric minute-of-day buckets
        if "STD_MinOfDay" in out.columns:
            out["slot_15_bucket"] = (out["STD_MinOfDay"].fillna(-1) // 15) * 15
            if "slot_load" not in out.columns or out["slot_load"].isna().all():
                out["slot_load"] = out.groupby("slot_15_bucket")["Flight Number"].transform("count")

    # --------- Safety fills ----------
    if "SchedBlockMin" in out.columns:
        out["SchedBlockMin"] = out["SchedBlockMin"].fillna(0)
    if "slot_load" in out.columns:
        out["slot_load"] = out["slot_load"].fillna(0)

    return out
