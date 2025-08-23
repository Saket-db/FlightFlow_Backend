# src/features.py
import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Ensure dep_delayed_15 exists (DepartureDelayMin > 15)
    if "dep_delayed_15" not in out.columns and "DepartureDelayMin" in out.columns:
        out["dep_delayed_15"] = (out["DepartureDelayMin"] > 15).astype(int)

    # Ensure hour/weekday if needed from STD
    if "STD" in out.columns and "DayOfWeek" not in out.columns:
        out["DayOfWeek"] = out["STD"].dt.weekday
    if "STD" in out.columns and "hour" not in out.columns:
        out["hour"] = out["STD"].dt.hour
    elif "STD_MinOfDay" in out.columns and "hour" not in out.columns:
        out["hour"] = (out["STD_MinOfDay"] // 60).astype("Int64")

    # Peak flag
    if "hour" in out.columns:
        out["is_peak"] = out["hour"].between(6,9) | out["hour"].between(17,22)
    elif "TimeSlot" in out.columns:
        out["is_peak"] = out["TimeSlot"].astype(str).str.contains("6AM|9AM|5PM|6PM|7PM|8PM|9PM|10PM", case=False, regex=True)

    # Safety fills
    for c in ["slot_load","SchedBlockMin"]:
        if c in out.columns:
            out[c] = out[c].fillna(0)

    return out
