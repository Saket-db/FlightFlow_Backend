import pandas as pd
import numpy as np
import re

CSV_PATH = "data/processed/combined_clean_ml.csv"

def load_flights(path: str = CSV_PATH) -> pd.DataFrame:
    """
    Load the pre-cleaned combined CSV and add a few helper columns used downstream.
    Expected columns:
    S.No,Flight Number,Date,From,To,Aircraft,Flight time,STD,ATD,STA,ATA,
    TimeSlot,STD_MinOfDay,STA_MinOfDay,DepartureDelayMin,ArrivalDelayMin,
    SchedBlockMin,ActualBlockMin,Year,Month,Day,DayOfWeek,IsWeekend,Unnamed: 2
    """
    df = pd.read_csv(path)

    # Drop junk column if present
    if "Unnamed: 2" in df.columns:
        df = df.drop(columns=["Unnamed: 2"])

    # Normalize column names (strip spaces)
    df.columns = [c.strip() for c in df.columns]

    # Parse datetimes (coerce broken ones to NaT)
    for col in ["STD", "ATD", "STA", "ATA", "Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Ensure numeric types
    num_cols = [
        "STD_MinOfDay","STA_MinOfDay","DepartureDelayMin","ArrivalDelayMin",
        "SchedBlockMin","ActualBlockMin","Year","Month","Day","DayOfWeek","IsWeekend"
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Airline prefix from Flight Number (e.g., AI2509 -> AI)
    if "Flight Number" in df.columns:
        df["airline"] = df["Flight Number"].astype(str).str.extract(r"^([A-Z]+)")

    # Route string
    if {"From","To"} <= set(df.columns):
        df["route"] = df["From"].astype(str).str.replace("\xa0"," ", regex=False) + "->" + \
                      df["To"].astype(str).str.replace("\xa0"," ", regex=False)

    # 15-min slot from STD (fallback to STD_MinOfDay)
    if "STD" in df.columns and df["STD"].notna().any():
        df["slot_15"] = df["STD"].dt.floor("15min")
    else:
        # fallback using minute-of-day: floor to 15-minute buckets
        if "STD_MinOfDay" in df.columns:
            df["slot_15_min"] = (df["STD_MinOfDay"] // 15) * 15
            # if Date known, make a timestamp; else keep numeric bucket
            if "Date" in df.columns and df["Date"].notna().any():
                df["slot_15"] = df.apply(
                    lambda r: pd.Timestamp(r["Date"].date()) + pd.Timedelta(minutes=r["slot_15_min"])
                    if pd.notna(r.get("Date")) and pd.notna(r.get("slot_15_min")) else pd.NaT,
                    axis=1
                )
            else:
                df["slot_15"] = df["slot_15_min"]

    # Slot load (congestion proxy)
    if "slot_15" in df.columns:
        df["slot_load"] = df.groupby("slot_15")["Flight Number"].transform("count")
    else:
        df["slot_load"] = np.nan

    # Binary target: delayed > 15 min (departure)
    if "DepartureDelayMin" in df.columns:
        df["dep_delayed_15"] = (df["DepartureDelayMin"] > 15).astype(int)

    return df

if __name__ == "__main__":
    d = load_flights()
    print("✅ Loaded:", d.shape, "cols:", list(d.columns))
    print("✅ Loaded:", d.shape, "cols:", list(d.columns))
