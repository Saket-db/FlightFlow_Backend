# path: src/clean.py

import pandas as pd
import numpy as np
import re

INPUT_CSV = "d:/HoneyWell Hack/Hack_Actual/data/raw/combined.csv"
# Aligned with the Streamlit app's expected path
OUTPUT_CSV = "data/processed/combined_clean_ml.csv"

# --- Helpers ---
def parse_time_str(s):
    """Parse '06:20:00', '6:20 AM', or 'Landed 8:14 AM' -> datetime.time or NaT."""
    if pd.isna(s): return pd.NaT
    s = str(s).strip()
    if not s or s == "#####": return pd.NaT
    # grab first time token (works inside 'Landed 7:26 AM')
    m = re.search(r'(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)', s, flags=re.I)
    token = m.group(1) if m else s
    for fmt in ("%I:%M %p", "%I:%M:%S %p", "%H:%M:%S", "%H:%M"):
        try:
            return pd.to_datetime(token, format=fmt).time()
        except Exception:
            continue
    try:
        return pd.to_datetime(token).time()
    except Exception:
        return pd.NaT

def combine_dt(d, t):
    if pd.isna(d) or pd.isna(t): return pd.NaT
    return pd.Timestamp.combine(d.date(), t)

def minutes_since_midnight(t):
    if pd.isna(t): return np.nan
    return t.hour * 60 + t.minute + t.second/60.0

# --- Load ---
df = pd.read_csv(INPUT_CSV, dtype=str)
df = df.dropna(how="all")

# Normalise column names & drop Unnamed
df.columns = [re.sub(r'\s+', ' ', c).strip() for c in df.columns]
df = df.loc[:, ~df.columns.str.startswith("Unnamed") | df.columns.str.contains(r"^Unnamed: 2$", regex=True)]

# Trim cells
df = df.applymap(lambda x: np.nan if (pd.isna(x) or str(x).strip() == "") else str(x).strip())

# Forward-fill IDs
for col in ["S.No", "Flight Number"]:
    if col in df.columns:
        df[col] = df[col].ffill()

# Build Date from 'Date' or 'Unnamed: 2'
date_source = None
if "Date" in df.columns and df["Date"].notna().any():
    date_source = df["Date"]
if "Unnamed: 2" in df.columns and df["Unnamed: 2"].notna().any():
    ds = df.get("Date")
    date_source = ds.fillna(df["Unnamed: 2"]) if ds is not None else df["Unnamed: 2"]

if date_source is None:
    df["Date"] = pd.NaT
else:
    df["Date"] = pd.to_datetime(date_source, errors="coerce")

# Do not drop on missing Date yet
# Parse time columns
for tcol in ["STD", "ATD", "STA", "ATA"]:
    if tcol in df.columns:
        df[tcol] = df[tcol].apply(parse_time_str)

# Fill Date within a flight block if possible
if "Date" in df.columns:
    df["Date"] = df["Date"].ffill().bfill()

# Build datetimes
for tcol in ["STD", "ATD", "STA", "ATA"]:
    if tcol in df.columns:
        df[f"{tcol}_DT"] = df.apply(lambda r: combine_dt(r["Date"], r[tcol]), axis=1)

# Keep rows that have at least one time
has_any_time = df[["STD","ATD","STA","ATA"]].notna().any(axis=1)
df = df.loc[has_any_time].copy()

# Features
df["DepartureDelayMin"] = (df["ATD_DT"] - df["STD_DT"]).dt.total_seconds()/60
df["ArrivalDelayMin"]   = (df["ATA_DT"] - df["STA_DT"]).dt.total_seconds()/60
df["SchedBlockMin"]     = (df["STA_DT"] - df["STD_DT"]).dt.total_seconds()/60
df["ActualBlockMin"]    = (df["ATA_DT"] - df["ATD_DT"]).dt.total_seconds()/60
df["STD_MinOfDay"]      = df["STD"].apply(minutes_since_midnight) if "STD" in df.columns else np.nan
df["STA_MinOfDay"]      = df["STA"].apply(minutes_since_midnight) if "STA" in df.columns else np.nan

# Calendar features
df["Year"]      = df["Date"].dt.year
df["Month"]     = df["Date"].dt.month
df["Day"]       = df["Date"].dt.day
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["IsWeekend"] = df["DayOfWeek"].isin([5,6]).astype("Int64")

# Preferred order
preferred = ["S.No","Flight Number","Date","From","To","Aircraft","Flight time",
             "STD","ATD","STA","ATA","TimeSlot",
             "STD_MinOfDay","STA_MinOfDay",
             "DepartureDelayMin","ArrivalDelayMin","SchedBlockMin","ActualBlockMin",
             "Year","Month","Day","DayOfWeek","IsWeekend"]
ordered = [c for c in preferred if c in df.columns]
df = pd.concat([df[ordered], df.drop(columns=ordered, errors="ignore")], axis=1)

# Drop helper datetime cols if you don’t want them
df = df.drop(columns=[c for c in ["STD_DT","ATD_DT","STA_DT","ATA_DT"] if c in df.columns])

df.to_csv(OUTPUT_CSV, index=False)
print(f"Wrote cleaned, ML-ready CSV -> {OUTPUT_CSV}")
