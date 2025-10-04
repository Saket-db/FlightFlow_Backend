# path: src/analysis.py
import pandas as pd
import numpy as np
from typing import Optional, Tuple

"""
Analysis helpers for:
- Slot-level stats (flights, avg delay, p50/p90) using either timestamp slots or minute buckets
- Top routes / airlines by average departure delay
- Optional range-filtered slot stats (e.g., only 06:00–12:00 by minute-of-day)
- Utility to label 15-min buckets nicely for UI

NOTE:
- Uses 'DepartureDelayMin' as the delay column.
- Works with either:
    1) df['slot_15'] as a timestamp bucket, OR
    2) df['STD_MinOfDay'] numeric minute-of-day (0..1439), which we bucket to 15-min via floor.
- Robust to partial missing data; falls back safely.
"""


DELAY_COL = "DepartureDelayMin"


def _ensure_delay_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if DELAY_COL in out.columns:
        out[DELAY_COL] = pd.to_numeric(out[DELAY_COL], errors="coerce")
    else:
        out[DELAY_COL] = np.nan
    return out


def _ensure_slot_key(df: pd.DataFrame, prefer_timestamp: bool = True) -> Tuple[pd.DataFrame, str]:
    """
    Returns (df_with_slot, slot_key_column_name)
    Priority:
      - if df['slot_15'] exists and has at least one non-null -> use it
      - else derive 'slot_15_bucket' from STD_MinOfDay // 15 * 15 (integers)
    """
    out = df.copy()
    if prefer_timestamp and ("slot_15" in out.columns) and out["slot_15"].notna().any():
        return out, "slot_15"

    # Minute-of-day fallback
    if "STD_MinOfDay" not in out.columns:
        # Create a dummy key so groupby won't crash
        out["slot_15_bucket"] = 0
        return out, "slot_15_bucket"

    out["STD_MinOfDay"] = pd.to_numeric(out["STD_MinOfDay"], errors="coerce")
    out["slot_15_bucket"] = (out["STD_MinOfDay"].fillna(-1) // 15) * 15
    return out, "slot_15_bucket"


def _slot_label(col: pd.Series) -> pd.Series:
    """
    Pretty label for slot keys:
      - If timestamps: keep as is (Streamlit/Plotly will format)
      - If integer minute buckets: convert to 'HH:MM' like '06:30'
    """
    if np.issubdtype(col.dtype, np.datetime64):
        return col
    # integer buckets in minutes
    mins = pd.to_numeric(col, errors="coerce").fillna(-1).astype(int)
    h = (mins // 60).clip(lower=0)
    m = (mins % 60).clip(lower=0)
    return (h.astype(str).str.zfill(2) + ":" + m.astype(str).str.zfill(2))


def slot_stats(df: pd.DataFrame, green_threshold: int = 25) -> pd.DataFrame:
    """
    FIXED: Configurable green threshold (default 25 instead of 15)
    """
    df = _ensure_delay_numeric(df)
    df, key = _ensure_slot_key(df)

    g = (
        df.groupby(key)
          .agg(
              flights=("Flight Number", "count"),
              avg_dep_delay=(DELAY_COL, "mean"),
              p50_dep_delay=(DELAY_COL, lambda s: s.quantile(0.5)),
              p90_dep_delay=(DELAY_COL, lambda s: s.quantile(0.9)),
          )
          .reset_index()
    )

    # Green window threshold: p90 <= 25 minutes
    g["is_green"] = g["p90_dep_delay"] <= green_threshold

    # Friendly label for plotting if bucket is integer
    g["slot_label"] = _slot_label(g[key])

    # Sort and round for nicer display
    g = g.sort_values(by=key).reset_index(drop=True)
    return g.round({"avg_dep_delay": 2, "p50_dep_delay": 2, "p90_dep_delay": 2})


def slot_stats_in_range(
    df: pd.DataFrame,
    start_min: Optional[int] = None,
    end_min: Optional[int] = None
) -> pd.DataFrame:
    """
    Same as slot_stats but optionally filters by minute-of-day range [start_min, end_min].
    Useful for queries like "best between 06:00–09:00".
    """
    src = df.copy()
    if start_min is not None and end_min is not None and "STD_MinOfDay" in src.columns:
        src = src[(src["STD_MinOfDay"] >= start_min) & (src["STD_MinOfDay"] <= end_min)]
    return slot_stats(src)


def top_routes(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Top routes by average departure delay (then by flights), descending.
    Expects 'route' to be present (ingest step creates it).
    """
    df = _ensure_delay_numeric(df)
    if "route" not in df.columns:
        return pd.DataFrame(columns=["route", "flights", "avg_dep_delay"])

    out = (
        df.groupby("route")
          .agg(
              flights=("Flight Number", "count"),
              avg_dep_delay=(DELAY_COL, "mean"),
          )
          .reset_index()
          .sort_values(["avg_dep_delay", "flights"], ascending=[False, False])
          .head(n)
    )
    return out.round(2)


def top_airlines(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Top airlines by average departure delay (then by flights), descending.
    Expects 'airline' to be present (ingest step creates it).
    """
    df = _ensure_delay_numeric(df)
    if "airline" not in df.columns:
        return pd.DataFrame(columns=["airline", "flights", "avg_dep_delay", "p90_dep_delay"])

    out = (
        df.groupby("airline")
          .agg(
              flights=("Flight Number", "count"),
              avg_dep_delay=(DELAY_COL, "mean"),
              p90_dep_delay=(DELAY_COL, lambda s: s.quantile(0.9)),
          )
          .reset_index()
          .sort_values(["avg_dep_delay", "flights"], ascending=[False, False])
          .head(n)
    )
    return out.round(2)


def green_windows(df: pd.DataFrame, n: int = 20, threshold: int = 25) -> pd.DataFrame:
    """
    FIXED: More flexible green windows with configurable threshold.
    Default changed from 15 to 25 minutes for better results.
    """
    g = slot_stats(df)
    green = g[g["p90_dep_delay"] <= threshold].sort_values("p90_dep_delay", ascending=True).head(n)
    
    # If no "green" windows exist, return the best available
    if green.empty:
        print(f"No slots with P90 ≤ {threshold} min. Showing best available slots instead.")
        return g.sort_values("p90_dep_delay", ascending=True).head(n)
    
    return green
