# path: src/whatif.py
import pandas as pd
from .queueing import slot_wait_minutes, RunwayConfig

def shift_by_minutes(df: pd.DataFrame, flight_no: str, minutes_delta: int) -> pd.DataFrame:
    sim = df.copy()
    idx = sim.index[sim["Flight Number"].astype(str) == str(flight_no)]
    if len(idx) == 0:
        return sim
    i = idx[0]

    if "STD_MinOfDay" in sim.columns:
        base = float(sim.loc[i, "STD_MinOfDay"]) if pd.notna(sim.loc[i, "STD_MinOfDay"]) else 0.0
        sim.at[i, "STD_MinOfDay"] = max(0, min(1439, base + minutes_delta))

    # recompute convenient bucket
    sim["slot_15_bucket"] = (pd.to_numeric(sim["STD_MinOfDay"], errors="coerce").fillna(-1) // 15) * 15
    # keep slot_load as a transform (useful elsewhere)
    sim["slot_load"] = sim.groupby("slot_15_bucket")["Flight Number"].transform("count")
    return sim

def queueing_burden(df: pd.DataFrame, cfg: RunwayConfig) -> float:
    """
    Sum expected wait per slot using actual flight counts per 15-min bucket.
    This avoids relying on a possibly stale or empty 'slot_load' column.
    """
    if "slot_15" in df.columns and df["slot_15"].notna().any():
        # count flights per timestamp bucket
        counts = df.groupby("slot_15")["Flight Number"].size().values
    else:
        buckets = (pd.to_numeric(df.get("STD_MinOfDay"), errors="coerce").fillna(-1) // 15) * 15
        counts = df.groupby(buckets)["Flight Number"].size().values

    # sum the queueing wait for each bucket
    return float(sum(slot_wait_minutes(int(c), cfg) for c in counts))
