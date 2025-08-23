# src/whatif.py
import pandas as pd
from .queueing import slot_wait_minutes

def shift_by_minutes(df: pd.DataFrame, flight_no: str, minutes_delta: int) -> pd.DataFrame:
    sim = df.copy()
    idx = sim.index[sim["Flight Number"] == flight_no]
    if len(idx)==0: return sim
    i = idx[0]

    # Update minute-of-day (0..1439)
    if "STD_MinOfDay" in sim.columns:
        sim.at[i, "STD_MinOfDay"] = sim.loc[i, "STD_MinOfDay"] + minutes_delta
        # keep within day bounds
        sim.at[i, "STD_MinOfDay"] = max(0, min(1439, sim.loc[i, "STD_MinOfDay"]))

    # Recompute 15-min bucket & load
    sim["slot_15_bucket"] = (sim["STD_MinOfDay"] // 15) * 15
    sim["slot_load"] = sim.groupby("slot_15_bucket")["Flight Number"].transform("count")
    return sim

def queueing_burden(df: pd.DataFrame, mu_per_min=0.7) -> float:
    if "slot_15" in df.columns and df["slot_15"].notna().any():
        s = df.groupby("slot_15")["slot_load"].first()
    else:
        s = df.groupby((df["STD_MinOfDay"] // 15) * 15)["slot_load"].first()
    return sum(slot_wait_minutes(int(x), mu_per_min) for x in s.values)
