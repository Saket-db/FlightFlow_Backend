# path: src/cascade.py
import pandas as pd
import numpy as np

def cascade_risk(df: pd.DataFrame, top_n=20) -> pd.DataFrame:
    """
    Approximate cascading risk without tail numbers:
    - Tight scheduled block (SchedBlockMin) + high slot load + high P90 delay in slot.
    Score = w1*(1/(1+SchedBlockMin)) + w2*slot_load_norm + w3*p90_slot_norm
    """
    x = df.copy()

    # derive slot key
    if "slot_15" in x.columns and x["slot_15"].notna().any():
        key = "slot_15"
    else:
        x["slot_15_bucket"] = (pd.to_numeric(x.get("STD_MinOfDay"), errors="coerce").fillna(-1) // 15) * 15
        key = "slot_15_bucket"

    # slot p90 delay
    delay_col = "DepartureDelayMin"
    x[delay_col] = pd.to_numeric(x.get(delay_col), errors="coerce")
    slot_p90 = x.groupby(key)[delay_col].quantile(0.9).rename("slot_p90").reset_index()
    x = x.merge(slot_p90, on=key, how="left")

    # normalize helpers
    def norm(s):
        s = pd.to_numeric(s, errors="coerce").fillna(0)
        rng = s.max() - s.min()
        return (s - s.min())/rng if rng > 0 else s*0

    # ensure slot_load exists as a count
    if "slot_load" not in x.columns or x["slot_load"].isna().all():
        x["slot_load"] = x.groupby(key)["Flight Number"].transform("count")

    x["turn_tight"]    = 1.0 / (1.0 + pd.to_numeric(x.get("SchedBlockMin"), errors="coerce").fillna(0))
    x["slot_load_norm"] = norm(x["slot_load"])
    x["slot_p90_norm"]  = norm(x["slot_p90"])

    w1, w2, w3 = 0.45, 0.30, 0.25
    x["cascade_score"] = w1*x["turn_tight"] + w2*x["slot_load_norm"] + w3*x["slot_p90_norm"]

    keep = x[["Flight Number","From","To","TimeSlot","SchedBlockMin","slot_load","slot_p90","cascade_score"]].copy()
    return keep.sort_values("cascade_score", ascending=False).head(top_n).round(2)
