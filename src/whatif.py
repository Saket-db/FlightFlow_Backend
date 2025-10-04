# path: src/whatif.py
import pandas as pd
from .queueing import slot_wait_minutes, RunwayConfig

def shift_by_minutes(df: pd.DataFrame, flight_no: str, minutes_delta: int) -> pd.DataFrame:
    """Return a copy of `df` with the chosen flight's `STD_MinOfDay` shifted by minutes_delta.

    Also recomputes a generic `slot_bucket` (minute-of-day bucket) based on 15-min slots
    and a convenience `slot_load` column. This keeps compatibility with earlier code that
    expected `slot_15_bucket` while also exposing `slot_bucket` for variable bucket sizes.
    """
    sim = df.copy()
    idx = sim.index[sim["Flight Number"].astype(str) == str(flight_no)]
    if len(idx) == 0:
        return sim
    i = idx[0]

    if "STD_MinOfDay" in sim.columns:
        base = float(sim.loc[i, "STD_MinOfDay"]) if pd.notna(sim.loc[i, "STD_MinOfDay"]) else 0.0
        sim.at[i, "STD_MinOfDay"] = max(0, min(1439, base + minutes_delta))

    # recompute convenient bucket (generic name) and keep legacy 15-min bucket
    sim["slot_bucket"] = (pd.to_numeric(sim["STD_MinOfDay"], errors="coerce").fillna(-1) // 15) * 15
    sim["slot_15_bucket"] = sim["slot_bucket"]
    # keep slot_load as a transform (useful elsewhere)
    sim["slot_load"] = sim.groupby("slot_bucket")["Flight Number"].transform("count")
    return sim

def queueing_burden(df: pd.DataFrame, cfg: RunwayConfig, slot_minutes: int = 15) -> float:
    """
    Sum expected wait across the schedule using flight counts per time bucket.

    slot_minutes controls the bucket size (default 15). Using a smaller bucket (e.g. 1 or 5)
    increases sensitivity of the what-if analysis when shifting a flight.
    """
    # If a generic 'slot_bucket' exists use it (it was computed by shift_by_minutes),
    # otherwise fall back to STD_MinOfDay and the requested slot_minutes.
    if "slot_bucket" in df.columns and df["slot_bucket"].notna().any():
        buckets = df["slot_bucket"]
    else:
        buckets = (pd.to_numeric(df.get("STD_MinOfDay"), errors="coerce").fillna(-1) // slot_minutes) * slot_minutes

    # counts: number of flights per time bucket (Series indexed by bucket)
    counts = df.groupby(buckets)["Flight Number"].size()

    # Sum total expected wait across all flights: per-slot per-flight wait * number of flights
    total_wait = 0.0
    for c in counts.values:
        c_int = int(c)
        total_wait += float(c_int) * slot_wait_minutes(c_int, cfg)

    return float(total_wait)


def compute_queueing_stats(df: pd.DataFrame, cfg: RunwayConfig, slot_minutes: int = 15) -> dict:
    """Return a dict with breakdown of queueing burden for easier display.

    Returns:
      {
        "total_wait": float,            # total queued minutes across all flights
        "total_flights": int,           # number of flights considered
        "avg_wait_per_flight": float,   # total_wait / total_flights (or 0)
        "per_slot": [                   # list of per-slot summaries
            {"bucket": int, "count": int, "per_flight_wait": float, "total_wait": float}, ...
        ]
      }
    """
    if "slot_bucket" in df.columns and df["slot_bucket"].notna().any():
        buckets = df["slot_bucket"]
    else:
        buckets = (pd.to_numeric(df.get("STD_MinOfDay"), errors="coerce").fillna(-1) // slot_minutes) * slot_minutes

    counts = df.groupby(buckets)["Flight Number"].size()
    per_slot = []
    total_wait = 0.0
    total_flights = int(counts.sum()) if not counts.empty else 0

    for bucket, c in counts.items():
        c_int = int(c)
        per_flight_wait = slot_wait_minutes(c_int, cfg)
        slot_total = float(c_int) * per_flight_wait
        per_slot.append({
            "bucket": int(bucket) if pd.notna(bucket) else -1,
            "count": c_int,
            "per_flight_wait": float(per_flight_wait),
            "total_wait": float(slot_total),
        })
        total_wait += slot_total

    avg = float(total_wait / total_flights) if total_flights > 0 else 0.0
    return {
        "total_wait": float(total_wait),
        "total_flights": total_flights,
        "avg_wait_per_flight": float(avg),
        "per_slot": per_slot,
    }
