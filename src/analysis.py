# src/analysis.py
import pandas as pd

def slot_stats(df: pd.DataFrame) -> pd.DataFrame:
    if "slot_15" in df.columns and df["slot_15"].notna().any():
        key = "slot_15"
    else:
        # minute bucket fallback
        key = "STD_MinOfDay"
        df = df.copy()
        df["slot_15_bucket"] = (df[key] // 15) * 15
        key = "slot_15_bucket"

    g = (df.groupby(key)
            .agg(flights=("Flight Number","count"),
                 avg_dep_delay=("DepartureDelayMin","mean"),
                 p90_dep_delay=("DepartureDelayMin", lambda s: s.quantile(0.9)))
            .reset_index()
        )
    g["is_green"] = g["p90_dep_delay"] <= 15
    return g.sort_values(by=g.columns[0])

def top_routes(df: pd.DataFrame, n=10) -> pd.DataFrame:
    return (df.groupby("route")
              .agg(flights=("Flight Number","count"),
                   avg_dep_delay=("DepartureDelayMin","mean"))
              .reset_index()
              .sort_values(["avg_dep_delay","flights"], ascending=[False,False])
              .head(n))

def top_airlines(df: pd.DataFrame, n=10) -> pd.DataFrame:
    return (df.groupby("airline")
              .agg(flights=("Flight Number","count"),
                   avg_dep_delay=("DepartureDelayMin","mean"))
              .reset_index()
              .sort_values(["avg_dep_delay","flights"], ascending=[False,False])
              .head(n))
