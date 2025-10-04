# path: src/queueing.py
from dataclasses import dataclass
from typing import Dict, Tuple
import pandas as pd
import numpy as np

"""
Queueing & runway capacity utilities.

- RunwayConfig.mu(): effective service rate (flights/min) used in M/M/1.
- Optional 'realistic capacity' helpers for arrivals/departures per hour.
- mm1_wait() and slot_wait_minutes() used by the What-If simulator.
"""

@dataclass
class RunwayConfig:
    # Basic knobs
    mode: str = "single"            # "single" | "segregated" | "parallel_independent"
    weather: str = "VMC"            # "VMC" | "IMC" | "Rain" | "Storm"

    # For simple M/M/1 modeling (used by what-if/queueing_burden)
    base_mu_per_min: float = 0.70   # baseline throughput ≈ 42 movements/hour

    # For optional ‘realistic capacity’ views (arr/dep per hour)
    num_runways: int = 2
    base_arrival_capacity: float = 30.0   # arrivals/hour in VMC
    base_departure_capacity: float = 24.0 # departures/hour in VMC

    # ---------- Simple effective service rate for M/M/1 ----------
    def mu(self) -> float:
        """Effective service rate μ (flights/min) adjusted by mode & weather."""
        mode_factor = {
            "single": 1.00,              # shared runway
            "segregated": 1.10,          # dedicated arr/dep
            "parallel_independent": 1.25 # strong boost
        }.get(self.mode.lower(), 1.00)

        weather_factor = {
            "VMC": 1.00,
            "IMC": 0.85,
            "Rain": 0.80,
            "Storm": 0.65,
        }.get(self.weather, 1.00)

        return max(0.10, self.base_mu_per_min * mode_factor * weather_factor)

    # ---------- Optional: “realistic capacity” per hour ----------
    def get_realistic_capacity(self) -> Tuple[float, float]:
        """
        Returns (arrival_capacity_per_hour, departure_capacity_per_hour)
        for dashboards that want per-hour capacities instead of μ.
        """
        weather_factors = {
            "VMC": 1.00,
            "IMC": 0.75,
            "Rain": 0.65,
            "Storm": 0.45,
        }
        wf = weather_factors.get(self.weather, 1.0)

        if self.mode == "single":
            total = 35.0 * wf  # movements/hour
            arr_cap = min(self.base_arrival_capacity * wf, total * 0.7)
            dep_cap = max(0.0, total - arr_cap)
        elif self.mode == "segregated":
            arr_cap = self.base_arrival_capacity * wf
            dep_cap = self.base_departure_capacity * wf
        elif self.mode == "parallel_independent":
            arr_cap = self.base_arrival_capacity * wf * 1.8
            dep_cap = self.base_departure_capacity * wf * 1.8
        else:
            arr_cap = dep_cap = 25.0 * wf

        return arr_cap, dep_cap

    def get_operational_info(self) -> Dict:
        """Human-readable summary for UI."""
        arr_cap, dep_cap = self.get_realistic_capacity()
        weather_reduction = {
            "VMC": 0.0, "IMC": 0.25, "Rain": 0.35, "Storm": 0.55
        }.get(self.weather, 0.0)
        return {
            "mode": self.mode,
            "weather": self.weather,
            "arrival_capacity_per_hour": round(arr_cap, 1),
            "departure_capacity_per_hour": round(dep_cap, 1),
            "total_movements_per_hour": round(arr_cap + dep_cap, 1),
            "capacity_utilization": "High" if (arr_cap + dep_cap) > 50 else "Normal",
            "recommended_slot_buffer_min": 3 if self.mode == "single" else 2,
            "weather_impact": f"{int(weather_reduction*100)}% reduction",
        }

    # Optional: coarse delay buckets for per-slot narratives
    def get_delay_impact(self, flight_type: str, slot_load: int) -> float:
        arr_cap, dep_cap = self.get_realistic_capacity()
        if flight_type == "arrival":
            threshold1, threshold2 = arr_cap / 4.0, arr_cap / 3.0
            if slot_load <= threshold1: return 2.0
            if slot_load <= threshold2: return 8.0
            return 25.0
        else:
            threshold1, threshold2 = dep_cap / 4.0, dep_cap / 3.0
            if slot_load <= threshold1: return 3.0
            if slot_load <= threshold2: return 12.0
            return 35.0


# ----------------- Queueing helpers -----------------
def mm1_wait(lambda_per_min: float, mu_per_min: float) -> float:
    """Expected waiting time in queue (minutes) for M/M/1."""
    if mu_per_min <= 0:
        return float("inf")
    rho = lambda_per_min / mu_per_min
    if rho >= 1.0:
        # For saturated systems, return a large but finite value
        # This allows for meaningful comparisons when shifting flights
        return 1000.0 + (rho - 1.0) * 100.0
    return rho / (mu_per_min * (1.0 - rho))


def slot_wait_minutes(slot_load: int, cfg: RunwayConfig) -> float:
    """
    Expected wait (minutes) for a 15-min bucket containing `slot_load` flights,
    under the current runway config.
    """
    lam = slot_load / 15.0  # arrivals per minute (15-min slot)
    return mm1_wait(lam, cfg.mu())


# -------------- Optional analytics (not required by app) --------------
def analyze_runway_efficiency(df: pd.DataFrame, config: RunwayConfig) -> pd.DataFrame:
    """Illustrative per-slot analysis using coarse arrival/departure mix."""
    x = df.copy()
    # Derive 15-min bucket from minutes-of-day if needed
    if "slot_15_bucket" not in x.columns:
        x["slot_15_bucket"] = (pd.to_numeric(x.get("STD_MinOfDay"), errors="coerce").fillna(-1) // 15) * 15

    # Simulate flight type split (60% arrivals / 40% departures)
    rng = np.random.default_rng(42)
    x["flight_type"] = rng.choice(["arrival", "departure"], size=len(x), p=[0.6, 0.4])

    rows = []
    for slot, grp in x.groupby("slot_15_bucket"):
        slot_load = len(grp)
        n_arr = (grp["flight_type"] == "arrival").sum()
        n_dep = slot_load - n_arr
        arr_delay = config.get_delay_impact("arrival", n_arr) if n_arr > 0 else 0.0
        dep_delay = config.get_delay_impact("departure", n_dep) if n_dep > 0 else 0.0
        avg_delay = (arr_delay * n_arr + dep_delay * n_dep) / slot_load if slot_load else 0.0
        rows.append({
            "slot": slot,
            "slot_label": f"{int(slot//60):02d}:{int(slot%60):02d}",
            "total_flights": slot_load,
            "arrivals": n_arr,
            "departures": n_dep,
            "predicted_arr_delay": arr_delay,
            "predicted_dep_delay": dep_delay,
            "avg_predicted_delay": avg_delay,
            "capacity_stress": "High" if slot_load > 12 else ("Medium" if slot_load > 8 else "Low"),
        })
    return pd.DataFrame(rows)


def runway_comparison_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Compare several configurations side-by-side (demonstration)."""
    configs = [
        ("Single (VMC)", RunwayConfig("single", "VMC")),
        ("Segregated (VMC)", RunwayConfig("segregated", "VMC")),
        ("Segregated (Rain)", RunwayConfig("segregated", "Rain")),
        ("Parallel Independent (VMC)", RunwayConfig("parallel_independent", "VMC")),
    ]
    out = []
    for name, cfg in configs:
        ana = analyze_runway_efficiency(df, cfg)
        arr_cap, dep_cap = cfg.get_realistic_capacity()
        out.append({
            "Configuration": name,
            "Avg Delay (min)": round(float(ana["avg_predicted_delay"].mean()), 2) if not ana.empty else 0.0,
            "High Stress Slots": int((ana["capacity_stress"] == "High").sum()),
            "Total Capacity/hr": round(arr_cap + dep_cap, 1),
            "Weather Impact": cfg.get_operational_info()["weather_impact"],
        })
    return pd.DataFrame(out)
