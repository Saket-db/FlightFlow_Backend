# path: src/queueing.py
from dataclasses import dataclass

"""
Queueing model utilities for runway capacity / congestion.
- Implements an M/M/1 wait estimator.
- Adds RunwayConfig to reflect mode (single vs segregated) and weather (VMC/IMC/Rain/Storm).
- Effective service rate μ (mu_per_min) is adjusted by these factors.
"""

@dataclass
class RunwayConfig:
    mode: str = "single"        # "single" or "segregated"
    weather: str = "VMC"        # "VMC" (good), "IMC" (poor), "Rain", "Storm"
    base_mu_per_min: float = 0.70  # baseline service rate (flights/min) ~ 42 per hr

    def mu(self) -> float:
        """Effective service rate (μ, flights/min) adjusted for mode and weather."""
        # Mode factor
        mode_factor = {
            "single": 1.00,       # both arrivals and departures share runway
            "segregated": 1.10,   # separate runways slightly increase throughput
        }.get(self.mode.lower(), 1.00)

        # Weather factor
        weather_factor = {
            "VMC": 1.00,    # Visual Meteorological Conditions (good)
            "IMC": 0.85,    # Instrument conditions
            "Rain": 0.80,
            "Storm": 0.65,
        }.get(self.weather, 1.00)

        return max(0.10, self.base_mu_per_min * mode_factor * weather_factor)


def mm1_wait(lambda_per_min: float, mu_per_min: float) -> float:
    """Expected waiting time in queue (minutes) for M/M/1."""
    if mu_per_min <= 0:
        return float("inf")
    rho = lambda_per_min / mu_per_min
    if rho >= 1.0:
        return 999.0  # saturated
    return rho / (mu_per_min * (1.0 - rho))


def slot_wait_minutes(slot_load: int, cfg: RunwayConfig) -> float:
    """
    Expected waiting time (minutes) for a 15-min slot with given number of flights,
    under the current runway config.
    """
    lam = slot_load / 15.0  # arrivals per minute
    return mm1_wait(lam, cfg.mu())
