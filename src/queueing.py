# src/queueing.py
def mm1_wait(lambda_per_min: float, mu_per_min: float) -> float:
    if mu_per_min <= 0: return float("inf")
    rho = lambda_per_min / mu_per_min
    if rho >= 1.0: return 999.0
    return rho / (mu_per_min * (1.0 - rho))

def slot_wait_minutes(slot_load: int, mu_per_min: float = 0.7) -> float:
    lam = slot_load / 15.0  # 15-min bucket
    return mm1_wait(lam, mu_per_min)
