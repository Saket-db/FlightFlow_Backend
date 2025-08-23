# src/nlp.py
import re

def parse_intent(q: str):
    s = q.lower().strip()
    if re.search(r"\bbusiest\b|\bpeak\b", s):
        return {"intent":"busiest_slots"}
    if re.search(r"\bbest\b|\bgreen\b", s):
        return {"intent":"best_slots"}
    m = re.search(r"shift\s+([a-z0-9]+)\s+to\s+(\d{1,2}:\d{2})", s, re.I)
    if m: return {"intent":"shift", "flight":m.group(1).upper(), "time":m.group(2)}
    m = re.search(r"delay\s+of\s+([a-z0-9]+)", s, re.I)
    if m: return {"intent":"delay_of", "flight":m.group(1).upper()}
    if "cascade" in s or "ripple" in s:
        return {"intent":"cascade_top"}
    return {"intent":"help"}

