# path: src/nlp.py
import re

"""
NLP intent parser for the chatbot interface.

Supported intents:
- busiest / peak → {"intent":"busiest"}
- best / green → {"intent":"best"}
- cascade / ripple → {"intent":"cascade"}
- shift <flight> by <mins> → {"intent":"shift_by", "flight":..., "mins":...}
- shift <flight> to <HH:MM> → {"intent":"shift_to", "flight":..., "time":...}
- predict delay for <flight> → {"intent":"predict", "flight":...}
- set runway mode to (single|segregated|parallel_independent) → {"intent":"set_mode", "mode":...}
- set weather to (VMC|IMC|Rain|Storm) → {"intent":"set_weather", "weather":...}
- delay of <flight> (alias for predict) → {"intent":"predict", "flight":...}

NEW:
- (best|worst) flights on <ORIG>(-|->|to)<DEST> [top N]
  Examples:
    "best flights on BOM->DEL"
    "worst flights from DEL to BLR top 5"
    "best flights on route HYD-CCU"
  → {"intent":"route_rank", "kind":"best"|"worst", "origin":"BOM", "dest":"DEL", "top_n":5}
"""

def parse_intent(q: str):
    s = q.lower().strip()

    # ---------- runway config ----------
    m = re.search(r'set runway mode to (single|segregated|parallel_independent)', s)
    if m:
        return {"intent": "set_mode", "mode": m.group(1)}
    m = re.search(r'set weather to (vmc|imc|rain|storm)', s)
    if m:
        return {"intent": "set_weather", "weather": m.group(1).upper()}

    # ---------- core analytics ----------
    if re.search(r"\bbusiest\b|\bpeak\b", s):
        return {"intent": "busiest"}
    if re.search(r"\bbest\b|\bgreen\b", s):
        return {"intent": "best"}
    if "cascade" in s or "ripple" in s:
        return {"intent": "cascade"}

    # ---------- what-if shift ----------
    m = re.search(r"shift\s+([a-z0-9]+)\s+by\s+(-?\d+)\s*min", s, re.I)
    if m:
        return {"intent": "shift_by", "flight": m.group(1).upper(), "mins": int(m.group(2))}
    m = re.search(r"shift\s+([a-z0-9]+)\s+to\s+(\d{1,2}:\d{2})", s, re.I)
    if m:
        return {"intent": "shift_to", "flight": m.group(1).upper(), "time": m.group(2)}

    # ---------- predictions ----------
    m = re.search(r"predict\s+delay\s+for\s+([a-z0-9]+)", s, re.I)
    if m:
        return {"intent": "predict", "flight": m.group(1).upper()}
    m = re.search(r"delay\s+of\s+([a-z0-9]+)", s, re.I)
    if m:
        return {"intent": "predict", "flight": m.group(1).upper()}

    # ---------- NEW: best / worst flights for a specific route ----------
    # Accept:
    #   "best flights on bom->del [top 5]"
    #   "worst flights from bom to del"
    #   "best flights on route hyd-ccu top10"
    route_patterns = [
        r"(best|worst)\s+flights\s+(?:on|for|from)\s+([a-z]{3})\s*(?:-|->|to)\s*([a-z]{3})(?:.*?\btop\s*(\d+))?",
        r"(best|worst)\s+flights\s+on\s+route\s+([a-z]{3})\s*(?:-|->|to)\s*([a-z]{3})(?:.*?\btop\s*(\d+))?",
    ]
    for pat in route_patterns:
        m = re.search(pat, s, re.I)
        if m:
            kind = m.group(1).lower()              # best | worst
            origin = m.group(2).upper()
            dest = m.group(3).upper()
            top_n = int(m.group(4)) if m.lastindex and m.group(4) else 10
            return {"intent": "route_rank", "kind": kind, "origin": origin, "dest": dest, "top_n": top_n}

    return {"intent": "help"}
