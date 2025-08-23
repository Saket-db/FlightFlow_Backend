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
- set runway mode to (single|segregated) → {"intent":"set_mode", "mode":...}
- set weather to (VMC|IMC|Rain|Storm) → {"intent":"set_weather", "weather":...}
- delay of <flight> (alias for predict) → {"intent":"predict", "flight":...}
Fallback: {"intent":"help"}
"""

def parse_intent(q: str):
    s = q.lower().strip()

    # Runway mode
    m = re.search(r'set runway mode to (single|segregated)', s)
    if m:
        return {"intent": "set_mode", "mode": m.group(1)}

    # Weather condition
    m = re.search(r'set weather to (vmc|imc|rain|storm)', s)
    if m:
        return {"intent": "set_weather", "weather": m.group(1).upper()}

    # Busiest
    if re.search(r"\bbusiest\b|\bpeak\b", s):
        return {"intent": "busiest"}

    # Best slots
    if re.search(r"\bbest\b|\bgreen\b", s):
        return {"intent": "best"}

    # Cascade
    if "cascade" in s or "ripple" in s:
        return {"intent": "cascade"}

    # Shift by X minutes
    m = re.search(r"shift\s+([a-z0-9]+)\s+by\s+(-?\d+)\s*min", s, re.I)
    if m:
        return {"intent": "shift_by", "flight": m.group(1).upper(), "mins": int(m.group(2))}

    # Shift to specific time
    m = re.search(r"shift\s+([a-z0-9]+)\s+to\s+(\d{1,2}:\d{2})", s, re.I)
    if m:
        return {"intent": "shift_to", "flight": m.group(1).upper(), "time": m.group(2)}

    # Predict delay / delay of
    m = re.search(r"predict\s+delay\s+for\s+([a-z0-9]+)", s, re.I)
    if m:
        return {"intent": "predict", "flight": m.group(1).upper()}
    m = re.search(r"delay\s+of\s+([a-z0-9]+)", s, re.I)
    if m:
        return {"intent": "predict", "flight": m.group(1).upper()}

    return {"intent": "help"}
