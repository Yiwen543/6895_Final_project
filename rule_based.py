import re
from typing import Optional, Dict, Any


def try_rule_based(text: str, state: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
    t = text.lower()

    # Stop party / stop music (must come before party_mode to avoid false match)
    if re.search(r'\bstop\b.*\b(?:party|music|song|rick)\b|\b(?:party|music|song)\b.*\bstop\b', t):
        return {"type": "direct_command", "device": "light", "action": "turn_off", "value": None}

    # Party mode easter egg
    if re.search(r'\bparty\s*(?:mode|time)?\b|\bdisco\b', t):
        return {"type": "direct_command", "device": "light", "action": "party_mode", "value": None}

    # Relative brightness: "darker/dimmer" → -1, "brighter/lighter" → +1
    _darker   = re.search(r'\b(?:darker|dimmer|less\s+bright|lower\s+brightness|dim)\b', t)
    _brighter = re.search(r'\b(?:brighter|lighter|raise\s+brightness|more\s+bright)\b', t)
    if _darker or _brighter:
        if re.search(r'\blight\b', t) or re.search(r'\bmake\s+it\b', t):
            delta   = 1 if _brighter else -1
            current = (state or {}).get("brightness", 5)
            new_level = max(1, min(5, current + delta))
            return {"type": "direct_command", "device": "light",
                    "action": "set_brightness", "value": new_level * 20}

    # Relative color temperature: "warmer/cozier" → +1, "cooler/colder" → -1
    _warmer = re.search(r'\b(?:warmer|cozier|more\s+warm)\b', t)
    _cooler = re.search(r'\b(?:cooler|colder|more\s+coo?l|more\s+cold)\b', t)
    if _warmer or _cooler:
        has_light   = re.search(r'\blight\b', t)
        has_make_it = re.search(r'\bmake\s+it\b', t)
        # "make it cooler" is ambiguous (could mean AC) — require "light" for cooling
        if has_light or (_warmer and has_make_it):
            delta   = 1 if _warmer else -1
            current = (state or {}).get("color_temp", 3)
            new_val = max(1, min(5, current + delta))
            return {"type": "direct_command", "device": "light",
                    "action": "set_color_temp", "value": new_val}

    # AC temperature: "set the AC to 24 degrees" or "set 24 degrees on AC"
    m = re.search(r"(?:ac|air.?con).*?(\d+)\s*degree|(\d+)\s*degree.*?(?:ac|air.?con)", t)
    if m:
        val = int(m.group(1) or m.group(2))
        if 16 <= val <= 30:
            return {"type": "direct_command", "device": "ac", "action": "set_temperature", "value": val}

    # Brightness: "set brightness to 70" or "70 brightness"
    m = re.search(r"brightness.*?(\d+)|(\d+).*?brightness", t)
    if m:
        val = int(m.group(1) or m.group(2))
        if 0 <= val <= 100:
            return {"type": "direct_command", "device": "light", "action": "set_brightness", "value": val}

    # Color temp: "set color temp to 3" or "color temperature 4"
    m = re.search(r"color\s*temp(?:erature)?.*?([1-5])|([1-5]).*?color\s*temp(?:erature)?", t)
    if m:
        val = int(m.group(1) or m.group(2))
        return {"type": "direct_command", "device": "light", "action": "set_color_temp", "value": val}

    # Curtain/window position: "set curtain to 50 percent"
    m = re.search(r"(curtain|window).*?(\d+)\s*(?:percent|%)|(\d+)\s*(?:percent|%).*?(curtain|window)", t)
    if m:
        device = m.group(1) or m.group(4)
        val = int(m.group(2) or m.group(3))
        if 0 <= val <= 100:
            return {"type": "direct_command", "device": device, "action": "set_position", "value": val}

    # Curtain open with qualifier → set_position (must come before bare open pattern)
    if re.search(r'\b(?:open)\b.*\bcurtain\b|\bcurtain\b.*\b(?:open)\b', t):
        _qualifiers = [
            (r'\ba\s+little\b|\bslightly\b|\bjust\s+a\s+bit\b', 20),
            (r'\bhalfway\b|\bhalf(?:\s+way)?\b',                50),
            (r'\bmost\s+of\s+the\s+way\b|\bmostly\b|\bthree[\s-]quarter', 80),
        ]
        for pat, pct in _qualifiers:
            if re.search(pat, t):
                return {"type": "direct_command", "device": "curtain",
                        "action": "set_position", "value": pct}

    # Simple on/off/open/close patterns
    patterns = [
        (r"\b(?:turn on|switch on)\b.*\blight\b|\blight\b.*\b(?:turn on|switch on)\b",
         {"device": "light", "action": "turn_on", "value": None}),
        (r"\b(?:turn off|switch off)\b.*\blight\b|\blight\b.*\b(?:turn off|switch off)\b",
         {"device": "light", "action": "turn_off", "value": None}),
        (r"\brgb\b",
         {"device": "light", "action": "rgb_cycle", "value": None}),
        (r"\bopen\b.*\bcurtain\b|\bcurtain\b.*\bopen\b",
         {"device": "curtain", "action": "open", "value": None}),
        (r"\bclose\b.*\bcurtain\b|\bcurtain\b.*\bclose\b",
         {"device": "curtain", "action": "close", "value": None}),
        (r"\bopen\b.*\bwindow\b|\bwindow\b.*\bopen\b",
         {"device": "window", "action": "open", "value": None}),
        (r"\bclose\b.*\bwindow\b|\bwindow\b.*\bclose\b",
         {"device": "window", "action": "close", "value": None}),
        (r"\b(?:turn on|switch on)\b.*\b(?:ac|air.?con)\b",
         {"device": "ac", "action": "turn_on", "value": None}),
        (r"\b(?:turn off|switch off)\b.*\b(?:ac|air.?con)\b",
         {"device": "ac", "action": "turn_off", "value": None}),
    ]

    for pat, cmd in patterns:
        if re.search(pat, t):
            return {"type": "direct_command", **cmd}

    return None
