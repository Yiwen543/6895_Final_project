import re
from typing import Optional, Dict, Any


def try_rule_based(text: str, state: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
    t = text.lower()

    # Relative color temperature: "warmer/cozier" → +1, "cooler/colder" → -1
    _warmer = re.search(r'\b(?:warmer|cozier|more\s+warm)\b', t)
    _cooler = re.search(r'\b(?:cooler|colder|more\s+coo?l|more\s+cold)\b', t)
    if _warmer or _cooler:
        if re.search(r'\blight\b', t) or re.search(r'\bmake\s+it\b', t):
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
