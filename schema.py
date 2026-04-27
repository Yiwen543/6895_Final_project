"""
schema.py — device schema, validation, and execution.

Design principles:
  - validate_command: purely data-driven from COMMAND_SCHEMA, no if-else
  - execute_command:  dict lookup table, no if-else
  - build_execution_reply is intentionally ABSENT: the LLM generates
    a natural-language "reply" field directly inside the JSON output,
    so no hardcoded reply templates are needed here.
"""

from typing import Any, Dict, List, Tuple

# ── Device capability schema ──────────────────────────────────────────────────
# value rule: None → value must be null; (lo, hi) → value must be int in [lo, hi]

COMMAND_SCHEMA: Dict[str, Any] = {
    "light": {
        "actions": {
            "turn_on":        None,
            "turn_off":       None,
            "set_brightness": (0, 100),
            "rgb_cycle":      None,
        }
    },
    "curtain": {
        "actions": {
            "open":         None,
            "close":        None,
            "set_position": (0, 100),
        }
    },
    "window": {
        "actions": {
            "open":         None,
            "close":        None,
            "set_position": (0, 100),
        }
    },
    "ac": {
        "actions": {
            "turn_on":         None,
            "turn_off":        None,
            "set_temperature": (16, 30),
        }
    },
}

# ── Hardware execution table ───────────────────────────────────────────────────
# Maps (device, action) → log string sent to device driver.
# Use {value} placeholder where a numeric parameter is needed.

_EXEC_TABLE: Dict[Tuple[str, str], str] = {
    ("light",   "turn_on"):          "LIGHT -> ON",
    ("light",   "turn_off"):         "LIGHT -> OFF",
    ("light",   "set_brightness"):   "LIGHT -> BRIGHTNESS {value}%",
    ("light",   "rgb_cycle"):        "LIGHT -> RGB CYCLE",
    ("curtain", "open"):             "CURTAIN -> OPEN",
    ("curtain", "close"):            "CURTAIN -> CLOSE",
    ("curtain", "set_position"):     "CURTAIN -> POSITION {value}%",
    ("window",  "open"):             "WINDOW -> OPEN",
    ("window",  "close"):            "WINDOW -> CLOSE",
    ("window",  "set_position"):     "WINDOW -> POSITION {value}%",
    ("ac",      "turn_on"):          "AC -> ON",
    ("ac",      "turn_off"):         "AC -> OFF",
    ("ac",      "set_temperature"):  "AC -> TEMPERATURE {value}C",
}

# ── Option display map (for clarification UI) ─────────────────────────────────

OPTION_DISPLAY_MAP: Dict[str, str] = {
    "close_window":         "Close the window",
    "raise_ac_temperature": "Raise the AC temperature",
    "lower_ac_temperature": "Lower the AC temperature",
    "turn_on_light":        "Turn on the light",
    "open_curtain":         "Open the curtain",
    "turn_off_light":       "Turn off the light",
    "dim_light":            "Dim the light",
    "rgb_cycle":            "Start RGB cycle mode",
    "turn_on_ac":           "Turn on the AC",
    "turn_off_ac":          "Turn off the AC",
    "close_curtain":        "Close the curtain",
    "open_window":          "Open the window",
}


# ── Validation ────────────────────────────────────────────────────────────────

def validate_command(cmd: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Purely data-driven validation against COMMAND_SCHEMA.
    Returns (is_valid, reason_string).
    """
    if not isinstance(cmd, dict):
        return False, "command_not_dict"
    if set(cmd.keys()) - {"device", "action", "value", "reply"}:
        return False, "invalid_keys"
    if not {"device", "action", "value"} <= set(cmd.keys()):
        return False, "missing_required_keys"

    device = cmd["device"]
    action = cmd["action"]
    value  = cmd["value"]

    if device not in COMMAND_SCHEMA:
        return False, f"unknown_device:{device}"

    valid_actions = COMMAND_SCHEMA[device]["actions"]
    if action not in valid_actions:
        return False, f"unknown_action:{action}_for_{device}"

    rule = valid_actions[action]
    if rule is None:
        return (False, "value_must_be_null") if value is not None else (True, "ok")

    if not isinstance(value, int):
        return False, "value_must_be_int"
    lo, hi = rule
    if not (lo <= value <= hi):
        return False, f"value_{value}_out_of_range_{lo}_{hi}"

    return True, "ok"


# ── Execution ─────────────────────────────────────────────────────────────────

def execute_command(cmd: Dict[str, Any]) -> str:
    """
    Dict-lookup execution — no if-else chains.
    Returns a log string representing the hardware action taken.
    """
    key = (cmd["device"], cmd["action"])
    template = _EXEC_TABLE.get(key, "NO ACTION EXECUTED")
    return template.format(value=cmd.get("value") or "")


# ── Clarification helpers ─────────────────────────────────────────────────────

def option_to_display(option: str) -> str:
    return OPTION_DISPLAY_MAP.get(option, option.replace("_", " ").capitalize())


def build_clarification_reply(question: str, options: List[str]) -> str:
    lines = [question, "Here are your options:"]
    for idx, opt in enumerate(options, start=1):
        lines.append(f"Option {idx}: {option_to_display(opt)}.")
    lines.append("Please say the option number or describe what you want.")
    return " ".join(lines)
