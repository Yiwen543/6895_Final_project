from typing import Any, Dict, List, Tuple

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

OPTION_DISPLAY_MAP = {
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


def validate_command(cmd: Dict[str, Any]) -> Tuple[bool, str]:
    required = {"device", "action", "value"}
    if not isinstance(cmd, dict):
        return False, "command_not_dict"
    if set(cmd.keys()) != required:
        return False, "invalid_keys"

    device, action, value = cmd["device"], cmd["action"], cmd["value"]

    if device == "unknown" and action == "invalid":
        return False, "unrecognized_command"
    if device not in COMMAND_SCHEMA:
        return False, "invalid_device"

    valid_actions = COMMAND_SCHEMA[device]["actions"]
    if action not in valid_actions:
        return False, "invalid_action_for_device"

    rule = valid_actions[action]
    if rule is None:
        return (False, "value_must_be_null") if value is not None else (True, "ok")

    if not isinstance(value, int):
        return False, "value_must_be_int"
    lo, hi = rule
    if not (lo <= value <= hi):
        return False, "value_out_of_range"
    return True, "ok"


def execute_command(cmd: Dict[str, Any]) -> str:
    device, action, value = cmd["device"], cmd["action"], cmd["value"]
    if device == "light":
        if action == "turn_on":        return "LIGHT -> ON"
        if action == "turn_off":       return "LIGHT -> OFF"
        if action == "set_brightness": return f"LIGHT -> BRIGHTNESS {value}%"
        if action == "rgb_cycle":      return "LIGHT -> RGB CYCLE"
    if device == "curtain":
        if action == "open":           return "CURTAIN -> OPEN"
        if action == "close":          return "CURTAIN -> CLOSE"
        if action == "set_position":   return f"CURTAIN -> POSITION {value}%"
    if device == "window":
        if action == "open":           return "WINDOW -> OPEN"
        if action == "close":          return "WINDOW -> CLOSE"
        if action == "set_position":   return f"WINDOW -> POSITION {value}%"
    if device == "ac":
        if action == "turn_on":        return "AC -> ON"
        if action == "turn_off":       return "AC -> OFF"
        if action == "set_temperature": return f"AC -> TEMPERATURE {value}C"
    return "NO ACTION EXECUTED"


def build_execution_reply(cmd: Dict[str, Any]) -> str:
    device, action, value = cmd["device"], cmd["action"], cmd["value"]
    if device == "light":
        if action == "turn_on":        return "Okay, I turned on the light."
        if action == "turn_off":       return "Okay, I turned off the light."
        if action == "set_brightness": return f"Okay, I set the light brightness to {value} percent."
        if action == "rgb_cycle":      return "Okay, I started the RGB cycle mode."
    if device == "curtain":
        if action == "open":           return "Okay, I opened the curtain."
        if action == "close":          return "Okay, I closed the curtain."
        if action == "set_position":   return f"Okay, I set the curtain to {value} percent."
    if device == "window":
        if action == "open":           return "Okay, I opened the window."
        if action == "close":          return "Okay, I closed the window."
        if action == "set_position":   return f"Okay, I set the window to {value} percent."
    if device == "ac":
        if action == "turn_on":        return "Okay, I turned on the air conditioner."
        if action == "turn_off":       return "Okay, I turned off the air conditioner."
        if action == "set_temperature": return f"Okay, I set the air conditioner to {value} degrees."
    return "Okay, the command was executed."


def option_to_display(option: str) -> str:
    return OPTION_DISPLAY_MAP.get(option, option.replace("_", " ").capitalize())


def build_clarification_reply(question: str, options: List[str]) -> str:
    lines = [question, "Here are your options:"]
    for idx, opt in enumerate(options, start=1):
        lines.append(f"Option {idx}: {option_to_display(opt)}.")
    lines.append("Please say the option number or describe what you want.")
    return " ".join(lines)
