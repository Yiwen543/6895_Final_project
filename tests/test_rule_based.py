from rule_based import try_rule_based


def test_turn_on_light():
    assert try_rule_based("Nova, turn on the light") == {
        "type": "direct_command", "device": "light", "action": "turn_on", "value": None
    }


def test_turn_off_light():
    assert try_rule_based("Nova, switch off the light") == {
        "type": "direct_command", "device": "light", "action": "turn_off", "value": None
    }


def test_open_curtain():
    assert try_rule_based("Nova, open the curtain") == {
        "type": "direct_command", "device": "curtain", "action": "open", "value": None
    }


def test_close_window():
    assert try_rule_based("Nova, close the window") == {
        "type": "direct_command", "device": "window", "action": "close", "value": None
    }


def test_ac_temperature():
    assert try_rule_based("Nova, set the AC to 24 degrees") == {
        "type": "direct_command", "device": "ac", "action": "set_temperature", "value": 24
    }


def test_ac_temperature_out_of_range_returns_none():
    assert try_rule_based("Nova, set the AC to 50 degrees") is None


def test_set_brightness():
    assert try_rule_based("Nova, set brightness to 70") == {
        "type": "direct_command", "device": "light", "action": "set_brightness", "value": 70
    }


def test_set_curtain_position():
    assert try_rule_based("Nova, set curtain to 50 percent") == {
        "type": "direct_command", "device": "curtain", "action": "set_position", "value": 50
    }


def test_rgb_cycle():
    assert try_rule_based("Nova, RGB cycle") == {
        "type": "direct_command", "device": "light", "action": "rgb_cycle", "value": None
    }


def test_ambiguous_falls_through():
    assert try_rule_based("Nova, I feel cold") is None
    assert try_rule_based("Nova, it's dark") is None
    assert try_rule_based("Hello") is None


def test_case_insensitive():
    result = try_rule_based("NOVA, TURN ON THE LIGHT")
    assert result is not None
    assert result["action"] == "turn_on"
