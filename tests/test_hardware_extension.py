import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_window_pins_in_config():
    import config
    assert hasattr(config, "WINDOW_PINS")
    assert config.WINDOW_PINS == [17, 27, 22, 23]
    assert len(config.WINDOW_PINS) == 4

def test_window_total_steps_in_config():
    import config
    assert hasattr(config, "WINDOW_TOTAL_STEPS")
    assert config.WINDOW_TOTAL_STEPS > 0

def test_get_device_state_mock():
    """Test get_device_state without real hardware by monkey-patching lgpio."""
    import types, unittest.mock as mock
    import sys

    # Provide a fake lgpio so gpio_executor can be imported on non-Pi
    fake_lgpio = types.ModuleType("lgpio")
    fake_lgpio.gpiochip_open = mock.Mock(return_value=0)
    fake_lgpio.gpio_claim_output = mock.Mock()
    fake_lgpio.gpio_write = mock.Mock()
    fake_lgpio.tx_pwm = mock.Mock()
    fake_lgpio.gpiochip_close = mock.Mock()
    sys.modules["lgpio"] = fake_lgpio

    # Also stub pi5neo so LED init doesn't fail
    fake_pi5neo_mod = types.ModuleType("pi5neo")
    fake_pi5neo_mod.Pi5Neo = mock.Mock()
    sys.modules["pi5neo"] = fake_pi5neo_mod

    # Remove cached gpio_executor so it re-imports with our mocks
    if "gpio_executor" in sys.modules:
        del sys.modules["gpio_executor"]

    from gpio_executor import GPIOExecutor
    g = GPIOExecutor()

    state = g.get_device_state()
    assert state["color_temp"] == 3
    assert state["brightness"] == 100
    assert state["curtain_pos"] == 0
    assert state["window_pos"] == 0

def test_get_device_state_updates_after_command():
    import sys
    from gpio_executor import GPIOExecutor
    import unittest.mock as mock

    g = GPIOExecutor.__new__(GPIOExecutor)
    g._color_temp_level = 3
    g._brightness_level = 100
    g._curtain_pos = 0
    g._window_pos = 0
    g._strip = None
    g._rgb_stop = mock.Mock()
    g._rgb_lock = mock.MagicMock()
    g._rgb_thread = None
    g._fan_duty = 0.0
    g._step_index = 0
    g._window_step_index = 0
    g._h = 0

    # Simulate set_color_temp updating state
    with mock.patch.object(g, '_fill'), \
         mock.patch.object(g, '_stop_rgb_cycle'):
        g.execute({"device": "light", "action": "set_color_temp", "value": 4})
    assert g.get_device_state()["color_temp"] == 4

def test_warmer_increments_color_temp():
    from rule_based import try_rule_based
    result = try_rule_based("Cathey, make the light warmer", state={"color_temp": 3})
    assert result is not None
    assert result["action"] == "set_color_temp"
    assert result["value"] == 4

def test_cooler_decrements_color_temp():
    from rule_based import try_rule_based
    result = try_rule_based("Cathey, make the light cooler", state={"color_temp": 3})
    assert result is not None
    assert result["action"] == "set_color_temp"
    assert result["value"] == 2

def test_cozier_increments_color_temp():
    from rule_based import try_rule_based
    result = try_rule_based("Cathey, make it cozier", state={"color_temp": 2})
    assert result is not None
    assert result["value"] == 3

def test_color_temp_clamps_at_max():
    from rule_based import try_rule_based
    result = try_rule_based("Cathey, make the light warmer", state={"color_temp": 5})
    assert result["value"] == 5

def test_color_temp_clamps_at_min():
    from rule_based import try_rule_based
    result = try_rule_based("Cathey, make the light colder", state={"color_temp": 1})
    assert result["value"] == 1

def test_relative_defaults_to_neutral_without_state():
    from rule_based import try_rule_based
    result = try_rule_based("Cathey, make the light warmer", state=None)
    assert result["value"] == 4  # 3 + 1

def test_existing_rules_still_work_with_state():
    from rule_based import try_rule_based
    result = try_rule_based("Cathey, turn on the light", state={"color_temp": 3})
    assert result is not None
    assert result["action"] == "turn_on"

def test_make_it_cooler_without_light_falls_through():
    # "make it cooler" is ambiguous — should go to LLM, not rule-based
    from rule_based import try_rule_based
    result = try_rule_based("Cathey, make it cooler", state={"color_temp": 3})
    assert result is None

def test_agent_passes_state_to_rule_based():
    import unittest.mock as mock
    import sys, types

    # Stub lgpio + pi5neo for import if not already stubbed
    for mod in ("lgpio", "pi5neo"):
        if mod not in sys.modules:
            fake = types.ModuleType(mod)
            if mod == "lgpio":
                for fn in ("gpiochip_open","gpio_claim_output","gpio_write","tx_pwm","gpiochip_close"):
                    setattr(fake, fn, mock.Mock(return_value=0))
            else:
                fake.Pi5Neo = mock.Mock()
            sys.modules[mod] = fake

    # Fresh import of agent
    if "agent" in sys.modules:
        del sys.modules["agent"]
    from agent import CatheyAgent
    from gpio_executor import GPIOExecutor

    gpio = GPIOExecutor.__new__(GPIOExecutor)
    gpio._color_temp_level = 4
    gpio._brightness_level = 80
    gpio._curtain_pos = 0
    gpio._window_pos = 0

    llm_mock = mock.Mock()
    llm_mock.parse_unified = mock.Mock(return_value=({"type": "invalid"}, None, 0.0))
    agent = CatheyAgent(llm=llm_mock, memory=mock.Mock(), speak=mock.Mock(), gpio=gpio)

    with mock.patch("agent.try_rule_based", return_value=None) as mock_rb:
        agent.handle("Cathey, make the light warmer", verbose=False)
        assert mock_rb.called
        call_args = mock_rb.call_args[0]  # positional args tuple
        assert len(call_args) == 2, "try_rule_based should be called with (text, state)"
        passed_state = call_args[1]
        assert passed_state is not None
        assert passed_state["color_temp"] == 4
