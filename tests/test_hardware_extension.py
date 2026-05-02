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
