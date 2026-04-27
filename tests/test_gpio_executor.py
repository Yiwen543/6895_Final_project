import pytest
from unittest.mock import patch

# conftest.py has already injected mock lgpio into sys.modules
from gpio_executor import GPIOExecutor


@pytest.fixture
def executor():
    return GPIOExecutor()


def test_light_turn_on_sets_white(executor):
    with patch.object(executor, '_set_color') as mock_color:
        executor.execute({'device': 'light', 'action': 'turn_on', 'value': None})
    mock_color.assert_called_once_with(255, 255, 255)


def test_light_turn_off_sets_black(executor):
    with patch.object(executor, '_set_color') as mock_color:
        executor.execute({'device': 'light', 'action': 'turn_off', 'value': None})
    mock_color.assert_called_once_with(0, 0, 0)


def test_light_set_brightness_50(executor):
    with patch.object(executor, '_set_color') as mock_color:
        executor.execute({'device': 'light', 'action': 'set_brightness', 'value': 50})
    mock_color.assert_called_once_with(127, 127, 127)


def test_light_set_brightness_100(executor):
    with patch.object(executor, '_set_color') as mock_color:
        executor.execute({'device': 'light', 'action': 'set_brightness', 'value': 100})
    mock_color.assert_called_once_with(255, 255, 255)


def test_curtain_open_moves_to_100(executor):
    with patch.object(executor, '_move_to_position') as mock_move:
        executor.execute({'device': 'curtain', 'action': 'open', 'value': None})
    mock_move.assert_called_once_with(100)


def test_curtain_close_moves_to_0(executor):
    with patch.object(executor, '_move_to_position') as mock_move:
        executor.execute({'device': 'curtain', 'action': 'close', 'value': None})
    mock_move.assert_called_once_with(0)


def test_curtain_set_position(executor):
    with patch.object(executor, '_move_to_position') as mock_move:
        executor.execute({'device': 'curtain', 'action': 'set_position', 'value': 75})
    mock_move.assert_called_once_with(75)


def test_window_ac_return_stub(executor):
    assert '[STUB]' in executor.execute({'device': 'window', 'action': 'open', 'value': None})
    assert '[STUB]' in executor.execute({'device': 'ac', 'action': 'turn_on', 'value': None})


def test_position_tracking(executor):
    executor._curtain_pos = 0
    with patch.object(executor, '_do_step'):
        executor._move_to_position(50)
    assert executor._curtain_pos == 50

    with patch.object(executor, '_do_step'):
        executor._move_to_position(20)
    assert executor._curtain_pos == 20


def test_rgb_cycle_starts_and_stops_thread(executor):
    with patch.object(executor, '_set_color'):
        executor._start_rgb_cycle()
        assert executor._rgb_thread is not None
        assert executor._rgb_thread.is_alive()
        executor._stop_rgb_cycle()
        assert not executor._rgb_thread.is_alive()
