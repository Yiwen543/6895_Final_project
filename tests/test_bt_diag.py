# tests/test_bt_diag.py
import numpy as np
import pytest

def test_make_signal_length():
    from bt_diag import make_signal
    sig = make_signal(duration=5, rate=22050)
    assert len(sig) == 5 * 22050

def test_make_signal_range():
    from bt_diag import make_signal
    sig = make_signal(duration=5, rate=22050)
    assert sig.dtype == np.float32
    assert sig.max() <= 1.0
    assert sig.min() >= -1.0

def test_play_returns_int():
    # This test mocks sounddevice so it doesn't require audio hardware.
    import unittest.mock as mock
    from bt_diag import play_and_count_xruns, make_signal
    sig = make_signal(duration=0.1, rate=22050)
    with mock.patch("bt_diag.sd") as mock_sd:
        mock_stream = mock.MagicMock()
        mock_sd.OutputStream.return_value.__enter__ = lambda s: mock_stream
        mock_sd.OutputStream.return_value.__exit__ = mock.MagicMock(return_value=False)
        result = play_and_count_xruns(sig, rate=22050)
    assert isinstance(result, int)
