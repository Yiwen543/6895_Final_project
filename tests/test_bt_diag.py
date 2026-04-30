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

def test_callback_stops_on_buffer_exhausted():
    """Callback raises CallbackStop when signal is exhausted."""
    import unittest.mock as mock
    from bt_diag import play_and_count_xruns, make_signal

    callbacks_stopped = []

    class FakeStream:
        def __init__(self, **kwargs):
            self._cb = kwargs["callback"]
        def __enter__(self):
            # Invoke callback with a full-size outdata to drain the signal
            outdata = np.zeros((2205, 1), dtype=np.float32)
            status = mock.MagicMock()
            status.output_underflow = False
            # Feed signal in chunks until CallbackStop
            try:
                for _ in range(100):
                    self._cb(outdata, 2205, None, status)
            except Exception as e:
                callbacks_stopped.append(str(type(e).__name__))
            return self
        def __exit__(self, *a):
            return False

    sig = make_signal(duration=0.1, rate=22050)  # 2205 samples
    with mock.patch("bt_diag.sd.OutputStream", FakeStream):
        with mock.patch("bt_diag.threading.Event") as mock_evt:
            mock_evt.return_value.wait = lambda timeout=None: None
            result = play_and_count_xruns(sig, rate=22050)

    assert isinstance(result, int)
    assert "CallbackStop" in callbacks_stopped

def test_cmd_wifi_runs(monkeypatch):
    """cmd_wifi should run without error when mocked."""
    import unittest.mock as mock

    calls = []

    def fake_play(signal, rate=22050):
        calls.append("play")
        return 0  # 0 xruns

    def fake_run(cmd, **kwargs):
        calls.append(cmd[0])
        r = mock.MagicMock()
        r.stdout = "wlan0  wifi  connected\n"
        r.returncode = 0
        return r

    monkeypatch.setattr("bt_diag.play_and_count_xruns", fake_play)
    monkeypatch.setattr("bt_diag.subprocess.run", fake_run)

    from bt_diag import cmd_wifi
    cmd_wifi()  # must not raise
    assert calls.count("play") == 2
