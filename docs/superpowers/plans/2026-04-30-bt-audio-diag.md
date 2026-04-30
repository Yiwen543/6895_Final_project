# Bluetooth Audio Diagnostic Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `bt_diag.py` — a single Python script with 4 independent subcommands that diagnose root causes of choppy Bluetooth A2DP audio on Raspberry Pi 5.

**Architecture:** One file (`bt_diag.py`) with a shared `play_and_count_xruns()` helper and four subcommand functions (`cmd_wifi`, `cmd_quantum`, `cmd_rssi`, `cmd_cpu`). Each subcommand sets up conditions, calls the helper once or twice, prints raw data and a PASS/FAIL verdict, then tears down. `argparse` dispatches to the right subcommand.

**Tech Stack:** Python 3, `sounddevice` (OutputStream callback xrun counting), `numpy` (440 Hz test signal), `subprocess` (nmcli, pw-metadata, hcitool), `/proc/loadavg`.

---

## File Structure

- **Create:** `bt_diag.py` in project root (will be deployed to `~/nova/` on Pi)
- **Create:** `tests/test_bt_diag.py` — unit tests for shared helpers

---

### Task 1: Shared infrastructure — test signal + xrun counter

**Files:**
- Create: `bt_diag.py`
- Create: `tests/test_bt_diag.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/ezslaptop/Projects/6895_Final_project
python -m pytest tests/test_bt_diag.py -v
```

Expected: `ModuleNotFoundError: No module named 'bt_diag'`

- [ ] **Step 3: Create `bt_diag.py` with shared helpers**

```python
#!/usr/bin/env python3
"""
Bluetooth audio diagnostic script for Nova on Raspberry Pi 5.

Usage:
    python3 bt_diag.py wifi      # Test WiFi/BT coexistence
    python3 bt_diag.py quantum   # Test PipeWire quantum size
    python3 bt_diag.py rssi      # Test Bluetooth signal strength
    python3 bt_diag.py cpu       # Test CPU competition
"""

import argparse
import subprocess
import sys
import threading
import time

import numpy as np
import sounddevice as sd

BT_MAC = "68:59:32:F5:D3:BC"
XDG = {"XDG_RUNTIME_DIR": "/run/user/1000"}
import os
ENV = {**os.environ, **XDG}


def make_signal(duration: float = 5.0, rate: int = 22050) -> np.ndarray:
    """440 Hz sine wave, float32, mono."""
    t = np.linspace(0, duration, int(duration * rate), endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)


def play_and_count_xruns(signal: np.ndarray, rate: int = 22050) -> int:
    """Play signal via sounddevice OutputStream; return number of output underflows."""
    xruns = 0
    pos = 0

    def callback(outdata, frames, time_info, status):
        nonlocal xruns, pos
        if status.output_underflow:
            xruns += 1
        remaining = len(signal) - pos
        chunk = min(frames, remaining)
        outdata[:chunk, 0] = signal[pos:pos + chunk]
        if chunk < frames:
            outdata[chunk:] = 0
        pos += chunk

    with sd.OutputStream(
        samplerate=rate,
        channels=1,
        dtype="float32",
        callback=callback,
    ):
        # Wait for playback to finish
        duration = len(signal) / rate
        time.sleep(duration + 0.2)

    return xruns


def raw(label: str, value):
    print(f"[RAW]  {label}: {value}")


def result(passed: bool, reason: str):
    tag = "PASS" if passed else "FAIL"
    print(f"[RESULT] {tag} — {reason}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_bt_diag.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add bt_diag.py tests/test_bt_diag.py
git commit -m "feat: bt_diag shared infrastructure (signal + xrun counter)"
```

---

### Task 2: `wifi` subcommand — WiFi/BT coexistence

**Files:**
- Modify: `bt_diag.py` — add `cmd_wifi()` and wire into `main()`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_bt_diag.py
def test_cmd_wifi_runs(monkeypatch):
    """cmd_wifi should run without error when mocked."""
    import unittest.mock as mock
    from bt_diag import make_signal
    sig = make_signal(0.1)

    calls = []
    def fake_play(signal, rate=22050):
        calls.append("play")
        return 0  # 0 xruns

    def fake_run(cmd, **kwargs):
        calls.append(cmd[0])
        result = mock.MagicMock()
        result.stdout = "wlan0  wifi  connected\n"
        result.returncode = 0
        return result

    monkeypatch.setattr("bt_diag.play_and_count_xruns", fake_play)
    monkeypatch.setattr("bt_diag.subprocess.run", fake_run)

    from bt_diag import cmd_wifi
    cmd_wifi()  # must not raise
    assert calls.count("play") == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_bt_diag.py::test_cmd_wifi_runs -v
```

Expected: `AttributeError: module 'bt_diag' has no attribute 'cmd_wifi'`

- [ ] **Step 3: Implement `cmd_wifi()`**

Add after `result()` in `bt_diag.py`:

```python
def cmd_wifi():
    print("\n=== TEST: WiFi/BT Coexistence ===")
    sig = make_signal()

    # Check WiFi state
    out = subprocess.run(
        ["nmcli", "dev", "status"], capture_output=True, text=True, env=ENV
    )
    wifi_on = "wlan0" in out.stdout and "connected" in out.stdout
    raw("wifi_active", wifi_on)

    # Baseline: WiFi on
    print("[INFO] Playing with WiFi ON...")
    xrun_on = play_and_count_xruns(sig)
    raw("xrun_wifi_on", xrun_on)

    # Disconnect WiFi
    subprocess.run(["sudo", "nmcli", "dev", "disconnect", "wlan0"],
                   capture_output=True, env=ENV)
    time.sleep(1)

    # Test: WiFi off
    print("[INFO] Playing with WiFi OFF...")
    xrun_off = play_and_count_xruns(sig)
    raw("xrun_wifi_off", xrun_off)

    # Restore WiFi
    subprocess.run(["sudo", "nmcli", "dev", "connect", "wlan0"],
                   capture_output=True, env=ENV)

    passed = abs(xrun_on - xrun_off) <= 2
    if passed:
        result(True, "xrun count unchanged with WiFi on/off — WiFi is NOT the cause")
    else:
        result(False, f"xruns dropped {xrun_on}→{xrun_off} when WiFi disconnected — WiFi IS the root cause")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_bt_diag.py::test_cmd_wifi_runs -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bt_diag.py tests/test_bt_diag.py
git commit -m "feat: bt_diag wifi subcommand"
```

---

### Task 3: `quantum` subcommand — PipeWire quantum size

**Files:**
- Modify: `bt_diag.py` — add `cmd_quantum()`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_bt_diag.py
def test_cmd_quantum_runs(monkeypatch):
    import unittest.mock as mock
    calls = []

    def fake_play(signal, rate=22050):
        calls.append("play")
        return 0

    def fake_run(cmd, **kwargs):
        r = mock.MagicMock()
        r.stdout = "key: 'clock.force-quantum' value: '1024'\n"
        r.returncode = 0
        return r

    monkeypatch.setattr("bt_diag.play_and_count_xruns", fake_play)
    monkeypatch.setattr("bt_diag.subprocess.run", fake_run)

    from bt_diag import cmd_quantum
    cmd_quantum()
    assert calls.count("play") == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_bt_diag.py::test_cmd_quantum_runs -v
```

Expected: `AttributeError: module 'bt_diag' has no attribute 'cmd_quantum'`

- [ ] **Step 3: Implement `cmd_quantum()`**

```python
def cmd_quantum():
    print("\n=== TEST: PipeWire Quantum Size ===")
    sig = make_signal()

    # Read current quantum
    out = subprocess.run(
        ["pw-metadata", "-n", "settings"],
        capture_output=True, text=True, env=ENV
    )
    current_quantum = "1024"  # default fallback
    for line in out.stdout.splitlines():
        if "clock.force-quantum" in line or "clock.quantum" in line:
            parts = line.split("'")
            if len(parts) >= 4:
                current_quantum = parts[3]
                break
    raw("quantum_current", current_quantum)

    # Baseline: default quantum
    print(f"[INFO] Playing at quantum={current_quantum}...")
    xrun_default = play_and_count_xruns(sig)
    raw("xrun_default_quantum", xrun_default)

    # Set large quantum
    subprocess.run(
        ["pw-metadata", "-n", "settings", "0", "clock.force-quantum", "2048"],
        capture_output=True, env=ENV
    )
    time.sleep(0.5)

    # Test: large quantum
    print("[INFO] Playing at quantum=2048...")
    xrun_large = play_and_count_xruns(sig)
    raw("xrun_quantum_2048", xrun_large)

    # Restore
    subprocess.run(
        ["pw-metadata", "-n", "settings", "0", "clock.force-quantum", current_quantum],
        capture_output=True, env=ENV
    )

    passed = abs(xrun_default - xrun_large) <= 2
    if passed:
        result(True, "xrun count unchanged with larger quantum — quantum is NOT the cause")
    else:
        result(False, f"xruns dropped {xrun_default}→{xrun_large} with quantum=2048 — quantum IS the root cause")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_bt_diag.py::test_cmd_quantum_runs -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bt_diag.py tests/test_bt_diag.py
git commit -m "feat: bt_diag quantum subcommand"
```

---

### Task 4: `rssi` subcommand — Bluetooth signal strength

**Files:**
- Modify: `bt_diag.py` — add `cmd_rssi()`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_bt_diag.py
def test_cmd_rssi_runs(monkeypatch):
    import unittest.mock as mock
    call_count = [0]

    def fake_play(signal, rate=22050):
        return 0

    def fake_run(cmd, **kwargs):
        r = mock.MagicMock()
        r.stdout = "RSSI return value: -55\n"
        r.returncode = 0
        call_count[0] += 1
        return r

    monkeypatch.setattr("bt_diag.play_and_count_xruns", fake_play)
    monkeypatch.setattr("bt_diag.subprocess.run", fake_run)
    monkeypatch.setattr("bt_diag.time.sleep", lambda x: None)

    from bt_diag import cmd_rssi
    cmd_rssi()  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_bt_diag.py::test_cmd_rssi_runs -v
```

Expected: `AttributeError: module 'bt_diag' has no attribute 'cmd_rssi'`

- [ ] **Step 3: Implement `cmd_rssi()`**

```python
def cmd_rssi():
    print("\n=== TEST: Bluetooth Signal Strength ===")
    sig = make_signal()
    samples = []
    failures = 0

    def poll_rssi():
        nonlocal failures
        for _ in range(10):
            out = subprocess.run(
                ["hcitool", "rssi", BT_MAC],
                capture_output=True, text=True, env=ENV
            )
            line = out.stdout.strip()
            # "RSSI return value: -55"
            try:
                val = int(line.split()[-1])
                samples.append(val)
                raw("rssi_sample", val)
            except (ValueError, IndexError):
                failures += 1
                raw("rssi_sample", f"FAIL ({line!r})")
            time.sleep(0.5)

    poller = threading.Thread(target=poll_rssi, daemon=True)
    poller.start()
    play_and_count_xruns(sig)
    poller.join()

    if samples:
        avg = sum(samples) / len(samples)
        raw("rssi_min", min(samples))
        raw("rssi_max", max(samples))
        raw("rssi_avg", f"{avg:.1f}")
        raw("poll_failures", failures)

        passed = avg > -70 and failures < 3
        if passed:
            result(True, f"avg RSSI {avg:.1f} dBm > -70 — signal is NOT the cause")
        else:
            reason = []
            if avg <= -70:
                reason.append(f"avg RSSI {avg:.1f} dBm ≤ -70")
            if failures >= 3:
                reason.append(f"{failures} poll failures")
            result(False, "; ".join(reason) + " — weak signal / interference IS likely a cause")
    else:
        result(False, f"all {failures} RSSI polls failed — cannot measure signal")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_bt_diag.py::test_cmd_rssi_runs -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bt_diag.py tests/test_bt_diag.py
git commit -m "feat: bt_diag rssi subcommand"
```

---

### Task 5: `cpu` subcommand — CPU competition

**Files:**
- Modify: `bt_diag.py` — add `cmd_cpu()`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_bt_diag.py
def test_cmd_cpu_runs(monkeypatch):
    calls = []

    def fake_play(signal, rate=22050):
        calls.append("play")
        return 0

    monkeypatch.setattr("bt_diag.play_and_count_xruns", fake_play)
    monkeypatch.setattr("bt_diag.time.sleep", lambda x: None)

    from bt_diag import cmd_cpu
    cmd_cpu()
    assert calls.count("play") == 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_bt_diag.py::test_cmd_cpu_runs -v
```

Expected: `AttributeError: module 'bt_diag' has no attribute 'cmd_cpu'`

- [ ] **Step 3: Implement `cmd_cpu()`**

```python
def _read_load() -> str:
    with open("/proc/loadavg") as f:
        return f.read().split()[0]


def cmd_cpu():
    print("\n=== TEST: CPU Competition ===")
    sig = make_signal()

    # Baseline: idle
    load_before = _read_load()
    raw("load_avg_before", load_before)
    print("[INFO] Playing at idle...")
    xrun_idle = play_and_count_xruns(sig)
    raw("xrun_idle", xrun_idle)

    # Start CPU load (3 busy threads simulating LLM)
    stop_event = threading.Event()
    def busy():
        while not stop_event.is_set():
            pass

    threads = [threading.Thread(target=busy, daemon=True) for _ in range(3)]
    for t in threads:
        t.start()
    time.sleep(0.5)  # let load build up

    load_during = _read_load()
    raw("load_avg_during", load_during)
    print("[INFO] Playing under CPU load (3 busy threads)...")
    xrun_loaded = play_and_count_xruns(sig)
    raw("xrun_loaded", xrun_loaded)

    stop_event.set()
    for t in threads:
        t.join(timeout=1)

    passed = xrun_loaded <= xrun_idle + 2
    if passed:
        result(True, f"xruns {xrun_idle}→{xrun_loaded} under load — CPU is NOT the cause")
    else:
        result(False, f"xruns jumped {xrun_idle}→{xrun_loaded} under load — CPU starvation IS the root cause")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_bt_diag.py::test_cmd_cpu_runs -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add bt_diag.py tests/test_bt_diag.py
git commit -m "feat: bt_diag cpu subcommand"
```

---

### Task 6: Wire `main()` + final integration test

**Files:**
- Modify: `bt_diag.py` — add `main()` with argparse

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_bt_diag.py
def test_main_dispatches_correctly(monkeypatch):
    called = []
    monkeypatch.setattr("bt_diag.cmd_wifi",    lambda: called.append("wifi"))
    monkeypatch.setattr("bt_diag.cmd_quantum", lambda: called.append("quantum"))
    monkeypatch.setattr("bt_diag.cmd_rssi",    lambda: called.append("rssi"))
    monkeypatch.setattr("bt_diag.cmd_cpu",     lambda: called.append("cpu"))

    import sys
    from bt_diag import main

    for cmd in ["wifi", "quantum", "rssi", "cpu"]:
        called.clear()
        monkeypatch.setattr(sys, "argv", ["bt_diag.py", cmd])
        main()
        assert called == [cmd]

def test_main_unknown_command_exits(monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv", ["bt_diag.py", "badcmd"])
    from bt_diag import main
    with pytest.raises(SystemExit):
        main()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_bt_diag.py::test_main_dispatches_correctly tests/test_bt_diag.py::test_main_unknown_command_exits -v
```

Expected: `AttributeError: module 'bt_diag' has no attribute 'main'`

- [ ] **Step 3: Implement `main()`**

Append to `bt_diag.py`:

```python
def main():
    parser = argparse.ArgumentParser(
        description="Nova Bluetooth audio diagnostic tool"
    )
    parser.add_argument(
        "test",
        choices=["wifi", "quantum", "rssi", "cpu"],
        help="Which diagnostic test to run",
    )
    args = parser.parse_args()
    {"wifi": cmd_wifi, "quantum": cmd_quantum,
     "rssi": cmd_rssi, "cpu": cmd_cpu}[args.test]()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all tests**

```bash
python -m pytest tests/test_bt_diag.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Quick smoke test on Mac (no audio hardware needed)**

```bash
python3 bt_diag.py --help
```

Expected output:
```
usage: bt_diag.py [-h] {wifi,quantum,rssi,cpu}
...
```

- [ ] **Step 6: Commit**

```bash
git add bt_diag.py tests/test_bt_diag.py
git commit -m "feat: bt_diag main entrypoint + all tests passing"
```

---

### Task 7: Deploy to Pi and smoke test

**Files:**
- No code changes — deploy and verify on target hardware

- [ ] **Step 1: Copy script to Pi**

```bash
rsync -avz bt_diag.py tl3461@192.168.100.1:~/nova/bt_diag.py
```

- [ ] **Step 2: Verify help on Pi**

```bash
ssh tl3461@192.168.100.1 "python3 nova/bt_diag.py --help"
```

Expected:
```
usage: bt_diag.py [-h] {wifi,quantum,rssi,cpu}
```

- [ ] **Step 3: Smoke test `rssi` (no side effects)**

```bash
ssh tl3461@192.168.100.1 "XDG_RUNTIME_DIR=/run/user/1000 python3 nova/bt_diag.py rssi"
```

Expected: prints `[RAW] rssi_sample: ...` lines, ends with `[RESULT] PASS` or `[RESULT] FAIL`.

- [ ] **Step 4: Commit**

```bash
git add bt_diag.py
git commit -m "chore: verify bt_diag deployed and smoke-tested on Pi"
```
