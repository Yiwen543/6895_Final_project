# Nova Pi 5 Deployment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy Nova Smart Home Assistant to Raspberry Pi 5 (8GB) as a systemd service, with Grove RGB LED + 28BYJ-48 stepper motor control via GPIO, Piper TTS, and four latency optimisations (rule-based fast path, parallel GPIO+TTS, max_new_tokens=96, Piper).

**Architecture:** `nova.py` (pipeline) delegates hardware to `gpio_executor.py` (LED + stepper via lgpio). A `rule_based.py` module handles unambiguous commands without LLM. `deploy.sh` runs from the dev machine: rsync → pip install → audio config → systemd registration.

**Tech Stack:** Python 3.11, lgpio (Pi 5 GPIO), piper-tts, faster-whisper, transformers (TinyLlama), sounddevice, PipeWire, systemd, Bluetooth

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `rule_based.py` | Create | Regex fast path for direct commands |
| `gpio_executor.py` | Create | P9813 LED + 28BYJ-48 stepper via lgpio |
| `nova.py` | Create | Main pipeline (refactored from Nova_4_16.ipynb) |
| `requirements_pi.txt` | Create | Pi-specific Python dependencies |
| `nova.service` | Create | systemd service for Nova |
| `bt-speaker.service` | Create | systemd Bluetooth auto-connect |
| `deploy.sh` | Create | One-command deploy from dev machine |
| `tests/__init__.py` | Create | Test package marker |
| `tests/conftest.py` | Create | Mock lgpio for dev-machine tests |
| `tests/test_gpio_executor.py` | Create | Unit tests for GPIOExecutor |
| `tests/test_rule_based.py` | Create | Unit tests for try_rule_based() |
| `Nova_4_16.ipynb` | Read only | Source reference for nova.py — not modified |

---

### Task 1: Project scaffolding

**Files:**
- Create: `requirements_pi.txt`
- Create: `.gitignore` (append)
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create requirements_pi.txt**

```
faster-whisper
transformers
accelerate
sentencepiece
torch --index-url https://download.pytorch.org/whl/cpu
sounddevice
soundfile
lgpio
rpi-lgpio
piper-tts
numpy
```

- [ ] **Step 2: Gitignore the Piper voice model**

```bash
echo "voices/*.onnx" >> .gitignore
```

- [ ] **Step 3: Create tests/__init__.py**

```bash
mkdir -p tests && touch tests/__init__.py
```

- [ ] **Step 4: Create tests/conftest.py**

Injects a mock lgpio into `sys.modules` before any test imports `gpio_executor`, so tests run on macOS/Linux dev machines without Pi hardware.

```python
import sys
from unittest.mock import MagicMock

lgpio_mock = MagicMock()
lgpio_mock.gpiochip_open.return_value = 42
sys.modules['lgpio'] = lgpio_mock
```

- [ ] **Step 5: Verify pytest is available**

```bash
python3 -m pytest --version
```

Expected: `pytest 7.x.x` or higher. If missing: `pip3 install pytest`.

- [ ] **Step 6: Commit**

```bash
git add requirements_pi.txt .gitignore tests/__init__.py tests/conftest.py
git commit -m "feat: project scaffolding for Pi 5 deployment"
```

---

### Task 2: Rule-based fast path tests

**Files:**
- Create: `tests/test_rule_based.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
python3 -m pytest tests/test_rule_based.py -v
```

Expected: `ImportError: No module named 'rule_based'`

- [ ] **Step 3: Create rule_based.py**

```python
import re
from typing import Optional, Dict, Any


def try_rule_based(text: str) -> Optional[Dict[str, Any]]:
    t = text.lower()

    m = re.search(r"(?:ac|air.?con).*?(\d+)\s*degree|(\d+)\s*degree.*?(?:ac|air.?con)", t)
    if m:
        val = int(m.group(1) or m.group(2))
        if 16 <= val <= 30:
            return {"type": "direct_command", "device": "ac", "action": "set_temperature", "value": val}

    m = re.search(r"brightness.*?(\d+)|(\d+).*?brightness", t)
    if m:
        val = int(m.group(1) or m.group(2))
        if 0 <= val <= 100:
            return {"type": "direct_command", "device": "light", "action": "set_brightness", "value": val}

    m = re.search(r"(curtain|window).*?(\d+)\s*(?:percent|%)|(\d+)\s*(?:percent|%).*?(curtain|window)", t)
    if m:
        device = m.group(1) or m.group(4)
        val = int(m.group(2) or m.group(3))
        if 0 <= val <= 100:
            return {"type": "direct_command", "device": device, "action": "set_position", "value": val}

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
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/test_rule_based.py -v
```

Expected: all 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_rule_based.py rule_based.py
git commit -m "feat: rule-based fast path with tests"
```

---

### Task 3: GPIOExecutor tests

**Files:**
- Create: `tests/test_gpio_executor.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run to confirm failure**

```bash
python3 -m pytest tests/test_gpio_executor.py -v
```

Expected: `ImportError: No module named 'gpio_executor'`

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/test_gpio_executor.py
git commit -m "test: add failing tests for GPIOExecutor"
```

---

### Task 4: GPIOExecutor implementation

**Files:**
- Create: `gpio_executor.py`

- [ ] **Step 1: Create gpio_executor.py**

```python
import lgpio
import time
import threading
import colorsys

# Signal pins (Pi 5 BCM numbering, verified against 40-pin header)
DATA_PIN = 17   # Grove RGB LED DATA  → Pin 11
CLK_PIN  = 27   # Grove RGB LED CLK   → Pin 13
MOTOR_PINS = [5, 6, 13, 19]  # ULN2003 IN1-IN4 → Pins 29,31,33,35

# ⚠️ VCC and GND for LED and motor board come from an external PSU.
# The external PSU GND must share a common ground with a Pi GND pin
# (e.g. Pin 6 or Pin 9) for GPIO signal levels to be valid.

CURTAIN_TOTAL_STEPS = 2048  # half-revolution; calibrate after first wiring test
STEP_DELAY = 0.002          # 2 ms between half-steps


class GPIOExecutor:
    HALF_STEP_SEQ = [
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 1],
    ]

    def __init__(self):
        self._h = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_output(self._h, DATA_PIN)
        lgpio.gpio_claim_output(self._h, CLK_PIN)
        for pin in MOTOR_PINS:
            lgpio.gpio_claim_output(self._h, pin)
        self._step_index = 0
        self._curtain_pos = 0
        self._rgb_stop = threading.Event()
        self._rgb_thread = None
        self._set_color(0, 0, 0)
        self._release_motor()

    # ── P9813 LED ──────────────────────────────────────────────────────────────

    def _send_bit(self, bit: int) -> None:
        lgpio.gpio_write(self._h, DATA_PIN, bit)
        lgpio.gpio_write(self._h, CLK_PIN, 1)
        lgpio.gpio_write(self._h, CLK_PIN, 0)

    def _send_byte(self, byte: int) -> None:
        for i in range(7, -1, -1):
            self._send_bit((byte >> i) & 1)

    def _set_color(self, r: int, g: int, b: int) -> None:
        for _ in range(32):          # start frame
            self._send_bit(0)
        prefix = 0xC0
        prefix |= ((~b) & 0xC0) >> 2
        prefix |= ((~g) & 0xC0) >> 4
        prefix |= ((~r) & 0xC0) >> 6
        self._send_byte(prefix & 0xFF)
        self._send_byte(b & 0xFF)
        self._send_byte(g & 0xFF)
        self._send_byte(r & 0xFF)
        for _ in range(32):          # end frame
            self._send_bit(0)

    def _start_rgb_cycle(self) -> None:
        self._stop_rgb_cycle()
        self._rgb_stop.clear()
        self._rgb_thread = threading.Thread(target=self._rgb_cycle_loop, daemon=True)
        self._rgb_thread.start()

    def _stop_rgb_cycle(self) -> None:
        if self._rgb_thread and self._rgb_thread.is_alive():
            self._rgb_stop.set()
            self._rgb_thread.join(timeout=0.5)

    def _rgb_cycle_loop(self) -> None:
        hue = 0.0
        while not self._rgb_stop.is_set():
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            self._set_color(int(r * 255), int(g * 255), int(b * 255))
            hue = (hue + 0.01) % 1.0
            time.sleep(0.1)

    # ── Stepper motor ──────────────────────────────────────────────────────────

    def _do_step(self, direction: int) -> None:
        self._step_index = (self._step_index + direction) % 8
        for i, pin in enumerate(MOTOR_PINS):
            lgpio.gpio_write(self._h, pin, self.HALF_STEP_SEQ[self._step_index][i])
        time.sleep(STEP_DELAY)

    def _release_motor(self) -> None:
        for pin in MOTOR_PINS:
            lgpio.gpio_write(self._h, pin, 0)

    def _move_to_position(self, target_pct: int) -> None:
        steps = int((target_pct - self._curtain_pos) / 100 * CURTAIN_TOTAL_STEPS)
        direction = 1 if steps >= 0 else -1
        for _ in range(abs(steps)):
            self._do_step(direction)
        self._curtain_pos = target_pct
        self._release_motor()

    # ── Public interface ───────────────────────────────────────────────────────

    def execute(self, cmd: dict) -> str:
        device = cmd.get('device')
        action = cmd.get('action')
        value  = cmd.get('value')

        if device == 'light':
            if action == 'turn_on':
                self._stop_rgb_cycle()
                self._set_color(255, 255, 255)
                return 'LIGHT -> ON'
            if action == 'turn_off':
                self._stop_rgb_cycle()
                self._set_color(0, 0, 0)
                return 'LIGHT -> OFF'
            if action == 'set_brightness':
                self._stop_rgb_cycle()
                v = int(2.55 * value)
                self._set_color(v, v, v)
                return f'LIGHT -> BRIGHTNESS {value}%'
            if action == 'rgb_cycle':
                self._start_rgb_cycle()
                return 'LIGHT -> RGB CYCLE'

        if device == 'curtain':
            if action == 'open':
                self._move_to_position(100)
                return 'CURTAIN -> OPEN'
            if action == 'close':
                self._move_to_position(0)
                return 'CURTAIN -> CLOSE'
            if action == 'set_position':
                self._move_to_position(value)
                return f'CURTAIN -> POSITION {value}%'

        return f'[STUB] {device} {action}'

    def cleanup(self) -> None:
        self._stop_rgb_cycle()
        self._set_color(0, 0, 0)
        self._release_motor()
        lgpio.gpiochip_close(self._h)
```

- [ ] **Step 2: Run tests**

```bash
python3 -m pytest tests/test_gpio_executor.py -v
```

Expected: all 10 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add gpio_executor.py
git commit -m "feat: GPIOExecutor — P9813 RGB LED and 28BYJ-48 stepper via lgpio"
```

---

### Task 5: nova.py — read notebook and build core structure

**Files:**
- Read: `Nova_4_16.ipynb`
- Create: `nova.py`

- [ ] **Step 1: Dump the full notebook source for reference**

```bash
python3 -c "
import json
with open('Nova_4_16.ipynb') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells']):
    print(f'=== Cell {i} ===')
    print(''.join(cell['source']))
    print()
" 2>/dev/null | less
```

Note: (a) the exact `max_new_tokens` value in the `model.generate()` call, (b) the complete `handle_transcribed_text` function body, (c) the complete main audio loop, (d) all helper functions in Sections 5–9.

- [ ] **Step 2: Create nova.py — header and imports**

```python
#!/usr/bin/env python3
"""Nova Smart Home Assistant — Raspberry Pi 5 service entry point."""

import os
import io
import re
import gc
import json
import time
import wave
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

from typing import Dict, Any, Optional, Tuple
from collections import deque
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from piper.voice import PiperVoice

from rule_based import try_rule_based
from gpio_executor import GPIOExecutor

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SAMPLE_RATE = 16000
CHANNELS = 1
PIPER_MODEL_PATH = os.path.join(_SCRIPT_DIR, "voices", "en_US-lessac-medium.onnx")
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

- [ ] **Step 3: Copy Sections 3 and 4 from notebook verbatim**

From the notebook dump (Step 1), copy into nova.py:
- Section 3: `COMMAND_SCHEMA`, `INVALID_COMMAND`, `validate_command`, `execute_command`, `build_execution_reply`
- Section 4: `ASSISTANT_NAME`, `ASSISTANT_NAME_VARIANTS`, `contains_assistant_name`

Do not modify any of these.

- [ ] **Step 4: Copy Section 5 from notebook — change max_new_tokens to 96**

Copy `UNIFIED_SYSTEM_PROMPT`, `FOLLOWUP_RESOLUTION_SYSTEM_PROMPT`, and all LLM helper functions (`extract_first_json_object`, `normalize_unified_result`, `llm_generate_json`, `llm_parse_unified`, `llm_resolve_followup`) verbatim from the notebook.

Then find the `model.generate()` call (inside `llm_generate_json` or equivalent) and change its `max_new_tokens` parameter to `96`:

```python
outputs = model.generate(
    input_ids,
    max_new_tokens=96,      # JSON output fits in 80 tokens; cap to avoid waste
    do_sample=False,
    temperature=1.0,
    pad_token_id=tokenizer.eos_token_id,
)
```

- [ ] **Step 5: Copy Section 6 STT init from notebook verbatim**

```python
print("Loading STT model...")
stt_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
print("STT ready.")
```

- [ ] **Step 6: Commit**

```bash
git add nova.py
git commit -m "feat: nova.py core — schema, LLM, STT, max_new_tokens=96"
```

---

### Task 6: nova.py — Piper TTS

**Files:**
- Modify: `nova.py`

- [ ] **Step 1: Add Piper TTS init and speak() function**

Add after the STT init block in nova.py:

```python
# ── TTS ────────────────────────────────────────────────────────────────────────
print("Loading Piper TTS...")
_piper_voice = PiperVoice.load(PIPER_MODEL_PATH)
print("TTS ready.")


def speak(text: str) -> None:
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, 'wb') as wf:
        _piper_voice.synthesize(text, wf)
    wav_buf.seek(0)
    with wave.open(wav_buf, 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        rate = wf.getframerate()
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    sd.play(audio, rate)
    sd.wait()
```

- [ ] **Step 2: Download Piper voice model locally for testing**

```bash
mkdir -p voices
python3 -c "
import urllib.request, os
url = 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx'
path = 'voices/en_US-lessac-medium.onnx'
if not os.path.exists(path):
    print('Downloading...')
    urllib.request.urlretrieve(url, path)
    print('Done.')
else:
    print('Already present.')
"
```

Expected: `Done.` or `Already present.`

- [ ] **Step 3: Smoke-test TTS**

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from nova import speak
speak('Hello, I am Nova.')
"
```

Expected: voice plays through default audio output. If silent: confirm PipeWire is running (`systemctl --user status pipewire`) and a speaker is connected.

- [ ] **Step 4: Commit**

```bash
git add nova.py
git commit -m "feat: Piper TTS in nova.py, replaces pyttsx3"
```

---

### Task 7: nova.py — GPIO, parallel execution, handler, main loop

**Files:**
- Modify: `nova.py`

- [ ] **Step 1: Add GPIOExecutor init**

Add after the TTS block:

```python
# ── GPIO ───────────────────────────────────────────────────────────────────────
print("Initialising GPIO...")
gpio = GPIOExecutor()
print("GPIO ready.")
```

- [ ] **Step 2: Add parallel execution helper**

```python
def _execute_and_speak(cmd: dict, reply_text: str) -> None:
    gpio_thread = threading.Thread(target=gpio.execute, args=(cmd,), daemon=True)
    tts_thread  = threading.Thread(target=speak, args=(reply_text,), daemon=True)
    gpio_thread.start()
    tts_thread.start()
    gpio_thread.join()
    tts_thread.join()
```

- [ ] **Step 3: Copy handle_transcribed_text from notebook, apply three patches**

Copy the complete `handle_transcribed_text` function from Section 8 of the notebook, then apply exactly these three edits:

**Patch A — rule-based fast path** (add before the `llm_parse_unified` call):
```python
fast_result = try_rule_based(text)
if fast_result is not None:
    semantic = fast_result
    llm_latency_ms = 0.0
    raw_output = "(rule-based)"
else:
    semantic, raw_output, llm_latency_ms = llm_parse_unified(text, verbose=verbose)
```

**Patch B — parallel GPIO + TTS** (replace the `direct_command` branch's speak call):
```python
# Replace:  execute_command(cmd); speak(build_execution_reply(cmd))
# With:
hw_result = execute_command(cmd)
reply     = build_execution_reply(cmd)
_execute_and_speak(cmd, reply)
```

**Patch C — no changes** to `needs_clarification`, `general_qa`, and `invalid` branches; they call `speak()` directly which is correct.

- [ ] **Step 4: Copy transcribe_audio_numpy from notebook Section 7 verbatim**

The function is already in the notebook (uses `io.BytesIO`, no disk I/O). Copy it unchanged.

- [ ] **Step 5: Copy the main audio loop from notebook Section 9 verbatim**

Copy `collect_one_utterance_from_stream` and the main loop function unchanged.

- [ ] **Step 6: Add entry point with cleanup**

```python
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        gpio.cleanup()
        print("Nova shut down.")
```

- [ ] **Step 7: Commit**

```bash
git add nova.py
git commit -m "feat: GPIO integration, rule-based fast path, parallel execution in nova.py"
```

---

### Task 8: Systemd service files

**Files:**
- Create: `bt-speaker.service`
- Create: `nova.service`

- [ ] **Step 1: Create bt-speaker.service**

The string `PLACEHOLDER_BT_MAC` is replaced by `deploy.sh` at install time using `sed`.

```ini
[Unit]
Description=Auto-connect Bluetooth Speaker
After=bluetooth.service
Wants=bluetooth.service

[Service]
Type=oneshot
ExecStart=/usr/bin/bluetoothctl connect PLACEHOLDER_BT_MAC
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 2: Create nova.service**

```ini
[Unit]
Description=Nova Smart Home Assistant
After=sound.target bt-speaker.service
Wants=bt-speaker.service

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/nova
ExecStart=/usr/bin/python3 nova.py
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 3: Commit**

```bash
git add bt-speaker.service nova.service
git commit -m "feat: systemd service files for Nova and Bluetooth speaker"
```

---

### Task 9: deploy.sh

**Files:**
- Create: `deploy.sh`

- [ ] **Step 1: Create deploy.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

PI_HOST="${1:?Usage: ./deploy.sh pi@<IP> <BT_MAC>}"
BT_MAC="${2:?Usage: ./deploy.sh pi@<IP> <BT_MAC>}"

echo "==> Syncing files to ${PI_HOST}:~/nova/"
rsync -avz --exclude '__pycache__' --exclude '.git' --exclude 'tests/' \
    ./ "${PI_HOST}:~/nova/"

echo "==> Installing system dependencies"
ssh "$PI_HOST" "sudo apt-get install -y --no-install-recommends \
    python3-pip libportaudio2 libasound2-dev bluetooth bluez"

echo "==> Installing Python dependencies"
ssh "$PI_HOST" "pip3 install --break-system-packages -r ~/nova/requirements_pi.txt"

echo "==> Downloading Piper voice model (if needed)"
ssh "$PI_HOST" "mkdir -p ~/nova/voices && \
    [ -f ~/nova/voices/en_US-lessac-medium.onnx ] || \
    wget -q --show-progress \
      -O ~/nova/voices/en_US-lessac-medium.onnx \
      'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx'"

echo "==> Setting SunFounder USB microphone as default PipeWire input"
ssh "$PI_HOST" "
    USB_SRC=\$(pactl list sources short 2>/dev/null | grep -i usb | awk '{print \$2}' | head -1)
    if [ -n \"\$USB_SRC\" ]; then
        pactl set-default-source \"\$USB_SRC\"
        echo \"Default input set to: \$USB_SRC\"
    else
        echo 'WARNING: USB microphone not detected. Plug it in and re-run deploy.sh.'
    fi
"

echo "==> Writing bt-speaker.service (MAC: ${BT_MAC})"
ssh "$PI_HOST" "sed 's/PLACEHOLDER_BT_MAC/${BT_MAC}/g' ~/nova/bt-speaker.service | \
    sudo tee /etc/systemd/system/bt-speaker.service > /dev/null"

echo "==> Installing nova.service"
ssh "$PI_HOST" "sudo cp ~/nova/nova.service /etc/systemd/system/nova.service"

echo "==> Enabling services"
ssh "$PI_HOST" "sudo systemctl daemon-reload && sudo systemctl enable bt-speaker nova"

echo ""
echo "==> Deploy complete."
echo ""
echo "If this is your first deploy, pair the Bluetooth speaker first:"
echo "  ssh ${PI_HOST}"
echo "  bluetoothctl"
echo "    power on"
echo "    agent on"
echo "    scan on          # wait for speaker MAC to appear"
echo "    pair   ${BT_MAC}"
echo "    trust  ${BT_MAC}"
echo "    connect ${BT_MAC}"
echo "    exit"
echo ""
read -rp "Start Nova now? [y/N] " answer
if [[ "${answer,,}" == "y" ]]; then
    ssh "$PI_HOST" "sudo systemctl start bt-speaker && sleep 2 && sudo systemctl start nova"
    echo "Nova started. Follow logs: ssh ${PI_HOST} 'journalctl -u nova -f'"
fi
```

- [ ] **Step 2: Make executable**

```bash
chmod +x deploy.sh
```

- [ ] **Step 3: Syntax check**

```bash
bash -n deploy.sh && echo "Syntax OK"
```

Expected: `Syntax OK`

- [ ] **Step 4: Commit**

```bash
git add deploy.sh
git commit -m "feat: one-command deploy.sh for Pi 5"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by |
|-----------------|-----------|
| File structure (nova.py, gpio_executor.py, deploy.sh, services) | Tasks 4–9 |
| Pi 5 uses lgpio, not RPi.GPIO | Task 4 (gpio_executor.py constants) |
| GPIO 17/27 for RGB LED (DATA/CLK) | Task 4 |
| GPIO 5/6/13/19 for stepper | Task 4 |
| External PSU + common ground note | Task 4 (comment in gpio_executor.py) |
| P9813 full action mapping (turn_on/off, brightness, rgb_cycle) | Task 4 |
| Stepper half-step, position tracking, curtain open/close/set_position | Task 4 |
| SunFounder USB mic → default PipeWire input | Task 9 (deploy.sh) |
| Bluetooth auto-connect via bt-speaker.service | Tasks 8–9 |
| Piper TTS replaces pyttsx3 | Tasks 6, 9 |
| Rule-based fast path (~70% commands skip LLM) | Tasks 2, 7 |
| GPIO + TTS parallel execution | Task 7 |
| max_new_tokens=96 | Task 5 |
| deploy.sh: rsync + pip + Piper model + USB mic + systemd | Task 9 |
| nova.service After=bt-speaker.service | Task 8 |
| Restart=on-failure, WorkingDirectory=/home/pi/nova | Task 8 |

**Placeholder scan:** No TBD, no TODO, no "similar to Task N". All code blocks complete.

**Type consistency:**
- `GPIOExecutor.execute(cmd: dict) -> str` — defined Task 4, used Task 7 ✅
- `try_rule_based(text: str) -> Optional[Dict]` — defined Task 2, imported Task 7 ✅
- `speak(text: str) -> None` — defined Task 6, used Task 7 ✅
- `_execute_and_speak(cmd, reply_text)` — defined and used Task 7 ✅
- `gpio.cleanup()` — defined Task 4, called Task 7 ✅
