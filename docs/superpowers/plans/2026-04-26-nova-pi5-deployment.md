# Nova Pi 5 Deployment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy Nova Smart Home Assistant to Raspberry Pi 5 (8GB) as a systemd service, with Grove RGB LED + 28BYJ-48 stepper motor control via GPIO, Piper TTS, and four latency optimisations (rule-based fast path, parallel GPIO+TTS, max_new_tokens=96, Piper).

**Architecture:** The project is already modular: `config.py` (all config), `schema.py` (device schema/validation), `llm_parser.py` (LLMParser class, Qwen2.5-1.5B), `agent.py` (NovaAgent pipeline), `audio.py` (STT/TTS/AudioListener), `memory.py` (MemoryManager). `nova.py` is a thin orchestration entry point that initialises all components and runs `AudioListener`. `gpio_executor.py` is injected into `NovaAgent`. `deploy.sh` runs from the dev machine.

**Tech Stack:** Python 3.11, Qwen2.5-1.5B-Instruct, lgpio (Pi 5 GPIO), piper-tts, faster-whisper, transformers, sentence-transformers, sounddevice, PipeWire, systemd, Bluetooth

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `rule_based.py` | ✅ Done | Regex fast path for direct commands |
| `tests/test_rule_based.py` | ✅ Done | Unit tests for try_rule_based() |
| `requirements_pi.txt` | ✅ Done + update | Pi-specific Python dependencies |
| `tests/__init__.py` | ✅ Done | Test package marker |
| `tests/conftest.py` | ✅ Done | Mock lgpio for dev-machine tests |
| `gpio_executor.py` | Create | P9813 LED + 28BYJ-48 stepper via lgpio |
| `tests/test_gpio_executor.py` | Create | Unit tests for GPIOExecutor |
| `config.py` | Modify | Change LLM_MAX_NEW_TOKENS 160→96, add PIPER_MODEL_PATH |
| `audio.py` | Modify | Replace TTSEngine pyttsx3 → Piper |
| `agent.py` | Modify | Add gpio param, rule-based fast path, parallel GPIO+TTS |
| `nova.py` | Create | Thin orchestration entry point |
| `nova.service` | Create | systemd service for Nova |
| `bt-speaker.service` | Create | systemd Bluetooth auto-connect |
| `deploy.sh` | Create | One-command deploy from dev machine |

---

### Task 1: Project scaffolding ✅ COMPLETE

Already done. `requirements_pi.txt`, `.gitignore`, `tests/__init__.py`, `tests/conftest.py` all created and committed.

---

### Task 2: Rule-based fast path ✅ COMPLETE

Already done. `rule_based.py` and `tests/test_rule_based.py` created, 11 tests passing, committed.

---

### Task 3+4: GPIOExecutor tests + implementation

**Files:**
- Create: `tests/test_gpio_executor.py`
- Create: `gpio_executor.py`

- [ ] **Step 1: Write failing tests — create tests/test_gpio_executor.py**

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
cd /Users/ezslaptop/Projects/6895_Final_project && python3 -m pytest tests/test_gpio_executor.py -v 2>&1 | head -10
```

Expected: `ImportError: No module named 'gpio_executor'`

- [ ] **Step 3: Commit failing tests**

```bash
cd /Users/ezslaptop/Projects/6895_Final_project && git add tests/test_gpio_executor.py && git commit -m "test: add failing tests for GPIOExecutor"
```

- [ ] **Step 4: Create gpio_executor.py**

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
        for _ in range(32):
            self._send_bit(0)
        prefix = 0xC0
        prefix |= ((~b) & 0xC0) >> 2
        prefix |= ((~g) & 0xC0) >> 4
        prefix |= ((~r) & 0xC0) >> 6
        self._send_byte(prefix & 0xFF)
        self._send_byte(b & 0xFF)
        self._send_byte(g & 0xFF)
        self._send_byte(r & 0xFF)
        for _ in range(32):
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

- [ ] **Step 5: Run tests — must all pass**

```bash
cd /Users/ezslaptop/Projects/6895_Final_project && python3 -m pytest tests/test_gpio_executor.py -v
```

Expected: 10 tests PASSED.

- [ ] **Step 6: Commit**

```bash
cd /Users/ezslaptop/Projects/6895_Final_project && git add gpio_executor.py && git commit -m "feat: GPIOExecutor — P9813 RGB LED and 28BYJ-48 stepper via lgpio"
```

---

### Task 5: config.py update + nova.py entry point

**Files:**
- Modify: `config.py` — change `LLM_MAX_NEW_TOKENS`, add `PIPER_MODEL_PATH`
- Modify: `requirements_pi.txt` — add `sentence-transformers`
- Create: `nova.py` — thin orchestration entry point

**Context:** The project is already modular. `nova.py` only needs to import and wire together existing components: `LLMParser`, `MemoryManager`, `STTModel`, `TTSEngine` (Piper, after Task 6), `GPIOExecutor`, `NovaAgent`, and `AudioListener`.

- [ ] **Step 1: Update config.py — change LLM_MAX_NEW_TOKENS and add PIPER_MODEL_PATH**

In `config.py`, change:
```python
LLM_MAX_NEW_TOKENS = 160
```
to:
```python
LLM_MAX_NEW_TOKENS = 96
```

And add after the `# ── TTS` block:
```python
PIPER_MODEL_PATH = "voices/en_US-lessac-medium.onnx"
```

- [ ] **Step 2: Add sentence-transformers to requirements_pi.txt**

Append to `requirements_pi.txt`:
```
sentence-transformers
```

- [ ] **Step 3: Create nova.py**

```python
#!/usr/bin/env python3
"""Nova Smart Home Assistant — Raspberry Pi 5 entry point."""

import os
import sys

from sentence_transformers import SentenceTransformer

from audio import AudioListener, STTModel, TTSEngine
from llm_parser import LLMParser
from memory import MemoryManager
from agent import NovaAgent
from gpio_executor import GPIOExecutor
from config import EMBED_MODEL_NAME


def main() -> None:
    print("=== Nova starting up ===")

    stt    = STTModel()
    tts    = TTSEngine()
    llm    = LLMParser()
    embed  = SentenceTransformer(EMBED_MODEL_NAME)
    memory = MemoryManager(embed)
    gpio   = GPIOExecutor()

    nova = NovaAgent(llm=llm, memory=memory, speak=tts.speak, gpio=gpio)

    listener = AudioListener(agent=nova, stt=stt)

    print("=== Nova ready. Listening... ===")
    try:
        listener.continuous_loop()
    except KeyboardInterrupt:
        pass
    finally:
        gpio.cleanup()
        print("Nova shut down.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify nova.py imports resolve (no runtime errors on import)**

```bash
cd /Users/ezslaptop/Projects/6895_Final_project && python3 -c "
import sys
# mock lgpio so gpio_executor imports cleanly on dev machine
from unittest.mock import MagicMock
sys.modules['lgpio'] = MagicMock()
import config, schema, llm_parser, agent, rule_based
print('All imports OK')
"
```

Expected: `All imports OK`

- [ ] **Step 5: Commit**

```bash
cd /Users/ezslaptop/Projects/6895_Final_project && git add config.py requirements_pi.txt nova.py && git commit -m "feat: nova.py entry point, max_new_tokens=96, add sentence-transformers"
```

---

### Task 6: Piper TTS — replace TTSEngine in audio.py

**Files:**
- Modify: `audio.py` — replace `TTSEngine` pyttsx3 implementation with Piper
- Create: `voices/` directory + download voice model

**Context:** `TTSEngine` is instantiated in `nova.py` and passed to `NovaAgent` as `speak=tts.speak`. The interface must stay identical: `speak(text: str, verbose: bool = True)`. Only the internals change.

- [ ] **Step 1: Download Piper voice model**

```bash
mkdir -p /Users/ezslaptop/Projects/6895_Final_project/voices
python3 -c "
import urllib.request, os
url = 'https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx'
path = 'voices/en_US-lessac-medium.onnx'
if not os.path.exists(path):
    print('Downloading Piper voice model (~60MB)...')
    urllib.request.urlretrieve(url, path)
    print('Done.')
else:
    print('Already present.')
"
```

Expected: `Done.` or `Already present.`

- [ ] **Step 2: Replace TTSEngine in audio.py**

Replace the entire `TTSEngine` class (from `class TTSEngine:` to the end of `runAndWait()`) with:

```python
class TTSEngine:
    """Piper-based TTS. Interface unchanged: speak(text, verbose=True)."""

    def __init__(self, model_path: str = None):
        from piper.voice import PiperVoice
        import wave, io
        from config import PIPER_MODEL_PATH
        _path = model_path or PIPER_MODEL_PATH
        print(f"Loading Piper TTS ({_path}) ...")
        self._voice = PiperVoice.load(_path)
        self._wave  = wave
        self._io    = io
        print("TTS ready.")

    def speak(self, text: str, verbose: bool = True) -> None:
        if verbose:
            print("[TTS]", text)
        try:
            wav_buf = self._io.BytesIO()
            with self._wave.open(wav_buf, 'wb') as wf:
                self._voice.synthesize(text, wf)
            wav_buf.seek(0)
            with self._wave.open(wav_buf, 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                rate   = wf.getframerate()
            import numpy as np
            import sounddevice as sd
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            sd.play(audio, rate)
            sd.wait()
        except Exception as e:
            print("[TTS ERROR]", e)
```

Also remove the `import pyttsx3` line at the top of `audio.py` (it is no longer needed).

- [ ] **Step 3: Smoke-test Piper TTS**

```bash
cd /Users/ezslaptop/Projects/6895_Final_project && python3 -c "
from audio import TTSEngine
tts = TTSEngine()
tts.speak('Hello, I am Nova.')
"
```

Expected: voice plays through speakers. If silent, check default audio output device.

- [ ] **Step 4: Commit**

```bash
cd /Users/ezslaptop/Projects/6895_Final_project && git add audio.py && git commit -m "feat: Piper TTS in TTSEngine, replaces pyttsx3"
```

---

### Task 7: agent.py — GPIO injection, rule-based fast path, parallel execution

**Files:**
- Modify: `agent.py` — add `gpio` param, rule-based fast path, parallel GPIO+TTS

**Context:** `NovaAgent.__init__` currently takes `llm`, `memory`, `speak`. We add `gpio: GPIOExecutor` as a fourth parameter. The rule-based fast path runs before `self._llm.parse_unified()` in `_handle_new_request`. For `direct_command`, GPIO execution and TTS run concurrently in two threads.

- [ ] **Step 1: Add gpio import and update __init__**

At the top of `agent.py`, add:
```python
import threading
from gpio_executor import GPIOExecutor
from rule_based import try_rule_based
```

Change `NovaAgent.__init__` signature and body:
```python
def __init__(self, llm, memory, speak: Callable[[str], None],
             gpio: GPIOExecutor = None):
    self._llm    = llm
    self._memory = memory
    self._speak  = speak
    self._gpio   = gpio
    self._state  = self._blank_state()
```

- [ ] **Step 2: Add rule-based fast path in _handle_new_request**

In `_handle_new_request`, insert before the `self._llm.parse_unified(text, verbose=verbose)` call:

```python
# Rule-based fast path: skip LLM for unambiguous direct commands
fast = try_rule_based(text)
if fast is not None:
    fast["reply"] = fast.get("reply") or self._rule_reply(fast)
    return self._do_direct_command(fast, text, 0.0)

semantic, _, ms = self._llm.parse_unified(text, verbose=verbose)
```

Add helper method to `NovaAgent`:
```python
@staticmethod
def _rule_reply(semantic: dict) -> str:
    device = semantic.get("device", "")
    action = semantic.get("action", "")
    value  = semantic.get("value")
    if action == "turn_on":
        return f"Sure, turning on the {device}."
    if action == "turn_off":
        return f"Sure, turning off the {device}."
    if action == "set_brightness":
        return f"Sure, setting brightness to {value} percent."
    if action == "rgb_cycle":
        return "Sure, starting RGB cycle."
    if action == "open":
        return f"Sure, opening the {device}."
    if action == "close":
        return f"Sure, closing the {device}."
    if action == "set_position":
        return f"Sure, setting {device} to {value} percent."
    if action == "set_temperature":
        return f"Sure, setting AC to {value} degrees."
    return "Done."
```

- [ ] **Step 3: Add parallel GPIO + TTS in _do_direct_command**

Replace the `_do_direct_command` method body with:

```python
def _do_direct_command(self, semantic, text, ms) -> Dict[str, Any]:
    cmd = {k: semantic[k] for k in ("device", "action", "value")}
    ok, reason = validate_command(cmd)
    if ok:
        reply = semantic.get("reply") or "Done."
        hw_result = execute_command(cmd)

        # GPIO and TTS run concurrently
        gpio_thread = threading.Thread(
            target=self._gpio.execute, args=(cmd,), daemon=True
        ) if self._gpio else None
        tts_thread = threading.Thread(
            target=self._speak, args=(reply,), daemon=True
        )
        if gpio_thread:
            gpio_thread.start()
        tts_thread.start()
        if gpio_thread:
            gpio_thread.join()
        tts_thread.join()

        self._update_pref(cmd)
        self._memory.save_episode(text, "direct_command", reply)
        self._memory.push_working("nova", reply)
        return self._result(True, semantic, True, reason,
                            hw_result, reply, round(ms, 3))

    self._memory.push_working("nova", "(command invalid)")
    return self._result(True, semantic, False, reason, "SKIPPED", None, round(ms, 3))
```

- [ ] **Step 4: Run all tests to confirm nothing broken**

```bash
cd /Users/ezslaptop/Projects/6895_Final_project && python3 -m pytest tests/ -v
```

Expected: all tests PASS (test_rule_based.py + test_gpio_executor.py).

- [ ] **Step 5: Commit**

```bash
cd /Users/ezslaptop/Projects/6895_Final_project && git add agent.py && git commit -m "feat: agent.py — GPIO injection, rule-based fast path, parallel execution"
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
cd /Users/ezslaptop/Projects/6895_Final_project && git add bt-speaker.service nova.service && git commit -m "feat: systemd service files for Nova and Bluetooth speaker"
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

- [ ] **Step 2: Make executable and syntax check**

```bash
chmod +x deploy.sh && bash -n deploy.sh && echo "Syntax OK"
```

Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
cd /Users/ezslaptop/Projects/6895_Final_project && git add deploy.sh && git commit -m "feat: one-command deploy.sh for Pi 5"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Covered by |
|-----------------|-----------|
| Pi 5 uses lgpio, not RPi.GPIO | Task 3+4 |
| GPIO 17/27 RGB LED, GPIO 5/6/13/19 stepper | Task 3+4 |
| External PSU + common ground note | Task 3+4 (comment in gpio_executor.py) |
| P9813 full action mapping | Task 3+4 |
| Stepper half-step + position tracking | Task 3+4 |
| max_new_tokens=96 | Task 5 (config.py) |
| nova.py thin orchestration entry point | Task 5 |
| sentence-transformers in requirements | Task 5 |
| Piper TTS replaces pyttsx3 in TTSEngine | Task 6 |
| Rule-based fast path in agent.py | Task 7 |
| Parallel GPIO + TTS execution | Task 7 |
| GPIO injected into NovaAgent | Task 7 |
| SunFounder USB mic → default PipeWire input | Task 9 |
| Bluetooth auto-connect via bt-speaker.service | Tasks 8–9 |
| nova.service After=bt-speaker.service | Task 8 |
| deploy.sh: rsync + pip + Piper model + systemd | Task 9 |

**Type consistency:**
- `GPIOExecutor.execute(cmd: dict) -> str` — defined Task 3+4, used Task 7 ✅
- `try_rule_based(text: str)` — defined Task 2 (done), imported Task 7 ✅
- `TTSEngine.speak(text, verbose)` — interface unchanged Tasks 6–7 ✅
- `NovaAgent(llm, memory, speak, gpio)` — defined Task 7, constructed Task 5 ✅
- `gpio.cleanup()` — defined Task 3+4, called in nova.py Task 5 ✅
