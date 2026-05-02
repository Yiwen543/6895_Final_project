# Hardware Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add window stepper motor (GPIO 17,27,22,23), LLM curtain position percentage inference, and rule-based relative color temperature adjustment.

**Architecture:** Four independent commits touching config → gpio_executor → rule_based → agent → llm_parser. Each task is self-contained and testable before the next starts.

**Tech Stack:** lgpio, 28BYJ-48 + ULN2003, llama-cpp-python, pytest

---

## File Map

| File | Change |
|------|--------|
| `config.py` | Add `WINDOW_PINS`, `WINDOW_TOTAL_STEPS` |
| `gpio_executor.py` | Import WINDOW_PINS; add window motor init/move/release; track `_color_temp_level`, `_brightness_level`; add `get_device_state()`; add window branch in `execute()`; update `cleanup()` |
| `rule_based.py` | Add `state: dict = None` param; add relative color temp patterns before existing rules |
| `agent.py` | Pass `self._gpio.get_device_state()` to `try_rule_based` |
| `llm_parser.py` | Add curtain set_position percentage examples to `UNIFIED_SYSTEM_PROMPT` |
| `tests/test_hardware_extension.py` | Unit tests for rule_based relative logic and get_device_state |

---

## Task 1: config.py — Window pins and steps

**Files:**
- Modify: `config.py`
- Test: `tests/test_hardware_extension.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_hardware_extension.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/Projects/6895_Final_project
pytest tests/test_hardware_extension.py::test_window_pins_in_config -v
```

Expected: `FAILED` — `AttributeError: module 'config' has no attribute 'WINDOW_PINS'`

- [ ] **Step 3: Add constants to config.py**

In `config.py`, add after the stepper motor comment in the GPIO section. There is no dedicated GPIO section yet — add it after the Audio capture block:

```python
# ── GPIO hardware ─────────────────────────────────────────────────────────────
WINDOW_PINS        = [17, 27, 22, 23]   # ULN2003 IN1-IN4 → Pi Pins 11,13,15,16
WINDOW_TOTAL_STEPS = 2048               # calibrate to physical window mechanism
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_hardware_extension.py::test_window_pins_in_config tests/test_hardware_extension.py::test_window_total_steps_in_config -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_hardware_extension.py
git commit -m "feat: add WINDOW_PINS and WINDOW_TOTAL_STEPS to config"
```

---

## Task 2: gpio_executor.py — Window motor + device state tracking

**Files:**
- Modify: `gpio_executor.py`
- Modify: `tests/test_hardware_extension.py`

The window motor is a second 28BYJ-48. It reuses the same HALF_STEP_SEQ but has its own pins, step index, and position. Additionally this task adds `_color_temp_level` / `_brightness_level` state tracking and a `get_device_state()` method.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_hardware_extension.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_hardware_extension.py::test_get_device_state_mock tests/test_hardware_extension.py::test_get_device_state_updates_after_command -v
```

Expected: `FAILED` — `AttributeError: 'GPIOExecutor' object has no attribute 'get_device_state'`

- [ ] **Step 3: Implement window motor + state tracking in gpio_executor.py**

**3a.** Add import at top of file:

```python
from config import WINDOW_PINS, WINDOW_TOTAL_STEPS
```

**3b.** Add module-level constant (after existing `STEP_DELAY`):

```python
WINDOW_STEP_DELAY = 0.002
```

**3c.** In `__init__`, after the existing stepper motor block, add:

```python
        # ── Window stepper motor ──────────────────────────────────────────────
        for pin in WINDOW_PINS:
            lgpio.gpio_claim_output(self._h, pin)
        self._window_step_index = 0
        self._window_pos = 0
        self._release_window_motor()

        # ── Device state ──────────────────────────────────────────────────────
        self._color_temp_level = 3
        self._brightness_level = 100
```

**3d.** Add window motor helper methods after `_release_motor`:

```python
    def _do_window_step(self, direction: int) -> None:
        self._window_step_index = (self._window_step_index + direction) % 8
        for i, pin in enumerate(WINDOW_PINS):
            lgpio.gpio_write(self._h, pin, self.HALF_STEP_SEQ[self._window_step_index][i])
        time.sleep(WINDOW_STEP_DELAY)

    def _release_window_motor(self) -> None:
        for pin in WINDOW_PINS:
            lgpio.gpio_write(self._h, pin, 0)

    def _move_window(self, target_pct: int) -> None:
        target_pct = 0 if target_pct <= 50 else 100
        steps     = int((target_pct - self._window_pos) / 100 * WINDOW_TOTAL_STEPS)
        direction = 1 if steps >= 0 else -1
        for _ in range(abs(steps)):
            self._do_window_step(direction)
        self._window_pos = target_pct
        self._release_window_motor()
```

**3e.** Add `get_device_state()` after `_move_window`:

```python
    def get_device_state(self) -> dict:
        return {
            "color_temp": self._color_temp_level,
            "brightness": self._brightness_level,
            "curtain_pos": self._curtain_pos,
            "window_pos": self._window_pos,
        }
```

**3f.** In `execute()`, update `set_color_temp` to track state:

Replace:
```python
            if action == "set_color_temp":
                self._stop_rgb_cycle()
                r, g, b = _COLOR_TEMP_RGB.get(int(value), (255, 255, 255))
                self._fill(r, g, b)
                return f"LIGHT -> COLOR TEMP {value}"
```
With:
```python
            if action == "set_color_temp":
                self._stop_rgb_cycle()
                self._color_temp_level = int(value)
                r, g, b = _COLOR_TEMP_RGB.get(self._color_temp_level, (255, 255, 255))
                self._fill(r, g, b)
                return f"LIGHT -> COLOR TEMP {value}"
```

**3g.** Update `set_brightness` to track state:

Replace:
```python
            if action == "set_brightness":
                self._stop_rgb_cycle()
                self._fill_brightness(int(value))
                return f"LIGHT -> BRIGHTNESS {value}%"
```
With:
```python
            if action == "set_brightness":
                self._stop_rgb_cycle()
                self._brightness_level = int(value)
                self._fill_brightness(self._brightness_level)
                return f"LIGHT -> BRIGHTNESS {value}%"
```

**3h.** Add window branch in `execute()`, after the curtain block and before the ac block:

```python
        if device == "window":
            if action == "open":
                self._move_window(100)
                return "WINDOW -> OPEN"
            if action == "close":
                self._move_window(0)
                return "WINDOW -> CLOSE"
```

**3i.** Update `cleanup()` to release window motor, after `self._release_motor()`:

```python
        self._release_window_motor()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_hardware_extension.py::test_get_device_state_mock tests/test_hardware_extension.py::test_get_device_state_updates_after_command -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add gpio_executor.py tests/test_hardware_extension.py
git commit -m "feat: window stepper motor + device state tracking in GPIOExecutor"
```

---

## Task 3: rule_based.py — Relative color temperature

**Files:**
- Modify: `rule_based.py`
- Modify: `tests/test_hardware_extension.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_hardware_extension.py`:

```python
from rule_based import try_rule_based

def test_warmer_increments_color_temp():
    result = try_rule_based("Cathey, make the light warmer", state={"color_temp": 3})
    assert result is not None
    assert result["action"] == "set_color_temp"
    assert result["value"] == 4

def test_cooler_decrements_color_temp():
    result = try_rule_based("Cathey, make the light cooler", state={"color_temp": 3})
    assert result is not None
    assert result["action"] == "set_color_temp"
    assert result["value"] == 2

def test_cozier_increments_color_temp():
    result = try_rule_based("Cathey, make it cozier", state={"color_temp": 2})
    assert result is not None
    assert result["value"] == 3

def test_color_temp_clamps_at_max():
    result = try_rule_based("Cathey, make the light warmer", state={"color_temp": 5})
    assert result["value"] == 5

def test_color_temp_clamps_at_min():
    result = try_rule_based("Cathey, make the light colder", state={"color_temp": 1})
    assert result["value"] == 1

def test_relative_defaults_to_neutral_without_state():
    result = try_rule_based("Cathey, make the light warmer", state=None)
    assert result["value"] == 4  # 3 + 1

def test_existing_rules_still_work_with_state():
    result = try_rule_based("Cathey, turn on the light", state={"color_temp": 3})
    assert result is not None
    assert result["action"] == "turn_on"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_hardware_extension.py -k "warmer or cooler or cozier or clamp or neutral_without or existing_rules" -v
```

Expected: all `FAILED` — `TypeError: try_rule_based() got an unexpected keyword argument 'state'`

- [ ] **Step 3: Implement relative color temp in rule_based.py**

Replace the function signature and add relative patterns at the top of the function body:

```python
import re
from typing import Optional, Dict, Any


def try_rule_based(text: str, state: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
    t = text.lower()

    # Relative color temperature: "warmer/cozier" → +1, "cooler/colder" → -1
    _warmer = re.search(r'\b(?:warmer|cozier|more\s+warm)\b', t)
    _cooler = re.search(r'\b(?:cooler|colder|more\s+coo?l|more\s+cold)\b', t)
    if _warmer or _cooler:
        if re.search(r'\blight\b', t) or re.search(r'\bmake\s+it\b', t):
            delta   = 1 if _warmer else -1
            current = (state or {}).get("color_temp", 3)
            new_val = max(1, min(5, current + delta))
            return {"type": "direct_command", "device": "light",
                    "action": "set_color_temp", "value": new_val}

    # AC temperature: ...  (rest of existing code unchanged)
```

Keep all existing rules below unchanged.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_hardware_extension.py -k "warmer or cooler or cozier or clamp or neutral_without or existing_rules" -v
```

Expected: `7 passed`

- [ ] **Step 5: Run full test file**

```bash
pytest tests/test_hardware_extension.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add rule_based.py tests/test_hardware_extension.py
git commit -m "feat: relative color temp adjustment in rule_based (warmer/cooler ±1 step)"
```

---

## Task 4: agent.py — Wire device state into rule_based call

**Files:**
- Modify: `agent.py:178-181`

Current code in `_handle_new_request` (around line 178):
```python
        fast = try_rule_based(text)
```

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hardware_extension.py`:

```python
def test_agent_passes_state_to_rule_based():
    import unittest.mock as mock
    import sys, types

    # Stub lgpio + pi5neo for import
    for mod in ("lgpio", "pi5neo"):
        if mod not in sys.modules:
            fake = types.ModuleType(mod)
            if mod == "lgpio":
                for fn in ("gpiochip_open","gpio_claim_output","gpio_write","tx_pwm","gpiochip_close"):
                    setattr(fake, fn, mock.Mock(return_value=0))
            else:
                fake.Pi5Neo = mock.Mock()
            sys.modules[mod] = fake

    from agent import CatheyAgent
    from gpio_executor import GPIOExecutor

    gpio = GPIOExecutor.__new__(GPIOExecutor)
    gpio._color_temp_level = 4
    gpio._brightness_level = 80
    gpio._curtain_pos = 0
    gpio._window_pos = 0

    agent = CatheyAgent(llm=mock.Mock(), memory=mock.Mock(), speak=mock.Mock(), gpio=gpio)

    with mock.patch("agent.try_rule_based", return_value=None) as mock_rb:
        agent.handle("Cathey, make the light warmer", verbose=False)
        call_args = mock_rb.call_args
        assert call_args is not None
        passed_state = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("state")
        # try_rule_based receives positional text + state dict
        assert passed_state is not None
        assert passed_state["color_temp"] == 4
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_hardware_extension.py::test_agent_passes_state_to_rule_based -v
```

Expected: `FAILED` — state passed is `None` (current code calls `try_rule_based(text)` with no state)

- [ ] **Step 3: Update agent.py**

In `_handle_new_request`, replace:

```python
        fast = try_rule_based(text)
```

With:

```python
        fast = try_rule_based(text, self._gpio.get_device_state() if self._gpio else None)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_hardware_extension.py::test_agent_passes_state_to_rule_based -v
```

Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add agent.py tests/test_hardware_extension.py
git commit -m "feat: pass device state to try_rule_based for relative commands"
```

---

## Task 5: llm_parser.py — Curtain position percentage examples

**Files:**
- Modify: `llm_parser.py` (`UNIFIED_SYSTEM_PROMPT`)

- [ ] **Step 1: Locate insertion point**

In `llm_parser.py`, find the curtain examples block. Currently the prompt has:

```
Input: Cathey, it's too bright in here.
Output: ...
```

Add after the existing `gloomy` / `too bright` block, before the `make the light warmer` example.

- [ ] **Step 2: Add curtain position examples**

Insert these three examples into `UNIFIED_SYSTEM_PROMPT`:

```
Input: Cathey, open the curtain a little.
Output: {"type":"direct_command","device":"curtain","action":"set_position","value":20,"reply":"Opening the curtain a little."}

Input: Cathey, open the curtain halfway.
Output: {"type":"direct_command","device":"curtain","action":"set_position","value":50,"reply":"Opening the curtain halfway."}

Input: Cathey, open the curtain most of the way.
Output: {"type":"direct_command","device":"curtain","action":"set_position","value":80,"reply":"Opening the curtain most of the way."}
```

- [ ] **Step 3: Verify token count does not exceed n_ctx**

Run on Pi:

```bash
cd ~/nova && python3 -c "
from llm_parser import UNIFIED_SYSTEM_PROMPT
# Rough token estimate: words / 0.75
words = len(UNIFIED_SYSTEM_PROMPT.split())
print(f'Prompt words: {words}, ~tokens: {int(words/0.75)}')
"
```

Expected: `~tokens` well under 900 (n_ctx=2048 with user input headroom).

- [ ] **Step 4: Smoke test on Pi**

```bash
cd ~/nova && python3 -c "
import os; os.chdir('/home/tl3461/nova')
from llm_parser import LLMParser
llm = LLMParser()
for text in [
    'Cathey, open the curtain a little.',
    'Cathey, open the curtain halfway.',
    'Cathey, open the curtain most of the way.',
    'Cathey, open the curtain.',
]:
    r, _, ms = llm.parse_unified(text)
    print(f'{text!r} -> action={r.get(\"action\")} value={r.get(\"value\")} ({ms:.0f}ms)')
" 2>&1 | grep "->"
```

Expected output:
```
'Cathey, open the curtain a little.' -> action=set_position value=20
'Cathey, open the curtain halfway.' -> action=set_position value=50
'Cathey, open the curtain most of the way.' -> action=set_position value=80
'Cathey, open the curtain.' -> action=open value=None
```

- [ ] **Step 5: Commit**

```bash
git add llm_parser.py
git commit -m "feat: curtain set_position percentage inference via prompt examples"
```

---

## Task 6: Integration — sync to Pi and restart

- [ ] **Step 1: Run all tests locally**

```bash
pytest tests/test_hardware_extension.py -v
```

Expected: all pass.

- [ ] **Step 2: Sync to Pi** (confirm with user before running)

```bash
rsync -avz --exclude='.git' --exclude='__pycache__' \
  config.py gpio_executor.py rule_based.py agent.py llm_parser.py \
  tests/ \
  tl3461@192.168.100.1:~/nova/
```

- [ ] **Step 3: Restart nova service**

```bash
ssh tl3461@192.168.100.1 "sudo systemctl restart nova && sudo systemctl is-active nova"
```

Expected: `active`

- [ ] **Step 4: Verify window motor wiring**

Say: `"Cathey, open the window"` — window motor should step to 100%.
Say: `"Cathey, close the window"` — window motor should step back to 0%.

- [ ] **Step 5: Verify relative color temp**

Say: `"Cathey, make the light warmer"` → LED shifts one step warmer.
Say: `"Cathey, make the light warmer"` again → shifts one more step.
Say: `"Cathey, make it cooler"` → shifts one step cooler.
