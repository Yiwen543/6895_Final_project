# Hardware Extension: Window Motor + Curtain Position + Relative Color Temp

**Goal:** Add window stepper motor, LLM-inferred curtain position percentage, and rule-based relative color temperature adjustment (+1/-1 step).

**Architecture:** Three independent changes sharing one PR: (1) new window device in GPIOExecutor, (2) prompt examples for curtain percentage inference, (3) stateful rule_based intercept for relative color temp commands.

**Tech Stack:** lgpio, 28BYJ-48 + ULN2003, llama-cpp-python, difflib (existing)

---

## 1. Window Stepper Motor

**Hardware:** Second 28BYJ-48 + ULN2003 driver board.

**GPIO pins:**
```
WINDOW_PINS = [17, 27, 22, 23]   # ULN2003 IN1-IN4 → Pi Pins 11,13,15,16
```

**Behavior:** open/close only — no `set_position`. `_window_pos` tracks 0 (closed) or 100 (open).

**Files:**
- `config.py`: add `WINDOW_PINS = [17, 27, 22, 23]`
- `gpio_executor.py`: claim WINDOW_PINS in `__init__`, add window branch in `execute()`, reuse `_step_to_position` with target 0 or 100

**schema.py:** window device already exists with `open`/`close` actions — no change needed.

---

## 2. Curtain Position Percentage (Prompt Engineering)

LLM already supports `set_position` action with integer value 0-100. Add examples so it infers percentage from natural language adverbs.

**New examples in `UNIFIED_SYSTEM_PROMPT`:**

| Input | Output |
|-------|--------|
| "Cathey, open the curtain a little." | `set_position value:20` |
| "Cathey, open the curtain halfway." | `set_position value:50` |
| "Cathey, open the curtain most of the way." | `set_position value:80` |

`open` (no qualifier) → continues to use `action:open` (full 100%).

**Calibration:** `CURTAIN_TOTAL_STEPS` in `config.py` — user sets based on physical mechanism.

---

## 3. Relative Color Temperature Adjustment

**Problem:** "make the light warmer" requires knowing the current color temp level. Routing through LLM adds 5-40s latency unnecessarily.

**Solution:** Intercept in `rule_based.py` before LLM is called.

### State tracking

`gpio_executor.py` adds:
- `_color_temp_level: int = 3` (initialized to neutral)
- Updated on every `set_color_temp` execution
- Exposed via `get_device_state() -> dict` returning `{"color_temp": int, "brightness": int, "curtain_pos": int, "window_pos": int}`

### rule_based.py changes

```python
def try_rule_based(text: str, state: dict = None) -> Optional[dict]:
```

New relative color temp patterns (checked before existing rules):
- `warmer / cozier / more warm` → `+1`
- `cooler / colder / more cool / more cold` → `-1`

Logic:
```python
current = (state or {}).get("color_temp", 3)
new_val = max(1, min(5, current + delta))
return {"type": "direct_command", "device": "light",
        "action": "set_color_temp", "value": new_val}
```

### agent.py changes

```python
fast = try_rule_based(text, self._gpio.get_device_state() if self._gpio else None)
```

---

## Error Handling

- Window/curtain motor not physically connected: `lgpio` will raise on `gpio_claim_output` — caught by existing `try/except` in `GPIOExecutor.__init__`, falls back to stub.
- Color temp clamp: `max(1, min(5, current + delta))` prevents out-of-range.
- `state=None` in `try_rule_based`: defaults color_temp to 3 (neutral), safe fallback.

---

## What Is NOT in Scope

- Brightness relative adjustment ("a bit brighter") — same pattern but deferred.
- Curtain position feedback (no encoder on 28BYJ-48).
- Window `set_position` — window is open/close only by design.
