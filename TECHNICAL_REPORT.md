# Cathey — Technical Report
**EECS 6895 Final Project**

---

## 1. Project Overview

Cathey is a fully local, offline voice-controlled smart home assistant running on a Raspberry Pi 5. All inference (STT, SLM, TTS) is performed on-device with no cloud dependency. The system controls four physical devices: LED lighting, curtain motor, window motor, and a PWM fan simulating an air conditioner.

---

## 2. Hardware

| Component | Specification |
|-----------|--------------|
| Compute | Raspberry Pi 5 (Broadcom BCM2712, 4× Cortex-A76 @ 2.4 GHz) |
| Microphone | SunFounder USB microphone |
| Speaker | Bluetooth speaker (PipeWire audio) |
| LED | WS2812B 12-LED ring via SPI0 (GPIO 10 / Pin 19) |
| Curtain | 28BYJ-48 stepper + ULN2003 driver (GPIO 5, 6, 13, 26) |
| Window | 28BYJ-48 stepper + ULN2003 driver (GPIO 17, 27, 22, 23) |
| Fan (AC) | Noctua NF-A4x10 5V PWM (GPIO 12 / Pin 32, hardware PWM0) |

---

## 3. Software Stack

| Layer | Technology |
|-------|-----------|
| OS | Raspberry Pi OS (64-bit) |
| STT | faster-whisper tiny.en, int8, CPU |
| SLM | Qwen2.5-3B-Instruct Q3_K_M (GGUF), llama.cpp via llama-cpp-python |
| TTS | Piper en_US-lessac-medium, 170 WPM |
| Audio | sounddevice + PipeWire (pw-play) |
| GPIO | lgpio + pi5neo (SPI LED) |
| Memory | ChromaDB + sentence-transformers/all-MiniLM-L6-v2 |
| Dev (Mac) | HuggingFace transformers, MPS backend |

---

## 4. System Architecture

### 4.1 Pipeline

```
Microphone
    │
    ▼
VAD (energy-based, threshold=0.05)
    │  silence ≥ 0.5s → utterance complete
    ▼
STT (faster-whisper)          ~1.2 s
    │
    ▼
Wake word detection           <1 ms
    │  fuzzy match on "Cathey" variants
    ▼
Rule-based fast path          <5 ms  ─── hit (~70%) ──→ Execute
    │  miss (~30%)
    ▼
Episodic memory retrieval     ~0.1–1 s
    │
    ▼
SLM inference (llama.cpp)     15–50 s
    │
    ▼
Intent dispatch
    ├── direct_command   → validate → GPIO + TTS (concurrent)
    ├── needs_clarification → TTS question → await reply → resolve_followup
    ├── general_qa       → answer_qa (with memory context)
    └── invalid          → silent discard
    │
    ▼
Memory write (episodic / procedural / semantic)
```

### 4.2 File Structure

| File | Role |
|------|------|
| `cathey.py` | Entry point, component assembly |
| `agent.py` | Stateful intent dispatcher |
| `audio.py` | STT, TTS, VAD recording loop |
| `llm_parser.py` | LLM inference, system prompts |
| `rule_based.py` | Regex fast path |
| `schema.py` | Device schema, validation, execution table |
| `memory.py` | Four-layer memory manager |
| `gpio_executor.py` | Hardware execution (lgpio) |
| `config.py` | All configuration constants |

---

## 5. Intent Classification

### 5.1 Four Intent Types

| Type | Condition | Example |
|------|-----------|---------|
| `direct_command` | Explicit device + action named | "Turn on the light" |
| `needs_clarification` | Feeling/comfort expressed, no device named | "I feel cold" |
| `general_qa` | Unrelated to device control | "What's my name?" |
| `invalid` | Unintelligible or empty input | (noise) |

### 5.2 Classification Boundary

The critical boundary is **feeling vs. command**. Words such as "cold", "hot", "dark", "bright", "stuffy" are feelings — they never map directly to a device action. The system prompt enforces this with an explicit rule and few-shot examples.

### 5.3 Rule-Based Fast Path

Unambiguous direct commands are intercepted by `rule_based.py` via regex before the SLM is called. This handles ~70% of all direct commands with <5 ms latency. Examples:

- `"turn on.*light"` → `direct_command: light / turn_on`
- `"open.*curtain"` → `direct_command: curtain / open`
- `"set AC to (\d+) degrees"` → `direct_command: ac / set_temperature`
- `"open curtain a little"` → `direct_command: curtain / set_position 20`

### 5.4 LLM Prompt Design

`UNIFIED_SYSTEM_PROMPT` contains four sections:
1. **Output format**: Four possible JSON schemas, nothing else
2. **Value constraints**: `set_color_temp` 1–5, `set_position` 0–100
3. **Classification rules**: Concise rules for each intent type
4. **Few-shot examples**: ~10 Input/Output pairs covering edge cases

The prompt was iteratively trimmed to minimize token count (target <500 tokens) to reduce prefill latency on Pi.

### 5.5 Clarification Flow

```
needs_clarification
    │
    ├─ lookup_skill() → procedural memory hit → auto-execute
    │
    └─ no match → TTS question → await user reply
                      │
                      ▼
               resolve_followup() (FOLLOWUP_RESOLUTION_SYSTEM_PROMPT)
                      │
                      ▼
               map option → direct_command JSON → execute
```

Option-to-action mapping is hardcoded in the followup prompt:
- `close_window` → `device=window, action=close`
- `raise_ac_temperature` → `device=ac, action=set_temperature, value=26`
- `lower_ac_temperature` → `device=ac, action=set_temperature, value=20`

---

## 6. Memory Architecture

### 6.1 Four Layers

| Layer | Storage | Persistence | Purpose |
|-------|---------|-------------|---------|
| Working | Python deque (maxlen=8) | RAM only | Current session context |
| Episodic | ChromaDB vector store | Disk | Semantically similar past interactions |
| Procedural | `skills.json` | Disk | Learned trigger→action habits |
| Semantic | `user_prefs.json` | Disk | Structured personal preferences |

### 6.2 Key Design Decision: Semantic vs. Episodic for Personal Facts

Personal facts (e.g., user name) are stored in semantic memory as key-value pairs (`user_name: Alex`), not in the episodic vector store. This decision was made because:

- STT noise corrupts episodic entries (e.g., "Alex" transcribed as "A")
- Vector retrieval is fuzzy — incorrect entries can rank above correct ones
- Key-value lookup is deterministic and immune to transcription errors

The name is extracted via regex (`my name is ([A-Za-z]+)`) and written to `user_prefs.json` immediately.

### 6.3 Context Building

Before each SLM call, `build_context()` assembles relevant episodic episodes (cosine distance < 0.6) as a context string passed to `parse_unified()`. This enables single-call QA with memory recall.

---

## 7. Device Control

### 7.1 Schema-Driven Validation

`COMMAND_SCHEMA` in `schema.py` defines valid actions and value ranges per device. `validate_command()` is purely data-driven (dict lookup, no if-else chains).

| Device | Actions | Value |
|--------|---------|-------|
| light | turn_on, turn_off, set_brightness, set_color_temp, rgb_cycle, party_mode | brightness: 0–100, color_temp: 1–5 |
| curtain | open, close, set_position | position: 0–100% |
| window | open, close, set_position | position: 0–100% |
| ac | turn_on, turn_off, set_temperature | temperature: 16–30°C |

### 7.2 AC / Fan

The PWM fan maps AC temperature to two discrete speeds:
- 16–22°C → 20% duty cycle (slow)
- 23–30°C → 100% duty cycle (fast)
- `turn_on` default → 50%

### 7.3 LED Color Temperature

5-level color temperature scale mapped to RGB:

| Level | Color Temp | RGB |
|-------|-----------|-----|
| 1 | 6500K daylight | (180, 210, 255) |
| 2 | 5000K reading | (255, 255, 255) |
| 3 | 4000K warm | (255, 200, 80) |
| 4 | 3000K orange | (255, 120, 0) |
| 5 | 2700K deep orange | (255, 50, 0) |

---

## 8. Latency Profile

| Stage | Latency |
|-------|---------|
| VAD + recording | real-time |
| STT (faster-whisper tiny.en) | ~1.2 s |
| Wake word detection | <1 ms |
| Rule-based fast path | <5 ms |
| Episodic retrieval (embedding) | ~0.1–1 s |
| SLM inference (llama.cpp, n_ctx=1024) | 15–50 s |
| TTS synthesis + playback | ~1–2 s |
| GPIO execution | <100 ms |

**End-to-end (rule-based path):** ~2–3 s  
**End-to-end (SLM path):** ~18–55 s

### 8.1 Latency Optimizations Applied

- Rule-based fast path bypasses SLM for ~70% of commands
- `n_ctx` reduced from 2048 → 1024 (20–30% inference speedup)
- System prompt trimmed by ~40%
- Embedding model pre-warmed at startup (eliminates 10–30 s cold-start)
- Episodic retrieval skipped when memory is empty
- GPIO and TTS execute concurrently via threading

---

## 9. Model Configuration

```
SLM:      Qwen2.5-3B-Instruct Q3_K_M (GGUF)
Backend:  llama-cpp-python
n_ctx:    1024
n_threads: 4
temperature: 0 (deterministic)
max_new_tokens: 150
response_format: json_object
```

---

## 10. Performance

| Metric | Value |
|--------|-------|
| Classification accuracy | ~85% (20-command test set) |
| Rule-based coverage | ~70% of direct commands |
| STT accuracy | High (faster-whisper tiny.en, English) |
| Memory recall (name) | Deterministic via semantic memory |

---

## 11. What Was Not Used

| Item | Reason |
|------|--------|
| LoRA fine-tuning (`lora_training.ipynb`) | Few-shot prompt achieves comparable accuracy; re-quantization overhead not justified |
| Cloud APIs | By design — fully offline |
| Wake word detection model | Fuzzy string matching on Cathey variants sufficient |
