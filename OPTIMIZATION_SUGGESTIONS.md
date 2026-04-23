# Nova Smart Home Assistant — Optimization Suggestions

## Current State

| Metric | Value |
|--------|-------|
| Exact Match Accuracy | 33.33% (2/6 test cases) |
| Average LLM Latency | ~18,400 ms |
| STT Model | faster-whisper tiny.en (int8, CPU) |
| LLM Model | TinyLlama-1.1B-Chat-v1.0 (float32, CPU) |

---

## Issue 1 — LoRA Adapter Trained but Never Loaded

### What's wrong
`lora_training.ipynb` trains a LoRA adapter and saves it to `./tinyllama_home_lora/final_adapter`, but `Nova_4_16.ipynb` loads the bare base model without it. All fine-tuning work is silently ignored at runtime.

### Fix
In Section 5 of `Nova_4_16.ipynb`, after loading the base model, add:

```python
from peft import PeftModel

ADAPTER_PATH = "./tinyllama_home_lora/final_adapter"
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()
print("LoRA adapter loaded.")
```

### Expected impact
Direct accuracy improvement for all cases the adapter was trained on (~30–40% accuracy gain).

---

## Issue 2 — System Prompt Mismatch Between Training and Inference

### What's wrong
The LoRA adapter was trained with a short compressed system prompt (LoRA notebook), but inference uses a 600-token `UNIFIED_SYSTEM_PROMPT` (main notebook). The adapter was fine-tuned for one format and is being asked to operate in another. Additionally, the LoRA training prompt omits `general_qa` and `invalid` types entirely.

### Fix
Replace `UNIFIED_SYSTEM_PROMPT` in `Nova_4_16.ipynb` Section 5 with the version below. This is shorter (~200 tokens vs ~600), matches the training format, and adds the missing output types with concrete few-shot examples for the failing cases.

```python
UNIFIED_SYSTEM_PROMPT = """
Return exactly one JSON object and nothing else.

Allowed outputs:

{"type":"direct_command","device":"light|curtain|window|ac","action":"turn_on|turn_off|set_brightness|rgb_cycle|open|close|set_position|set_temperature","value":null_or_int}

{"type":"needs_clarification","question":"...","options":["...","..."]}

{"type":"general_qa","answer":"..."}

{"type":"invalid"}

Rules:
- Named device + clear action → direct_command
- Indirect/comfort/mood/feeling request → needs_clarification
- Non-device question → general_qa
- No meaningful request → invalid
- Output JSON only. No extra text.

Examples:
Input: Nova, turn on the light.
Output: {"type":"direct_command","device":"light","action":"turn_on","value":null}

Input: Nova, set the AC to 24 degrees.
Output: {"type":"direct_command","device":"ac","action":"set_temperature","value":24}

Input: Nova, I feel cold.
Output: {"type":"needs_clarification","question":"Would you like me to close the window or raise the AC temperature?","options":["close_window","raise_ac_temperature"]}

Input: Nova, it's a bit dark.
Output: {"type":"needs_clarification","question":"Would you like me to turn on the light or open the curtain?","options":["turn_on_light","open_curtain"]}

Input: Nova, fuck this light.
Output: {"type":"needs_clarification","question":"Would you like me to turn off the light or dim it?","options":["turn_off_light","dim_light"]}

Input: Nova, make this room lively.
Output: {"type":"needs_clarification","question":"Would you like me to turn on the RGB cycle or open the curtain?","options":["rgb_cycle","open_curtain"]}

Input: Nova, how do I eat an apple?
Output: {"type":"general_qa","answer":"Wash it first, then eat it."}

Input: Nova, can I still eat this dish after a night in the fridge?
Output: {"type":"general_qa","answer":"Yes, most cooked food is safe for up to 3-4 days in the fridge."}

Input: Hello.
Output: {"type":"invalid"}
""".strip()
```

### Expected impact
~30% latency reduction (fewer prompt tokens to encode). Fixes the failing test cases for `needs_clarification` and `general_qa` misclassification.

---

## Issue 3 — LLM Runs on CPU with Float32 (Major Latency Source)

### What's wrong
TinyLlama 1.1B with a 600-token prompt on CPU takes 12–27 seconds per call. The bulk of end-to-end latency comes from this single step.

### Fix A — Rule-Based Fast Path (recommended, highest ROI)

Most home commands are unambiguous. Skip the LLM for clear cases and only invoke it for indirect/ambiguous inputs.

Add this function before `llm_parse_unified` in Section 5 and call it first in `handle_transcribed_text`:

```python
import re

def try_rule_based(text: str):
    t = text.lower()

    # AC temperature: "set AC to 24 degrees"
    m = re.search(r"(?:ac|air.?con).*?(\d+)\s*degree|(\d+)\s*degree.*?(?:ac|air.?con)", t)
    if m:
        val = int(m.group(1) or m.group(2))
        if 16 <= val <= 30:
            return {"type": "direct_command", "device": "ac", "action": "set_temperature", "value": val}

    # Brightness: "set brightness to 70"
    m = re.search(r"brightness.*?(\d+)|(\d+).*?brightness", t)
    if m:
        val = int(m.group(1) or m.group(2))
        if 0 <= val <= 100:
            return {"type": "direct_command", "device": "light", "action": "set_brightness", "value": val}

    # Curtain/window position: "set curtain to 50 percent"
    m = re.search(r"(curtain|window).*?(\d+)\s*(?:percent|%)|(\d+)\s*(?:percent|%).*?(curtain|window)", t)
    if m:
        device = m.group(1) or m.group(4)
        val = int(m.group(2) or m.group(3))
        if 0 <= val <= 100:
            return {"type": "direct_command", "device": device, "action": "set_position", "value": val}

    patterns = [
        (r"\b(?:turn on|switch on)\b.*\blight\b|\bright\b.*\bturn on\b",
         {"device": "light", "action": "turn_on", "value": None}),
        (r"\b(?:turn off|switch off)\b.*\blight\b|\bright\b.*\bturn off\b",
         {"device": "light", "action": "turn_off", "value": None}),
        (r"\brgb\b|\brgb cycle\b",
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

    return None  # fall through to LLM
```

In `handle_transcribed_text`, call this before `llm_parse_unified`:

```python
# After the assistant name check, before calling LLM:
fast_result = try_rule_based(text)
if fast_result is not None:
    semantic = fast_result
    llm_latency_ms = 0.0
else:
    semantic, raw_output, llm_latency_ms = llm_parse_unified(text, verbose=verbose)
```

### Fix B — Use GGUF Quantized Model via llama-cpp-python

Replaces HuggingFace inference with llama.cpp, which is ~5–10x faster on CPU.

```bash
pip install llama-cpp-python
# Download TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf from HuggingFace
```

```python
from llama_cpp import Llama

llm_cpp = Llama(
    model_path="TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf",
    n_ctx=512,
    n_threads=4,
    verbose=False,
)

def llm_generate_json_cpp(system_prompt: str, user_prompt: str, max_new_tokens: int = 96):
    start = time.perf_counter()
    response = llm_cpp.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_new_tokens,
        temperature=0.0,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    raw = response["choices"][0]["message"]["content"].strip()
    json_str = extract_first_json_object(raw)
    if json_str is None:
        return {"type": "invalid"}, raw, latency_ms
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        return {"type": "invalid"}, raw, latency_ms
    return normalize_unified_result(parsed), raw, latency_ms
```

### Expected impact (Fix A)
Latency for unambiguous commands: **18,000 ms → <5 ms**. Covers roughly 70% of real-world home control requests.

### Expected impact (Fix B)
Latency for LLM calls: **12,000–27,000 ms → 2,000–4,000 ms**.

Both fixes can be applied together — rule-based handles clear commands, GGUF handles the rest.

---

## Issue 4 — Disk I/O on Every STT Call

### What's wrong
`save_audio()` writes a WAV file to disk, then `transcribe_audio_file()` reads it back. This adds ~50–100 ms of unnecessary file I/O on every utterance.

### Fix
Transcribe directly from an in-memory buffer:

```python
import io

def transcribe_audio_numpy(audio: np.ndarray) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="wav")
    buf.seek(0)
    segments, _ = stt_model.transcribe(buf, beam_size=1)
    return " ".join(seg.text.strip() for seg in segments).strip()
```

Replace all calls to `transcribe_audio_file(audio_path)` with `transcribe_audio_numpy(audio)`, and remove the `save_audio()` call that precedes them.

### Expected impact
~50–100 ms saved per utterance, plus eliminates disk wear from constant temp file writes.

---

## Issue 5 — Wake Word Detection is Fragile

### What's wrong
Whisper frequently mishears "Nova" as unrelated words ("Nanaoba", "no", "Henry", etc.), causing valid commands to be silently dropped by the pre-filter. The variant list currently only contains `["nova"]`.

### Fix A — Extend the variant list

```python
ASSISTANT_NAME_VARIANTS = [
    "nova", "nava", "no va", "noba", "noa", "nove", "novia",
]
```

### Fix B — Dedicated wake word detector (long-term)

Use [openWakeWord](https://github.com/dscripka/openWakeWord) to detect the wake word from raw audio before running Whisper. This avoids running STT on every utterance — only process audio that already contains a confirmed "Nova".

```bash
pip install openwakeword
```

```python
from openwakeword.model import Model as WakeWordModel

oww_model = WakeWordModel(wakeword_models=["nova"], inference_framework="onnx")

def has_wake_word(audio: np.ndarray) -> bool:
    prediction = oww_model.predict(audio)
    return prediction.get("nova", 0) > 0.5
```

In `collect_one_utterance_from_stream`, call `has_wake_word(audio)` before passing to STT. Reduces Whisper calls by ~80%.

### Expected impact
Eliminates missed commands caused by STT mishearing "Nova", particularly for non-native English speakers or noisy environments.

---

## Issue 6 — LoRA Training Data Missing Key Intent Classes

### What's wrong
The `lora_training.ipynb` training data (`train.jsonl`) only includes `direct_command` and `needs_clarification` examples. The `general_qa` and `invalid` types are not in the training set, so the fine-tuned model never learned to output them reliably.

### Fix
Add training examples for the missing types to `train.jsonl` and `val.jsonl`, then re-train:

```jsonl
{"input": "Nova, how do I eat an apple?", "output": "{\"type\":\"general_qa\",\"answer\":\"Wash it first, then eat it.\"}"}
{"input": "Nova, what time is it?", "output": "{\"type\":\"general_qa\",\"answer\":\"I don't have access to a clock, but you can check your phone.\"}"}
{"input": "Nova, can I eat leftovers after 3 days?", "output": "{\"type\":\"general_qa\",\"answer\":\"Yes, most cooked food is safe for 3-4 days in the fridge.\"}"}
{"input": "Hello.", "output": "{\"type\":\"invalid\"}"}
{"input": "Never mind.", "output": "{\"type\":\"invalid\"}"}
{"input": "Nova, nothing.", "output": "{\"type\":\"invalid\"}"}
```

Also increase training epochs from 3 to 5 and add a learning rate scheduler, since the current training loss (0.878) is still high.

### Expected impact
Fixes the `general_qa`/`needs_clarification` confusion and prevents the model from hallucinating `direct_command` for indirect requests.

---

## Recommended Priority Order

| Priority | Fix | Latency Impact | Accuracy Impact | Effort |
|----------|-----|---------------|-----------------|--------|
| 1 | Load LoRA adapter | none | +30–40% | 3 lines |
| 2 | Replace system prompt | −30% LLM latency | +10% | paste replacement |
| 3 | Rule-based fast path | −95% for direct cmds | +5% | ~50 lines |
| 4 | In-memory STT | −50–100 ms | none | 5 lines |
| 5 | Wake word variants | none | fewer missed cmds | 1 line |
| 6 | Add missing LoRA training data + re-train | none | +15–20% | new data + re-train |
| 7 | GGUF via llama-cpp-python | −80% LLM latency | neutral | install + swap |

With fixes 1–5 applied, the expected outcome is:
- **Accuracy: 33% → ~80%+**
- **Latency for direct commands: 18 s → <10 ms**
- **Latency for ambiguous commands: 18 s → ~3–5 s** (with GGUF), or ~12–18 s (without)
