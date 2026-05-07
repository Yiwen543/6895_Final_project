# TinyLlama + LoRA + Bug Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch deployment LLM from Qwen2.5-3B to TinyLlama-1.1B + LoRA adapter, fix `Hello` → `general_qa` misclassification, and fix memory recall via `answer_qa` + `build_context`.

**Architecture:** `config.py` declares model paths; `llm_parser.py` loads TinyLlama base + optional LoRA adapter and fixes the system prompt; `agent.py` injects semantic memory (prefs) into classification context and always calls `answer_qa` with full memory for `general_qa` intents; `lora_training.ipynb` gets its inference helper and training data corrected so training and inference use the same prompt format.

**Tech Stack:** Python 3, HuggingFace `transformers`, `peft` (LoRA), `pytest`, Jupyter notebook (`NotebookEdit`)

---

## File Map

| Action | File | What changes |
|--------|------|--------------|
| Modify | `config.py` | `LLM_MODEL_NAME` → TinyLlama; add `LORA_ADAPTER_PATH` |
| Modify | `llm_parser.py` | Fix `UNIFIED_SYSTEM_PROMPT`; load LoRA adapter after base model |
| Modify | `agent.py` | Prefs in `parse_unified` context; remove `pre_answer` shortcut |
| Create | `tests/test_agent_memory.py` | Unit tests for the two agent changes (mocked LLM + memory) |
| Modify | `lora_training.ipynb` | Fix `run_inference` prompt format; fix Section 6.5 data; add adapter save cell |

---

## Task 1: Update `config.py`

**Files:**
- Modify: `config.py`

- [ ] **Step 1: Add TinyLlama model name and adapter path**

Replace in `config.py`:
```python
# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_MODEL_NAME     = "Qwen/Qwen2.5-3B-Instruct"
LLM_GGUF_PATH      = "models/qwen2.5-3b-instruct-q3_k_m.gguf"
```
With:
```python
# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_MODEL_NAME     = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LLM_GGUF_PATH      = "models/qwen2.5-3b-instruct-q3_k_m.gguf"
LORA_ADAPTER_PATH  = "cathey_lora_adapter/final_adapter"
```

- [ ] **Step 2: Verify with a quick import check**

```bash
python3 -c "from config import LLM_MODEL_NAME, LORA_ADAPTER_PATH; print(LLM_MODEL_NAME, LORA_ADAPTER_PATH)"
```
Expected output:
```
TinyLlama/TinyLlama-1.1B-Chat-v1.0 cathey_lora_adapter/final_adapter
```

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "config: switch LLM to TinyLlama-1.1B, add LORA_ADAPTER_PATH"
```

---

## Task 2: Fix `UNIFIED_SYSTEM_PROMPT` in `llm_parser.py`

**Files:**
- Modify: `llm_parser.py` (lines 34–83)
- Test: `tests/test_agent_memory.py` (prompt content assertions — no model needed)

- [ ] **Step 1: Write failing test for prompt content**

Create `tests/test_agent_memory.py`:

```python
from llm_parser import UNIFIED_SYSTEM_PROMPT


def test_hello_is_not_invalid_example():
    """The system prompt must not teach the model that 'Hello' is invalid."""
    # The old wrong example was: Input: Hello.\nOutput: {"type":"invalid"}
    assert '"type":"invalid"' not in UNIFIED_SYSTEM_PROMPT.split("Hello")[1][:60]


def test_prompt_teaches_hello_as_general_qa():
    assert "general_qa" in UNIFIED_SYSTEM_PROMPT
    assert "greetings" in UNIFIED_SYSTEM_PROMPT.lower()


def test_prompt_has_lower_temperature_example():
    assert "lower the temperature" in UNIFIED_SYSTEM_PROMPT.lower()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest tests/test_agent_memory.py::test_hello_is_not_invalid_example tests/test_agent_memory.py::test_prompt_teaches_hello_as_general_qa tests/test_agent_memory.py::test_prompt_has_lower_temperature_example -v
```
Expected: 3 FAILs

- [ ] **Step 3: Fix `UNIFIED_SYSTEM_PROMPT` in `llm_parser.py`**

In `UNIFIED_SYSTEM_PROMPT` (around line 34), make these changes:

**a. Replace the rules block** — change the `general_qa` and `invalid` rule lines from:
```
- general_qa: anything unrelated to device control, including greetings and personal questions.
- invalid: unintelligible or empty input only.
```
To:
```
- general_qa: anything unrelated to device control, including greetings ("hello", "hi", "thanks"), farewells, personal questions, and conversational filler.
- invalid: unintelligible sounds or completely empty input only. "hello", "never mind", "okay" are NOT invalid.
```

**b. Replace the last example block** — change from:
```
Input: Cathey, how do I eat an apple?
Output: {"type":"general_qa","answer":"Wash it first, then eat it."}

Input: Cathey, my name is Alex.
Output: {"type":"general_qa","answer":"Nice to meet you, Alex!"}

Input: Hello.
Output: {"type":"invalid"}
```
To:
```
Input: Cathey, how do I eat an apple?
Output: {"type":"general_qa","answer":"Wash it first, then eat it."}

Input: Cathey, my name is Alex.
Output: {"type":"general_qa","answer":"Nice to meet you, Alex!"}

Input: Cathey, hello!
Output: {"type":"general_qa","answer":"Hello! How can I help you?"}

Input: Cathey, lower the temperature.
Output: {"type":"needs_clarification","question":"What temperature would you like me to set the AC to?","options":["lower_ac_temperature","raise_ac_temperature"]}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_agent_memory.py::test_hello_is_not_invalid_example tests/test_agent_memory.py::test_prompt_teaches_hello_as_general_qa tests/test_agent_memory.py::test_prompt_has_lower_temperature_example -v
```
Expected: 3 PASSes

- [ ] **Step 5: Commit**

```bash
git add llm_parser.py tests/test_agent_memory.py
git commit -m "fix: correct UNIFIED_SYSTEM_PROMPT — hello→general_qa, add lower-temp example"
```

---

## Task 3: Inject prefs into `parse_unified` context (`agent.py`)

**Files:**
- Modify: `agent.py` (`_handle_new_request`, lines ~194–202)
- Test: `tests/test_agent_memory.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_agent_memory.py`:

```python
from unittest.mock import MagicMock, call
from agent import CatheyAgent


def _make_agent(parse_result, qa_result=""):
    llm = MagicMock()
    llm.parse_unified.return_value = (parse_result, "", 100.0)
    llm.answer_qa.return_value = (qa_result, 50.0)

    memory = MagicMock()
    memory.episodes.count.return_value = 0
    memory.prefs = {"user_name": "Alex"}
    memory.build_context.return_value = "## User preferences\n- user_name: Alex"
    memory.skills = []

    speak = MagicMock()
    return CatheyAgent(llm=llm, memory=memory, speak=speak, gpio=None), llm, memory, speak


def test_prefs_included_in_parse_unified_context():
    """parse_unified must receive context that contains current user prefs."""
    agent, llm, memory, speak = _make_agent(
        {"type": "general_qa", "answer": "Your name is Alex."}, "Your name is Alex."
    )
    agent.handle("Cathey, what's my name?", verbose=False)

    _, kwargs = llm.parse_unified.call_args
    context = kwargs.get("context", "")
    assert "user_name" in context
    assert "Alex" in context
```

- [ ] **Step 2: Run to verify it fails**

```bash
python3 -m pytest tests/test_agent_memory.py::test_prefs_included_in_parse_unified_context -v
```
Expected: FAIL — context does not contain "user_name"

- [ ] **Step 3: Implement the fix in `agent.py:_handle_new_request`**

Find the block (around line 194):
```python
        context = ""
        if self._memory.episodes.count() > 0:
            eps = self._memory.retrieve_episodes(text, n=2)
            ep_lines = [
                f"Previously: user said \"{ep['text']}\", you replied \"{ep['meta']['cathey_reply']}\""
                for ep in eps if ep["distance"] < 0.6
            ]
            context = "\n".join(ep_lines)
        semantic, _, ms = self._llm.parse_unified(text, context=context, verbose=verbose)
```

Replace with:
```python
        context = ""
        if self._memory.episodes.count() > 0:
            eps = self._memory.retrieve_episodes(text, n=2)
            ep_lines = [
                f"Previously: user said \"{ep['text']}\", you replied \"{ep['meta']['cathey_reply']}\""
                for ep in eps if ep["distance"] < 0.6
            ]
            context = "\n".join(ep_lines)
        if self._memory.prefs:
            pref_lines = [f"- {k}: {v}" for k, v in self._memory.prefs.items()]
            pref_block = "\n".join(pref_lines)
            context = f"{context}\n{pref_block}".strip() if context else pref_block
        semantic, _, ms = self._llm.parse_unified(text, context=context, verbose=verbose)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python3 -m pytest tests/test_agent_memory.py::test_prefs_included_in_parse_unified_context -v
```
Expected: PASS

- [ ] **Step 5: Run all tests to check no regression**

```bash
python3 -m pytest tests/test_rule_based.py tests/test_agent_memory.py -v
```
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add agent.py tests/test_agent_memory.py
git commit -m "fix: include user prefs in parse_unified context for memory-aware classification"
```

---

## Task 4: Remove `pre_answer` shortcut — always call `answer_qa` (`agent.py`)

**Files:**
- Modify: `agent.py` (`_do_general_qa` and its caller in `_handle_new_request`)
- Test: `tests/test_agent_memory.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_agent_memory.py`:

```python
def test_general_qa_always_calls_answer_qa():
    """answer_qa must always be called for general_qa, never using the classification answer directly."""
    agent, llm, memory, speak = _make_agent(
        {"type": "general_qa", "answer": "classification-side answer"},
        "memory-aware answer"
    )
    agent.handle("Cathey, what is your name?", verbose=False)

    llm.answer_qa.assert_called_once()
    speak.assert_called_once_with("memory-aware answer")
```

- [ ] **Step 2: Run to verify it fails**

```bash
python3 -m pytest tests/test_agent_memory.py::test_general_qa_always_calls_answer_qa -v
```
Expected: FAIL — `speak` is called with "classification-side answer", not "memory-aware answer"

- [ ] **Step 3: Update `_do_general_qa` signature and body**

Find `_do_general_qa` (around line 305):
```python
    def _do_general_qa(self, text, ms, verbose, pre_answer: str = "") -> Dict[str, Any]:
        if pre_answer:
            answer = pre_answer
            qa_ms = 0.0
        else:
            context = self._memory.build_context(text)
            answer, qa_ms = self._llm.answer_qa(text, context, verbose=verbose)
```

Replace with:
```python
    def _do_general_qa(self, text, ms, verbose) -> Dict[str, Any]:
        context = self._memory.build_context(text)
        answer, qa_ms = self._llm.answer_qa(text, context, verbose=verbose)
```

- [ ] **Step 4: Update the caller in `_handle_new_request`**

Find (around line 211):
```python
        if semantic["type"] == "general_qa":
            return self._do_general_qa(text, ms, verbose, pre_answer=semantic.get("answer", ""))
```

Replace with:
```python
        if semantic["type"] == "general_qa":
            return self._do_general_qa(text, ms, verbose)
```

- [ ] **Step 5: Run all tests**

```bash
python3 -m pytest tests/test_rule_based.py tests/test_agent_memory.py -v
```
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add agent.py tests/test_agent_memory.py
git commit -m "fix: remove pre_answer shortcut — general_qa always calls answer_qa with full memory context"
```

---

## Task 5: Load LoRA adapter in `llm_parser.py`

**Files:**
- Modify: `llm_parser.py` (imports + `__init__`, transformers backend block)

The adapter directory may not exist during local dev (only after Colab training). The loading is guarded by `os.path.isdir`, so the system works without an adapter and silently loads one when present.

- [ ] **Step 1: Add `import os` and `LORA_ADAPTER_PATH` import**

At the top of `llm_parser.py`, the existing imports include `from config import ...`. Add `os` and `LORA_ADAPTER_PATH`:

```python
import os
```
(Add after the existing `import re` line.)

And in the `from config import ...` line, add `LORA_ADAPTER_PATH`:
```python
from config import LLM_BACKEND, LLM_MODEL_NAME, LLM_GGUF_PATH, LLM_DEVICE, LLM_DTYPE, LLM_MAX_NEW_TOKENS, LORA_ADAPTER_PATH
```

- [ ] **Step 2: Add LoRA loading after base model setup**

In `LLMParser.__init__`, find the transformers backend block that ends with:
```python
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.eval()
            self._llama = None
```

Replace with:
```python
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if os.path.isdir(LORA_ADAPTER_PATH):
                from peft import PeftModel
                print(f"Loading LoRA adapter from {LORA_ADAPTER_PATH} ...")
                self.model = PeftModel.from_pretrained(self.model, LORA_ADAPTER_PATH)
                print("LoRA adapter loaded.")
            self.model.eval()
            self._llama = None
```

- [ ] **Step 3: Verify import works (no model download)**

```bash
python3 -c "import ast, sys; ast.parse(open('llm_parser.py').read()); print('syntax OK')"
```
Expected: `syntax OK`

- [ ] **Step 4: Smoke test in `dev_debug.ipynb`**

In the notebook, re-run cell 2 (Setup & Imports) and cell 3 (Load models). With no adapter directory present it should print:
```
Loading LLM (TinyLlama/TinyLlama-1.1B-Chat-v1.0) on mps [torch.float16] ...
LLM ready.
```
(No "LoRA adapter" line — expected since adapter not yet trained.)

- [ ] **Step 5: Commit**

```bash
git add llm_parser.py
git commit -m "feat: load LoRA adapter in llm_parser when cathey_lora_adapter/final_adapter exists"
```

---

## Task 6: Fix `lora_training.ipynb`

**Files:**
- Modify: `lora_training.ipynb` — three cells changed, one cell added

### 6a: Fix `run_inference` to use `apply_chat_template`

- [ ] **Step 1: Replace the `run_inference` function (cell `d1d84058`)**

The current function builds a plain-text prompt (`f"{SYSTEM_PROMPT}\n\nInput: {test_input}\nOutput:"`). This does not match the training format, causing garbage outputs. Replace the entire cell with:

```python
def run_inference(test_input: str, max_new_tokens: int = 80):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": test_input},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_length = inputs["input_ids"].shape[1]

    stopping_criteria = StoppingCriteriaList([
        JsonStopOnComplete(tokenizer, prompt_length)
    ])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
            use_cache=True,
        )

    generated_ids = outputs[0][prompt_length:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    json_str = extract_first_json_object(raw_output)
    return {"raw_output": raw_output, "first_json": json_str}
```

### 6b: Fix Section 6.5 training data

- [ ] **Step 2: Replace Section 6.5 cell (`6b1bc753`) with corrected data**

Replace the entire cell with:

```python
# Section 6.5: 补充/修正 general_qa 和 invalid 训练样本

import json

EXTRA_TRAIN_EXAMPLES = [
    # general_qa — greetings and personal questions (NOT invalid)
    {"input": "Cathey, hello!", "output": '{"type":"general_qa","answer":"Hello! How can I help you?"}'},
    {"input": "Cathey, never mind.", "output": '{"type":"general_qa","answer":"No problem, just let me know if you need anything."}'},
    {"input": "Cathey, okay thanks.", "output": '{"type":"general_qa","answer":"You\'re welcome!"}'},
    # general_qa — knowledge questions
    {"input": "Cathey, how do I eat an apple?", "output": '{"type":"general_qa","answer":"Wash it first, then eat it."}'},
    {"input": "Cathey, what time is it?", "output": '{"type":"general_qa","answer":"I don\'t have access to a clock, but you can check your phone."}'},
    {"input": "Cathey, can I eat leftovers after 3 days?", "output": '{"type":"general_qa","answer":"Yes, most cooked food is safe for 3-4 days in the fridge."}'},
    {"input": "Cathey, what\'s the weather like?", "output": '{"type":"general_qa","answer":"I don\'t have weather data, but you can check a weather app."}'},
    {"input": "Cathey, how do I boil an egg?", "output": '{"type":"general_qa","answer":"Place the egg in boiling water for 6-12 minutes depending on your preference."}'},
    {"input": "Cathey, what is 2 plus 2?", "output": '{"type":"general_qa","answer":"4."}'},
    # needs_clarification — vague temperature request
    {"input": "Cathey, lower the temperature.", "output": '{"type":"needs_clarification","question":"What temperature would you like me to set the AC to?","options":["lower_ac_temperature","raise_ac_temperature"]}'},
    {"input": "Cathey, make it warmer.", "output": '{"type":"needs_clarification","question":"Would you like me to raise the AC temperature or close the window?","options":["raise_ac_temperature","close_window"]}'},
    # invalid — truly unintelligible only
    {"input": "Hmm.", "output": '{"type":"invalid"}'},
    {"input": "Just testing.", "output": '{"type":"invalid"}'},
]

EXTRA_VAL_EXAMPLES = [
    {"input": "Cathey, how long should I microwave rice?", "output": '{"type":"general_qa","answer":"About 2-3 minutes, stirring halfway through."}'},
    {"input": "Cathey, hello there!", "output": '{"type":"general_qa","answer":"Hello! How can I help?"}'},
    {"input": "Cathey, raise the temperature a little.", "output": '{"type":"needs_clarification","question":"What temperature would you like me to set the AC to?","options":["lower_ac_temperature","raise_ac_temperature"]}'},
    {"input": "Hmm hmm.", "output": '{"type":"invalid"}'},
]

with open(TRAIN_FILE, "a", encoding="utf-8") as f:
    for ex in EXTRA_TRAIN_EXAMPLES:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

with open(VAL_FILE, "a", encoding="utf-8") as f:
    for ex in EXTRA_VAL_EXAMPLES:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Appended {len(EXTRA_TRAIN_EXAMPLES)} train examples, {len(EXTRA_VAL_EXAMPLES)} val examples")
```

### 6c: Add adapter save cell after training

- [ ] **Step 3: Add a new cell after the `trainer.train()` cell (`adbfb8e4`) to save the adapter**

Add a new cell:

```python
# Section 15b: 保存 LoRA adapter 和 tokenizer

import shutil, os

# Save LoRA adapter weights
trainer.model.save_pretrained(FINAL_ADAPTER_DIR)
tokenizer.save_pretrained(FINAL_ADAPTER_DIR)
print(f"Adapter saved to {FINAL_ADAPTER_DIR}")

# Verify saved files exist
expected = ["adapter_config.json", "adapter_model.safetensors"]
for f in expected:
    path = os.path.join(FINAL_ADAPTER_DIR, f)
    status = "✓" if os.path.exists(path) else "✗ MISSING"
    print(f"  {status}  {f}")
```

- [ ] **Step 4: Commit notebook**

```bash
git add lora_training.ipynb
git commit -m "fix: lora_training — apply_chat_template inference, corrected data labels, add adapter save cell"
```

---

## Task 7: End-to-end verification in `dev_debug.ipynb`

No code changes — verification only.

- [ ] **Step 1: Restart kernel and run all cells top-to-bottom**

- [ ] **Step 2: Verify bug fixes in Section 8 (Full Agent Pipeline)**

Add/run these test calls and confirm expected outputs:

```python
# Bug 1 fix — Hello → general_qa
run("Cathey, hello!")
# Expected: result["reason"] == "general_qa_answered"

# Bug 2 fix — memory recall
run("Cathey, my name is Alex.")
run("Cathey, what's my name?")
# Expected: second result["spoken_reply"] contains "Alex"

# Bug 3 — lower the temperature (already handled by rescue path; LLM may now classify directly)
run("Cathey, lower the temperature.")
# Expected: result["reason"] in ("clarification_requested", "clarification_requested")
```

- [ ] **Step 3: Run full test suite**

```bash
python3 -m pytest tests/test_rule_based.py tests/test_agent_memory.py -v
```
Expected: all PASS

- [ ] **Step 4: Commit verification notebook state**

```bash
git add dev_debug.ipynb
git commit -m "test: verify all three bug fixes in dev_debug notebook"
```
