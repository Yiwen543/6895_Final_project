# Nova Embedded V2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the embedded branch deployable on Pi 5 by adding a GGUF/llama-cpp-python inference backend, making chromadb optional with an in-memory fallback, and fixing config/deploy alignment issues.

**Architecture:** `config.py` gains a `LLM_BACKEND` switch (`"transformers"` on Mac, `"llama_cpp"` on Pi). `llm_parser.py` dispatches to the appropriate backend while keeping the public API identical. `memory.py` try-imports chromadb and falls back to a simple in-memory episodic store. All other modules (`agent.py`, `audio.py`, `gpio_executor.py`, `schema.py`, `rule_based.py`) are untouched.

**Tech Stack:** Python 3.11, Qwen2.5-3B-Instruct (GGUF Q4_K_M), llama-cpp-python, transformers, faster-whisper, lgpio, piper-tts

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `config.py` | Modify | Add `LLM_BACKEND`, `LLM_GGUF_PATH`; remove `LLM_LOAD_IN_4BIT` |
| `llm_parser.py` | Modify | Add llama-cpp backend alongside transformers |
| `memory.py` | Modify | Try/except chromadb, add `_InMemoryEpisodic` fallback |
| `nova.py` | Modify | Fix `MemoryManager` embed_fn call signature |
| `deploy.sh` | Modify | Add libopenblas-dev, GGUF model download |
| `requirements_pi.txt` | Create | Pi-specific deps with llama-cpp-python |

---

### Task 1: config.py — add LLM_BACKEND switch, remove bitsandbytes config

**Files:**
- Modify: `config.py`

- [ ] **Step 1: Replace LLM config block**

Replace lines 1–27 of `config.py` with:

```python
import torch

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_MODEL_NAME     = "Qwen/Qwen2.5-3B-Instruct"
LLM_GGUF_PATH      = "models/qwen2.5-3b-instruct-q4_k_m.gguf"

if torch.cuda.is_available():
    LLM_BACKEND       = "transformers"
    LLM_DEVICE        = "cuda"
    LLM_DTYPE         = torch.float16
elif torch.backends.mps.is_available():
    LLM_BACKEND       = "transformers"
    LLM_DEVICE        = "mps"
    LLM_DTYPE         = torch.float16
else:
    LLM_BACKEND       = "llama_cpp"
    LLM_DEVICE        = "cpu"
    LLM_DTYPE         = torch.float32

LLM_MAX_NEW_TOKENS = 160
```

This removes `LLM_LOAD_IN_4BIT` and adds `LLM_BACKEND` + `LLM_GGUF_PATH`.

- [ ] **Step 2: Verify config imports cleanly**

```bash
cd /Users/ezslaptop/Projects/6895_Final_project && python3 -c "from config import LLM_BACKEND, LLM_GGUF_PATH; print('LLM_BACKEND:', LLM_BACKEND); print('LLM_GGUF_PATH:', LLM_GGUF_PATH)"
```

Expected: `LLM_BACKEND: transformers` and `LLM_GGUF_PATH: models/qwen2.5-3b-instruct-q4_k_m.gguf` (Mac has MPS so backend is transformers).

- [ ] **Step 3: Commit**

```bash
git add config.py && git commit -m "feat: add LLM_BACKEND switch, remove bitsandbytes config"
```

---

### Task 2: llm_parser.py — add llama-cpp-python backend

**Files:**
- Modify: `llm_parser.py`

- [ ] **Step 1: Replace imports and config import line**

Replace lines 1–26 (module docstring + imports) with:

```python
"""
LLM module: loads a configurable instruction-tuned model (default: Qwen2.5-3B-Instruct)
and exposes three inference methods:
  - parse_unified()    classify user text → direct_command / needs_clarification / general_qa / invalid
  - resolve_followup() resolve user reply to a clarification question → direct_command / invalid
  - answer_qa()        generate a plain-text answer using RAG context

On Mac (MPS/CUDA): uses HuggingFace transformers with float16.
On Pi 5 (CPU): uses llama-cpp-python with a GGUF Q4_K_M quantized model.
Backend is selected via LLM_BACKEND in config.py.
"""

import re
import json
import time
import torch
from typing import Any, Dict, List, Optional, Tuple

from config import LLM_BACKEND, LLM_MODEL_NAME, LLM_GGUF_PATH, LLM_DEVICE, LLM_DTYPE, LLM_MAX_NEW_TOKENS

if LLM_BACKEND == "transformers":
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        StoppingCriteria,
        StoppingCriteriaList,
    )
else:
    from llama_cpp import Llama
```

- [ ] **Step 2: Replace `_JsonStop` class with a guard**

Replace the `_JsonStop` class (lines ~150-159 after old numbering) with:

```python
# ── Stopping criterion (transformers backend only) ───────────────────────────

if LLM_BACKEND == "transformers":
    class _JsonStop(StoppingCriteria):
        def __init__(self, tokenizer, prompt_length: int):
            self._tokenizer = tokenizer
            self._prompt_len = prompt_length

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            gen = input_ids[0][self._prompt_len:]
            text = self._tokenizer.decode(gen, skip_special_tokens=True)
            return LLMParser._has_complete_json(text)
```

- [ ] **Step 3: Replace `LLMParser.__init__`**

Replace the current `__init__` method (from `def __init__` through `print("LLM ready.")`) with:

```python
    def __init__(self, model_name: str = LLM_MODEL_NAME, dtype=LLM_DTYPE):
        self._backend = LLM_BACKEND

        if self._backend == "llama_cpp":
            print(f"Loading LLM via llama.cpp ({LLM_GGUF_PATH}) ...")
            self._llama = Llama(
                model_path=LLM_GGUF_PATH,
                n_ctx=512,
                n_threads=4,
                verbose=False,
            )
            self.tokenizer = None
            self.model = None
        else:
            print(f"Loading LLM ({model_name}) on {LLM_DEVICE} [{dtype}] ...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if LLM_DEVICE == "mps":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=dtype
                ).to("mps")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=dtype, device_map="auto"
                )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.eval()
            self._llama = None
        print("LLM ready.")
```

- [ ] **Step 4: Replace `_generate_json` method**

Replace the entire `_generate_json` method with:

```python
    def _generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = LLM_MAX_NEW_TOKENS,
        verbose: bool = False,
    ) -> Tuple[Dict[str, Any], str, float]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        if self._backend == "llama_cpp":
            t0 = time.perf_counter()
            resp = self._llama.create_chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0,
                repeat_penalty=1.1,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            raw = resp["choices"][0]["message"]["content"].strip()
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            plen = inputs["input_ids"].shape[1]

            stop = StoppingCriteriaList([_JsonStop(self.tokenizer, plen)])
            t0 = time.perf_counter()
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    use_cache=True,
                    stopping_criteria=stop,
                )
            latency_ms = (time.perf_counter() - t0) * 1000
            raw = self.tokenizer.decode(out[0][plen:], skip_special_tokens=True).strip()

        if verbose:
            print("Raw output:", raw)
            print(f"Latency: {latency_ms:.1f} ms")

        json_str = self._extract_first_json(raw)
        if json_str is None:
            return {"type": "invalid"}, raw, latency_ms
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            return {"type": "invalid"}, raw, latency_ms
        return self._normalize(parsed), raw, latency_ms
```

- [ ] **Step 5: Replace `answer_qa` method**

Replace the entire `answer_qa` method with:

```python
    def answer_qa(
        self,
        question: str,
        context: str = "",
        max_new_tokens: int = 80,
        verbose: bool = False,
    ) -> Tuple[str, float]:
        user_content = f"{context}\n\nUser: {question}" if context.strip() else f"User: {question}"
        messages = [
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

        if self._backend == "llama_cpp":
            t0 = time.perf_counter()
            resp = self._llama.create_chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            answer = resp["choices"][0]["message"]["content"].strip()
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            plen = inputs["input_ids"].shape[1]

            t0 = time.perf_counter()
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
            latency_ms = (time.perf_counter() - t0) * 1000
            answer = self.tokenizer.decode(out[0][plen:], skip_special_tokens=True).strip()

        if verbose:
            print("[QA answer]", answer)
        return answer, latency_ms
```

- [ ] **Step 6: Verify imports resolve on Mac (transformers backend)**

```bash
cd /Users/ezslaptop/Projects/6895_Final_project && python3 -c "from llm_parser import LLMParser; print('Import OK')"
```

Expected: `Import OK`

- [ ] **Step 7: Commit**

```bash
git add llm_parser.py && git commit -m "feat: dual LLM backend — transformers (Mac) + llama-cpp-python (Pi)"
```

---

### Task 3: memory.py — optional chromadb with in-memory fallback

**Files:**
- Modify: `memory.py`

- [ ] **Step 1: Replace chromadb import with try/except and add fallback class**

Replace lines 1–17 (docstring + imports through `from config import`) with:

```python
"""
MemoryManager: four-layer memory architecture for Nova.

  working    — RAM deque, current session only, destroyed on clear_working()
  episodic   — ChromaDB vector store (if available), else in-memory list
  semantic   — JSON file, structured user preferences (e.g. preferred AC temp)
  procedural — JSON file, successful trigger→action patterns learned over time
"""

import json
import uuid
import numpy as np
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

try:
    import chromadb
    _HAS_CHROMADB = True
except ImportError:
    _HAS_CHROMADB = False

from config import (
    MEMORY_DIR,
    WORKING_MAXLEN,
    SKILL_SIM_THRESHOLD,
    EPISODE_DIST_CUTOFF,
)


class _InMemoryEpisodic:
    """Fallback episodic store when chromadb is not available."""

    def __init__(self):
        self._entries: List[Dict] = []

    def count(self) -> int:
        return len(self._entries)

    def add(self, ids, embeddings, documents, metadatas):
        for eid, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
            self._entries.append({
                "id": eid,
                "embedding": np.array(emb, dtype=np.float32),
                "document": doc,
                "metadata": meta,
            })

    def query(self, query_embeddings, n_results, include=None):
        if not self._entries:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        q = np.array(query_embeddings[0], dtype=np.float32)
        q_norm = np.linalg.norm(q) + 1e-9
        scored = []
        for e in self._entries:
            v = e["embedding"]
            cos_dist = 1.0 - float(np.dot(q, v) / (q_norm * (np.linalg.norm(v) + 1e-9)))
            scored.append((cos_dist, e))
        scored.sort(key=lambda x: x[0])
        top = scored[:n_results]
        return {
            "documents": [[t[1]["document"] for t in top]],
            "metadatas": [[t[1]["metadata"] for t in top]],
            "distances": [[t[0] for t in top]],
        }
```

- [ ] **Step 2: Replace `MemoryManager.__init__` episodic setup**

Replace the `__init__` method of `MemoryManager` (from `def __init__` through `self.skills: List[Dict]` block) with:

```python
    def __init__(
        self,
        embed_fn: Callable[[str], List[float]],
        persist_dir: str = MEMORY_DIR,
        working_maxlen: int = WORKING_MAXLEN,
    ):
        self._embed = embed_fn
        Path(persist_dir).mkdir(exist_ok=True)

        # Working memory — session RAM only
        self.working: deque = deque(maxlen=working_maxlen)

        # Episodic memory — ChromaDB or in-memory fallback
        if _HAS_CHROMADB:
            self._chroma = chromadb.PersistentClient(path=f"{persist_dir}/chroma")
            self.episodes = self._chroma.get_or_create_collection(
                name="episodes", metadata={"hnsw:space": "cosine"}
            )
        else:
            self.episodes = _InMemoryEpisodic()

        # Semantic memory — JSON
        self._prefs_path = Path(f"{persist_dir}/user_prefs.json")
        self.prefs: Dict[str, Any] = (
            json.loads(self._prefs_path.read_text())
            if self._prefs_path.exists()
            else {}
        )

        # Procedural memory — JSON
        self._skills_path = Path(f"{persist_dir}/skills.json")
        self.skills: List[Dict] = (
            json.loads(self._skills_path.read_text())
            if self._skills_path.exists()
            else []
        )
```

The rest of `MemoryManager` (all other methods) stays exactly as-is — `self.episodes` has the same `.add()`, `.query()`, `.count()` interface in both backends.

- [ ] **Step 3: Verify import without chromadb**

```bash
cd /Users/ezslaptop/Projects/6895_Final_project && python3 -c "
import sys
sys.modules['chromadb'] = None  # simulate missing chromadb
from memory import _HAS_CHROMADB
print('_HAS_CHROMADB:', _HAS_CHROMADB)
"
```

Expected: `_HAS_CHROMADB: False`

- [ ] **Step 4: Verify import with chromadb**

```bash
cd /Users/ezslaptop/Projects/6895_Final_project && python3 -c "from memory import _HAS_CHROMADB; print('_HAS_CHROMADB:', _HAS_CHROMADB)"
```

Expected: `_HAS_CHROMADB: True`

- [ ] **Step 5: Commit**

```bash
git add memory.py && git commit -m "feat: optional chromadb with in-memory episodic fallback"
```

---

### Task 4: nova.py — fix MemoryManager embed_fn signature

**Files:**
- Modify: `nova.py`

- [ ] **Step 1: Fix the MemoryManager call**

In `nova.py`, replace:

```python
    memory = MemoryManager(embed)
```

with:

```python
    memory = MemoryManager(embed_fn=lambda text: embed.encode(text, convert_to_numpy=True).tolist())
```

- [ ] **Step 2: Verify nova.py imports resolve**

```bash
cd /Users/ezslaptop/Projects/6895_Final_project && python3 -c "
import sys
from unittest.mock import MagicMock
sys.modules['lgpio'] = MagicMock()
sys.modules['piper'] = MagicMock()
sys.modules['piper.voice'] = MagicMock()
import nova
print('nova.py imports OK')
"
```

Expected: `nova.py imports OK`

- [ ] **Step 3: Commit**

```bash
git add nova.py && git commit -m "fix: MemoryManager embed_fn signature in nova.py"
```

---

### Task 5: requirements_pi.txt + deploy.sh — GGUF model download and dependencies

**Files:**
- Create: `requirements_pi.txt`
- Modify: `deploy.sh`

- [ ] **Step 1: Create requirements_pi.txt**

```
faster-whisper
transformers
sentencepiece
sentence-transformers
sounddevice
soundfile
numpy
piper-tts
llama-cpp-python
```

- [ ] **Step 2: Add libopenblas-dev and GGUF download to deploy.sh**

In `deploy.sh`, add `libopenblas-dev` to the apt-get line. Replace:

```bash
echo "==> Installing system dependencies"
ssh "$PI_HOST" "sudo apt-get install -y --no-install-recommends \
    python3-pip libportaudio2 libasound2-dev bluetooth bluez"
```

with:

```bash
echo "==> Installing system dependencies"
ssh "$PI_HOST" "sudo apt-get install -y --no-install-recommends \
    python3-pip libportaudio2 libasound2-dev bluetooth bluez libopenblas-dev"
```

Then add a new step after the pip install block. After:

```bash
echo "==> Installing Python dependencies"
ssh "$PI_HOST" "pip3 install --break-system-packages -r ~/nova/requirements_pi.txt"
```

add:

```bash
echo "==> Installing llama-cpp-python with OpenBLAS"
ssh "$PI_HOST" "CMAKE_ARGS='-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS' \
    pip3 install llama-cpp-python --break-system-packages --no-cache-dir"

echo "==> Downloading GGUF model (if needed, ~2GB)"
ssh "$PI_HOST" "mkdir -p ~/nova/models && \
    [ -f ~/nova/models/qwen2.5-3b-instruct-q4_k_m.gguf ] || \
    wget -q --show-progress \
      -O ~/nova/models/qwen2.5-3b-instruct-q4_k_m.gguf \
      'https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf'"
```

- [ ] **Step 3: Syntax-check deploy.sh**

```bash
bash -n /Users/ezslaptop/Projects/6895_Final_project/deploy.sh && echo "Syntax OK"
```

Expected: `Syntax OK`

- [ ] **Step 4: Commit**

```bash
git add requirements_pi.txt deploy.sh && git commit -m "feat: add llama-cpp-python deps and GGUF model download to deploy"
```

---

### Task 6: Add models/ to .gitignore

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add models/ directory to .gitignore**

Append to `.gitignore`:

```
# GGUF model files (too large for git)
models/
```

- [ ] **Step 2: Commit**

```bash
git add .gitignore && git commit -m "chore: add models/ to .gitignore"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Covered by |
|-----------------|-----------|
| `LLM_BACKEND` switch in config.py | Task 1 |
| `LLM_GGUF_PATH` in config.py | Task 1 |
| Remove `LLM_LOAD_IN_4BIT` | Task 1 |
| llama-cpp-python backend in `__init__` | Task 2 |
| `_generate_json` dispatches to correct backend | Task 2 |
| `answer_qa` dispatches to correct backend | Task 2 |
| Public API unchanged | Task 2 (parse_unified/resolve_followup/answer_qa signatures identical) |
| Try/except chromadb import | Task 3 |
| `_InMemoryEpisodic` fallback class | Task 3 |
| `.add()` / `.query()` / `.count()` interface match | Task 3 |
| Fix `MemoryManager(embed)` → `MemoryManager(embed_fn=...)` | Task 4 |
| `deploy.sh` adds libopenblas-dev | Task 5 |
| `deploy.sh` downloads GGUF model | Task 5 |
| `requirements_pi.txt` has llama-cpp-python | Task 5 |
| GPIO pin update (DATA=24, IN4=26) | Already done in code, no task needed |
| `repetition_penalty=1.1` | Already done in code, preserved in Task 2 |
| Train data expansion | Already done, no task needed |

**Placeholder scan:** No TBD/TODO found. All steps have complete code.

**Type consistency:**
- `LLMParser.__init__` — Task 2 removes `load_in_4bit` param, matches Task 1 removing `LLM_LOAD_IN_4BIT` ✅
- `_InMemoryEpisodic.add(ids, embeddings, documents, metadatas)` — matches `chromadb` collection API used in `memory.py` ✅
- `_InMemoryEpisodic.query(query_embeddings, n_results, include)` — matches return format `{"documents": [[...]], ...}` ✅
- `MemoryManager(embed_fn=...)` — Task 4 passes a callable, matches `__init__(self, embed_fn: Callable)` ✅
