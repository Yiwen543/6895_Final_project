# Nova Embedded V2 Design — Spec Update

**Date:** 2026-04-27
**Base:** `2026-04-26-nova-pi5-deployment-design.md` (all unchanged sections still apply)
**Scope:** Three targeted changes to make the embedded branch deployable on Pi 5: GGUF inference backend, optional chromadb, and config/pinout alignment with current code.

---

## Change 1: LLM Inference — dual backend (transformers + llama-cpp-python)

### Problem
`bitsandbytes` (used for int4 NF4 quantization) depends on CUDA and x86 — it will not compile on Pi 5 ARM64. Without quantization, the 3B model is ~6 GB float16 and cannot fit in Pi 5 memory.

### Solution
Add a `LLM_BACKEND` switch in `config.py`. Two backends coexist in `llm_parser.py`:

| Backend | When | Model format | Library |
|---------|------|-------------|---------|
| `transformers` | Mac dev (MPS/CUDA) | HuggingFace safetensors | `transformers` |
| `llama_cpp` | Pi 5 (CPU) | GGUF Q4_K_M | `llama-cpp-python` |

### config.py changes
```python
LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
LLM_GGUF_PATH  = "models/qwen2.5-3b-instruct-q4_k_m.gguf"

if torch.cuda.is_available():
    LLM_BACKEND      = "transformers"
    LLM_DEVICE        = "cuda"
    LLM_DTYPE         = torch.float16
elif torch.backends.mps.is_available():
    LLM_BACKEND      = "transformers"
    LLM_DEVICE        = "mps"
    LLM_DTYPE         = torch.float16
else:
    LLM_BACKEND      = "llama_cpp"
    LLM_DEVICE        = "cpu"
    LLM_DTYPE         = torch.float32
```

Remove `LLM_LOAD_IN_4BIT` — no longer needed (GGUF replaces bitsandbytes).

### llm_parser.py changes
- Keep all existing code (system prompts, JSON parsing, `_generate_json`, public API).
- `LLMParser.__init__` checks `LLM_BACKEND`:
  - `"transformers"`: current behavior (AutoModelForCausalLM, unchanged).
  - `"llama_cpp"`: load `Llama(model_path=LLM_GGUF_PATH, n_ctx=512, n_threads=4)`.
- `_generate_json` dispatches to the appropriate backend:
  - `"transformers"`: current `model.generate()` path.
  - `"llama_cpp"`: `llama.create_chat_completion(messages, max_tokens, temperature=0)`, extract content string.
- The public API (`parse_unified`, `resolve_followup`, `answer_qa`) remains identical.

### GGUF model file
- File: `qwen2.5-3b-instruct-q4_k_m.gguf` (~2 GB)
- Source: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF (official Qwen quantized)
- Location on Pi: `~/nova/models/`
- `deploy.sh` downloads it if not present

### llama-cpp-python install on Pi 5
```bash
CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" \
pip3 install llama-cpp-python --break-system-packages
```
Requires `libopenblas-dev` (added to `deploy.sh` apt-get).

---

## Change 2: chromadb optional dependency (memory.py)

### Problem
`chromadb` fails to install on Pi 5 (heavy native dependencies). This causes `nova.py` to crash on import.

### Solution
Try importing chromadb; if unavailable, fall back to an in-memory list with manual cosine similarity search.

### Implementation
```python
try:
    import chromadb
    _HAS_CHROMADB = True
except ImportError:
    _HAS_CHROMADB = False
```

Two episodic memory backends:
- **chromadb available**: current `PersistentClient` behavior (unchanged).
- **chromadb unavailable**: `_InMemoryEpisodic` — a Python list of `{"id", "embedding", "document", "metadata"}` dicts. `add()` appends, `query()` computes cosine similarity via numpy, returns top-N. No disk persistence — memory resets on restart.

Semantic memory (user_prefs.json) and procedural memory (skills.json) are unaffected — they use plain JSON files.

### Performance impact
None. In-memory cosine search over <1000 entries is faster than chromadb's SQLite + HNSW overhead on Pi's SD card.

---

## Change 3: Config and pinout alignment

Updates to match current code state (already implemented, spec was stale):

### GPIO pins (already changed in gpio_executor.py)
| Device | Signal | GPIO | Reason |
|--------|--------|------|--------|
| Grove RGB LED | DATA | **24** (was 17) | GPIO 17 reserved by ReSpeaker HAT |
| ULN2003 | IN4 | **26** (was 19) | GPIO 19 reserved by ReSpeaker HAT |

### config.py values (already merged from main)
- `LLM_MODEL_NAME`: `"Qwen/Qwen2.5-3B-Instruct"` (was 1.5B)
- `LLM_MAX_NEW_TOKENS`: `160` (was 96)
- `PIPER_MODEL_PATH`: removed from config (Piper TTS path handled internally)

### llm_parser.py (already merged from main)
- `repetition_penalty=1.1` added to `model.generate()` — fixes LLM output repetition bug

### finetune/train_data.py (already merged from main)
- Expanded from ~50 to ~170 training samples

### nova.py bug fix needed
```python
# Current (broken):
memory = MemoryManager(embed)
# Fixed:
memory = MemoryManager(embed_fn=lambda text: embed.encode(text, convert_to_numpy=True).tolist())
```

### deploy.sh additions
- `apt-get install libopenblas-dev` (for llama-cpp-python)
- Download GGUF model file to `~/nova/models/` if not present
- `requirements_pi.txt`: add `llama-cpp-python`, remove `bitsandbytes`

---

## Out of Scope (unchanged from v1)

- LoRA adapter loading (GGUF models can have LoRA merged at conversion time)
- Real AC or window hardware
- Multi-room or networked device control
- Web UI or remote control interface

---

## Summary of files to modify

| File | Action | What changes |
|------|--------|-------------|
| `config.py` | Modify | Add `LLM_BACKEND`, `LLM_GGUF_PATH`; remove `LLM_LOAD_IN_4BIT` |
| `llm_parser.py` | Modify | Add llama-cpp-python backend alongside transformers |
| `memory.py` | Modify | Try/except chromadb import, add in-memory fallback |
| `nova.py` | Modify | Fix MemoryManager embed_fn signature |
| `deploy.sh` | Modify | Add libopenblas-dev, GGUF download step |
| `requirements_pi.txt` | Modify | Add llama-cpp-python, remove bitsandbytes |
