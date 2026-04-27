# Nova — Offline Smart Home Assistant

EECS 6895 Final Project · Columbia University

Nova is a fully offline voice assistant deployable on a **Raspberry Pi 5**.  
It listens for the wake word "Nova", classifies user intent into 4 categories, executes smart home commands, and learns user preferences over time using a four-layer memory system.

---

## Project Structure

```
6895_Final_project/
├── main.ipynb              # Entry point — run the full assistant
├── config.py               # All constants (model names, audio params, memory thresholds)
├── schema.py               # Device schema, command validation, execution
├── llm_parser.py           # LLM loading + 3 inference methods
├── memory.py               # Four-layer memory system (working/episodic/semantic/procedural)
├── agent.py                # NovaAgent — intent routing + dialogue state machine
├── audio.py                # STT (Whisper) + TTS (pyttsx3) + VAD audio listener
├── finetune/
│   ├── train_data.py       # Labelled training data (input → JSON pairs)
│   └── lora_finetune.ipynb # LoRA fine-tuning notebook
└── tests/
    ├── text_test.ipynb     # Text-level tests + batch evaluation
    └── audio_test.ipynb    # Microphone tests + continuous VAD loop
```

---

## Intent Categories

| Category | Trigger | Example |
|---|---|---|
| `direct_command` | Device + explicit action | "Nova, turn on the light." |
| `needs_clarification` | Vague feeling / preference | "Nova, it's a bit dark." |
| `general_qa` | Non-device question | "Nova, how do I eat an apple?" |
| `invalid` | No "Nova" wake word | "Turn on the light." |

---

## Four-Layer Memory

| Layer | Storage | Lifetime | Purpose |
|---|---|---|---|
| Working | RAM `deque` | Current session | Conversation window context |
| Episodic | ChromaDB (local) | Persistent | RAG retrieval of similar past interactions |
| Semantic | JSON file | Persistent | User preferences (e.g. preferred AC temp) |
| Procedural | JSON file | Persistent | Learned trigger → action patterns (skip re-asking) |

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/Yiwen543/6895_Final_project.git
cd 6895_Final_project
```

### 2. Create a Python environment

```bash
python -m venv nova_env
source nova_env/bin/activate          # Windows: nova_env\Scripts\activate
```

### 3. Install dependencies

```bash
pip install numpy pandas sounddevice soundfile faster-whisper \
            transformers accelerate sentencepiece pyttsx3 \
            chromadb sentence-transformers torch bitsandbytes
```

### 4. Set your Hugging Face token

The default LLM (`Qwen2.5-3B-Instruct`) is a gated model. Log in once:

```bash
export HF_TOKEN=hf_your_token_here      # Mac/Linux
set HF_TOKEN=hf_your_token_here         # Windows
```

### 5. Run the assistant

Open `main.ipynb` in Jupyter and run all cells top-to-bottom:

```bash
jupyter notebook main.ipynb
```

**Cell order in `main.ipynb`:**

| Cell | What it does |
|---|---|
| 1 | Install dependencies (first run only) |
| 2 | HF authentication via `$HF_TOKEN` |
| 3 | Load LLM (`Qwen2.5-1.5B-Instruct`) |
| 4 | Load STT (Whisper `tiny.en`) + TTS (pyttsx3) |
| 5 | Load embedding model + initialize memory |
| 6 | Create `NovaAgent` |
| 7 | Quick text demo (4 sample inputs) |
| 8 | Start continuous audio loop *(uncomment to enable)* |

---

## Running Tests

### Text tests (no microphone needed)

```bash
jupyter notebook tests/text_test.ipynb
```

**What runs:**
- **Group A** — 10 isolated regression cases covering all 4 intent types + hard cases (colloquial inputs, food-safety questions)
- **Group B** — Stateful 4-step demo showing procedural memory learning: first time → clarification asked; second time → auto-resolved from memory
- **Batch evaluation** — Accuracy table + average latency

Each test resets dialogue state before running so cases are fully independent.

### Audio tests (microphone required)

```bash
jupyter notebook tests/audio_test.ipynb
```

**What runs:**
- **Cell 2** — Single 3-second recording → STT → agent → result
- **Cell 3** — Continuous VAD loop: speak naturally, Nova captures full utterances automatically. Press `Ctrl+C` to stop.

---

## LoRA Fine-tuning

Fine-tuning teaches the model to reliably output valid JSON and correctly classify hard cases that the base model gets wrong (e.g. colloquial complaints, food questions).

```bash
jupyter notebook finetune/lora_finetune.ipynb
```

**Step-by-step inside the notebook:**

| Cell | Action |
|---|---|
| 1 | Install `peft`, `trl`, `datasets` |
| 2 | Load base LLM + print class distribution of training data |
| 3 | Format data using the model's chat template |
| 4 | Configure LoRA (`r=8`, `alpha=16`, 7 target layers, ~0.44% trainable params) |
| 5 | Train with `SFTTrainer` (3 epochs, cosine LR schedule) |
| 6 | Save LoRA adapter → merge into base model |
| 7 | Evaluate on hard cases: before vs. after comparison |

**Expected training time:**

| Hardware | Time per epoch |
|---|---|
| CPU (Mac / RPi 5) | ~15–30 min |
| GPU (CUDA) | ~1–3 min |

**To add more training data**, edit `finetune/train_data.py` — add `(user_input, json_output)` tuples to `RAW_TRAIN_DATA`. Aim for 300–500 samples for production quality.

**To load a fine-tuned model** instead of the base model, replace Cell 3 in `main.ipynb`:

```python
# Load fine-tuned (merged) model
from llm_parser import LLMParser
llm = LLMParser(model_name="nova_lora_merged")
```

---

## Deployment on Raspberry Pi 5

### Transfer the project

```bash
rsync -av --exclude 'nova_env' --exclude '__pycache__' \
    6895_Final_project/ pi@raspberrypi.local:~/nova/
```

### On the Pi, install dependencies

```bash
pip install numpy sounddevice soundfile faster-whisper \
            transformers accelerate sentencepiece pyttsx3 \
            chromadb sentence-transformers torch bitsandbytes
```

### Quantization on Pi 5 (automatic)

When running on CPU (i.e., on the Pi), `LLM_LOAD_IN_4BIT=True` is set automatically in `config.py`.
The model loads with **NF4 4-bit quantization** via `bitsandbytes`, keeping memory under **3 GB** for any model below 4B parameters.

**Supported models** — change `LLM_MODEL_NAME` in `config.py` to switch:

| Model ID | Params | int4 RAM | Notes |
|---|---|---|---|
| `Qwen/Qwen2.5-3B-Instruct` | 3B | ~2 GB | **Default** — drop-in upgrade, best accuracy |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | ~0.9 GB | Original lightweight option |
| `google/gemma-2-2b-it` | 2B | ~1.2 GB | Fastest inference on Pi 5 |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | ~1.8 GB | Strong structured JSON output |
| `microsoft/Phi-3.5-mini-instruct` | 3.8B | ~2.4 GB | Highest quality, needs HF token |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7B | ~1 GB | Ultra-light fallback |

### Run on the Pi

```bash
export HF_TOKEN=hf_your_token_here
jupyter notebook main.ipynb
# or headless:
# python -c "from llm_parser import LLMParser; ..."
```

---

## Configuration Reference

All tunable parameters live in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `LLM_MODEL_NAME` | `Qwen/Qwen2.5-1.5B-Instruct` | Base LLM |
| `WHISPER_MODEL_SIZE` | `tiny.en` | STT model size (`tiny` / `base` / `small`) |
| `ENERGY_THRESHOLD` | `0.01` | VAD silence cutoff |
| `SKILL_SIM_THRESHOLD` | `0.82` | Procedural memory cosine similarity cutoff |
| `EPISODE_DIST_CUTOFF` | `0.6` | Episodic RAG relevance cutoff |
| `WORKING_MAXLEN` | `8` | Max turns kept in working memory |
| `LORA_R` | `8` | LoRA rank |
| `LORA_ALPHA` | `16` | LoRA scaling factor |

---

## How Each Module Connects

```
main.ipynb
    │
    ├── LLMParser (llm_parser.py)
    │       ├── parse_unified()       ← classify intent → JSON
    │       ├── resolve_followup()    ← resolve clarification reply → JSON
    │       └── answer_qa()          ← RAG-augmented plain-text answer
    │
    ├── MemoryManager (memory.py)
    │       ├── push_working()       ← append to session RAM
    │       ├── save_episode()       ← write to ChromaDB
    │       ├── update_pref()        ← write to user_prefs.json
    │       ├── record_skill()       ← write to skills.json
    │       ├── lookup_skill()       ← cosine search in procedural memory
    │       └── build_context()      ← aggregate all layers → RAG prompt
    │
    ├── NovaAgent (agent.py)
    │       └── handle(text)         ← routes text through the full pipeline
    │
    └── AudioListener (audio.py)
            ├── run_one_round()      ← fixed-duration recording
            └── continuous_loop()   ← VAD-gated streaming loop
```

---

## Troubleshooting

**Model download is slow / fails**
→ Check `$HF_TOKEN` is set and the token has access to `Qwen/Qwen2.5-1.5B-Instruct`.

**`No module named 'sounddevice'`**
→ On Linux/Pi: `sudo apt install portaudio19-dev` then `pip install sounddevice`.

**TTS has no audio output on Pi**
→ `sudo apt install espeak` and check `aplay -l` to verify the audio device.

**VAD loop captures too much background noise**
→ Increase `ENERGY_THRESHOLD` in `config.py` (try `0.02` – `0.05`).

**LLM latency is too high on CPU**
→ Try `WHISPER_MODEL_SIZE = "tiny.en"` (already default) and consider int8 quantization (see Deployment section).

**ChromaDB error on first run**
→ The `nova_memory/` directory is created automatically. If it's corrupted, delete it: `rm -rf nova_memory/`.
