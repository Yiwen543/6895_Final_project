import torch

# ── LLM ──────────────────────────────────────────────────────────────────────
# Recommended models (swap LLM_MODEL_NAME to change):
#   "Qwen/Qwen2.5-3B-Instruct"           — default, drop-in upgrade from 1.5B (~2 GB int4)
#   "Qwen/Qwen2.5-1.5B-Instruct"         — original, lightest Qwen option (~0.9 GB int4)
#   "google/gemma-2-2b-it"               — fastest on Pi 5, ~1.2 GB int4
#   "meta-llama/Llama-3.2-3B-Instruct"   — strong JSON output, ~1.8 GB int4
#   "microsoft/Phi-3.5-mini-instruct"    — highest quality at 3.8B, ~2.4 GB int4
#   "HuggingFaceTB/SmolLM2-1.7B-Instruct" — ultra-light fallback, ~1 GB int4
LLM_MODEL_NAME     = "Qwen/Qwen2.5-3B-Instruct"

if torch.cuda.is_available():
    LLM_DEVICE       = "cuda"
    LLM_DTYPE        = torch.float16
    LLM_LOAD_IN_4BIT = False
elif torch.backends.mps.is_available():
    LLM_DEVICE       = "mps"
    LLM_DTYPE        = torch.float16
    LLM_LOAD_IN_4BIT = False
else:
    # CPU / Raspberry Pi 5 — use 4-bit quantization to keep memory under 3 GB
    LLM_DEVICE       = "cpu"
    LLM_DTYPE        = torch.float32
    LLM_LOAD_IN_4BIT = True

LLM_MAX_NEW_TOKENS = 160

# ── STT (Whisper) ─────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE   = "tiny.en"
WHISPER_DEVICE       = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ── TTS ───────────────────────────────────────────────────────────────────────
TTS_RATE = 170

# ── Assistant identity ────────────────────────────────────────────────────────
ASSISTANT_NAME = "nova"
ASSISTANT_NAME_VARIANTS = [
    "nova", "nava", "no va", "noba", "noa", "nove", "novia",
]

# ── Audio capture ─────────────────────────────────────────────────────────────
SAMPLE_RATE           = 16000
CHANNELS              = 1
AUDIO_DTYPE           = "float32"
ENERGY_THRESHOLD      = 0.01

FRAME_DURATION        = 0.1
FRAME_SAMPLES         = int(SAMPLE_RATE * FRAME_DURATION)
SILENCE_SECONDS       = 0.5
MIN_SPEECH_SECONDS    = 0.3
MAX_UTTERANCE_SECONDS = 8.0
PRE_ROLL_SECONDS      = 0.3

SILENCE_FRAMES = int(SILENCE_SECONDS / FRAME_DURATION)
PRE_ROLL_FRAMES = int(PRE_ROLL_SECONDS / FRAME_DURATION)
MAX_FRAMES     = int(MAX_UTTERANCE_SECONDS / FRAME_DURATION)

# ── Memory ────────────────────────────────────────────────────────────────────
MEMORY_DIR          = "nova_memory"
WORKING_MAXLEN      = 8
SKILL_SIM_THRESHOLD = 0.82   # procedural memory cosine similarity cutoff
EPISODE_DIST_CUTOFF = 0.6    # episodic RAG: keep episodes with distance < this

# ── LoRA fine-tuning ──────────────────────────────────────────────────────────
LORA_R           = 8
LORA_ALPHA       = 16
LORA_ADAPTER_DIR = "nova_lora_adapter"
LORA_MERGED_DIR  = "nova_lora_merged"
