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

# ── STT (Whisper) ─────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE   = "tiny.en"
WHISPER_DEVICE       = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

# ── Embedding ─────────────────────────────────────────────────────────────────
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ── TTS ───────────────────────────────────────────────────────────────────────
TTS_RATE = 170
PIPER_MODEL_PATH = "voices/en_US-lessac-medium.onnx"

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
