"""
Full voice pipeline smoke test: STT -> Agent -> TTS
Usage:
    cd ~/nova
    PULSE_SERVER=unix:/run/user/1000/pulse/native XDG_RUNTIME_DIR=/run/user/1000 python3 test_voice_pipeline.py
"""

import sounddevice as sd
import numpy as np
import time
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sentence_transformers import SentenceTransformer
from llm_parser import LLMParser
from memory import MemoryManager
from agent import NovaAgent
from audio import STTModel, TTSEngine, _INPUT_DEVICE
from config import SAMPLE_RATE, CHANNELS, AUDIO_DTYPE, ASSISTANT_NAME_VARIANTS

ROUNDS = 3
RECORD_SECS = 5


def main():
    print("=" * 60)
    print("Loading models (may take 20-30s) ...")
    _embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embed  = lambda t: _embed.encode(t, convert_to_numpy=True).tolist()
    llm    = LLMParser()
    tts    = TTSEngine()
    memory = MemoryManager(embed_fn=embed)
    nova   = NovaAgent(llm=llm, memory=memory, speak=tts.speak)
    stt    = STTModel()
    print("All models ready.\n")
    print("Suggested commands to try:")
    print("  'Nova, turn on the light'")
    print("  'Nova, set the AC to 24 degrees'")
    print("  'Nova, I feel cold'")
    print("=" * 60)
    print()

    for i in range(ROUNDS):
        print("=== Round %d/%d ===" % (i + 1, ROUNDS))
        for c in [3, 2, 1]:
            print("  %d..." % c)
            time.sleep(1)
        print("  >>> SPEAK NOW (%ds) <<<" % RECORD_SECS)

        audio = sd.rec(
            RECORD_SECS * SAMPLE_RATE,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=AUDIO_DTYPE,
            device=_INPUT_DEVICE,
        )
        sd.wait()

        mean_e = float(np.mean(np.abs(audio)))
        peak_e = float(np.max(np.abs(audio)))
        print("  energy: mean=%.4f  peak=%.4f" % (mean_e, peak_e))

        if peak_e < 0.05:
            print("  No speech detected, skipping\n")
            continue

        t0 = time.perf_counter()
        text = stt.transcribe(audio)
        stt_ms = (time.perf_counter() - t0) * 1000
        print("  STT (%.0fms): %r" % (stt_ms, text))

        nova.reset_dialogue()
        t1 = time.perf_counter()
        result = nova.handle(text, verbose=False)
        agent_ms = (time.perf_counter() - t1) * 1000

        semantic = result.get("semantic", {})
        print("  Agent (%.0fms)" % agent_ms)
        print("    prefilter : %s" % result.get("prefilter_passed"))
        print("    type      : %s" % semantic.get("type"))
        print("    execution : %s" % result.get("execution"))
        print("    reply     : %s" % result.get("spoken_reply", ""))
        print()


if __name__ == "__main__":
    main()
