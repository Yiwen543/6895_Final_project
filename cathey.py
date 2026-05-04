#!/usr/bin/env python3
"""Cathey Smart Home Assistant — Raspberry Pi 5 entry point."""

import sys
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
from sentence_transformers import SentenceTransformer

from audio import AudioListener, STTModel, TTSEngine
from llm_parser import LLMParser
from memory import MemoryManager
from agent import CatheyAgent
from gpio_executor import GPIOExecutor
from config import EMBED_MODEL_NAME


def main() -> None:
    print("=== Cathey starting up ===")

    stt    = STTModel()
    tts    = TTSEngine()
    llm    = LLMParser()
    embed  = SentenceTransformer(EMBED_MODEL_NAME)
    memory = MemoryManager(embed_fn=lambda text: embed.encode(text, convert_to_numpy=True).tolist())
    gpio   = GPIOExecutor()

    cathey = CatheyAgent(llm=llm, memory=memory, speak=tts.speak, gpio=gpio)

    listener = AudioListener(agent=cathey, stt=stt)

    embed.encode("warmup", convert_to_numpy=True)   # pre-warm embedding JIT
    print("=== Cathey ready. Listening... ===")
    try:
        listener.continuous_loop()
    except KeyboardInterrupt:
        pass
    finally:
        gpio.cleanup()
        print("Cathey shut down.")


if __name__ == "__main__":
    main()
