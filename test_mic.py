"""
Microphone + STT smoke test.
Usage:
    cd ~/nova
    PULSE_SERVER=unix:/run/user/1000/pulse/native XDG_RUNTIME_DIR=/run/user/1000 python3 test_mic.py
"""

import sounddevice as sd
import numpy as np
import time
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio import STTModel, _INPUT_DEVICE
from config import SAMPLE_RATE, CHANNELS, AUDIO_DTYPE, ASSISTANT_NAME_VARIANTS

ROUNDS = 3
RECORD_SECS = 4


def main():
    print("Loading STT ...")
    stt = STTModel()
    print("STT ready.  Input device: %s\n" % _INPUT_DEVICE)

    for i in range(ROUNDS):
        print("=== Round %d/%d: say 'Nova, turn on the light' ===" % (i + 1, ROUNDS))
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
        print("  mean=%.4f  peak=%.4f" % (mean_e, peak_e))

        t0 = time.perf_counter()
        text = stt.transcribe(audio)
        ms = (time.perf_counter() - t0) * 1000

        triggered = any(v in text.lower() for v in ASSISTANT_NAME_VARIANTS)
        print("  STT (%.0fms): %r" % (ms, text))
        print("  Wakeword: %s" % ("YES" if triggered else "NO"))
        print()


if __name__ == "__main__":
    main()
