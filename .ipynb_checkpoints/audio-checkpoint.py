"""
Audio module: STT, TTS, and continuous voice activity detection (VAD) listener.

  STTModel      — Whisper-based speech-to-text (numpy array or file path)
  TTSEngine     — pyttsx3 text-to-speech
  AudioListener — one-shot recording + continuous VAD loop, feeds into NovaAgent
"""

import io
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import pyttsx3
from collections import deque
from faster_whisper import WhisperModel
from typing import Optional

from config import (
    AUDIO_DTYPE,
    CHANNELS,
    ENERGY_THRESHOLD,
    FRAME_DURATION,
    FRAME_SAMPLES,
    MAX_FRAMES,
    MIN_SPEECH_SECONDS,
    PRE_ROLL_FRAMES,
    SAMPLE_RATE,
    SILENCE_FRAMES,
    TTS_RATE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL_SIZE,
)


class STTModel:
    """Faster-Whisper speech-to-text. Transcribes in-memory numpy arrays."""

    def __init__(
        self,
        model_size: str = WHISPER_MODEL_SIZE,
        device: str = WHISPER_DEVICE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
    ):
        print(f"Loading STT ({model_size}) ...")
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("STT ready.")

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe from a numpy float32 array without touching disk."""
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="wav")
        buf.seek(0)
        segs, _ = self._model.transcribe(buf, beam_size=1)
        return " ".join(s.text.strip() for s in segs).strip()

    def transcribe_file(self, path: str) -> str:
        segs, _ = self._model.transcribe(path, beam_size=1)
        return " ".join(s.text.strip() for s in segs).strip()


class TTSEngine:
    """Thin wrapper around pyttsx3 with error isolation."""

    def __init__(self, rate: int = TTS_RATE):
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", rate)

    def speak(self, text: str, verbose: bool = True):
        if verbose:
            print("[TTS]", text)
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as e:
            print("[TTS ERROR]", e)


class AudioListener:
    """
    Wraps audio I/O for the Nova pipeline.

    run_one_round()    — record a fixed-duration clip and process it.
    continuous_loop()  — VAD-gated loop; captures full utterances and feeds agent.
    """

    def __init__(self, agent, stt: STTModel):
        self._agent = agent
        self._stt   = stt

    # ── One-shot round ────────────────────────────────────────────────────────

    def run_one_round(self, duration_sec: int = 3) -> dict:
        print(f"[1/4] Recording {duration_sec}s ...")
        audio = sd.rec(
            int(duration_sec * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=AUDIO_DTYPE,
        )
        sd.wait()

        energy = float(np.mean(np.abs(audio)))
        print(f"[2/4] Energy: {energy:.6f}")
        if energy < ENERGY_THRESHOLD:
            print("[STOP] Too quiet.")
            return {"prefilter_passed": False, "reason": "too_quiet"}

        t0 = time.perf_counter()
        text = self._stt.transcribe(audio)
        stt_ms = (time.perf_counter() - t0) * 1000
        print(f"[3/4] STT ({stt_ms:.0f} ms): {text!r}")

        if not text:
            return {"prefilter_passed": False, "reason": "empty_transcription"}

        print("[4/4] Running agent ...")
        result = self._agent.handle(text, verbose=True)
        result["stt_latency_ms"] = round(stt_ms, 3)
        return result

    # ── VAD-gated utterance collection ───────────────────────────────────────

    @staticmethod
    def _frame_energy(frame: np.ndarray) -> float:
        return float(np.mean(np.abs(frame)))

    def _collect_utterance(self, stream) -> Optional[np.ndarray]:
        pre_buf   = deque(maxlen=PRE_ROLL_FRAMES)
        collected = []
        started = False
        silence_count = 0
        speech_frames = 0

        while True:
            frame, _ = stream.read(FRAME_SAMPLES)
            frame = frame.copy()
            energy = self._frame_energy(frame)

            if not started:
                pre_buf.append(frame)
                if energy >= ENERGY_THRESHOLD:
                    started = True
                    collected.extend(list(pre_buf))
                    collected.append(frame)
                    speech_frames += 1
            else:
                collected.append(frame)
                if energy >= ENERGY_THRESHOLD:
                    speech_frames += 1
                    silence_count = 0
                else:
                    silence_count += 1
                if silence_count >= SILENCE_FRAMES or len(collected) >= MAX_FRAMES:
                    break

        if not started:
            return None
        if speech_frames * FRAME_DURATION < MIN_SPEECH_SECONDS:
            return None
        return np.concatenate(collected, axis=0)

    # ── Continuous loop ───────────────────────────────────────────────────────

    def continuous_loop(self):
        print("Listening... Say 'Nova' to start a command. Ctrl+C to stop.\n")
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=AUDIO_DTYPE,
                blocksize=FRAME_SAMPLES,
            ) as stream:
                while True:
                    print("[Listener] Waiting for speech ...")
                    audio = self._collect_utterance(stream)
                    if audio is None:
                        continue

                    t0 = time.perf_counter()
                    text = self._stt.transcribe(audio)
                    stt_ms = (time.perf_counter() - t0) * 1000
                    print(f"[STT {stt_ms:.0f} ms] {text!r}")

                    result = self._agent.handle(text, verbose=True)
                    print("[Result]", result)
                    print("-" * 60)

        except KeyboardInterrupt:
            print("\nListening stopped.")
