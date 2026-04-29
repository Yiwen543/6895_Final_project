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
from collections import deque
from faster_whisper import WhisperModel
from typing import Optional


def _find_input_device():
    """Return the index of the pulse/PipeWire device if present, else None."""
    for i, dev in enumerate(sd.query_devices()):
        if dev['max_input_channels'] > 0 and 'pulse' in dev['name'].lower():
            return i
    return None

def _find_output_device():
    """Return the index of the pulse/PipeWire output device if present, else None."""
    for i, dev in enumerate(sd.query_devices()):
        if dev['max_output_channels'] > 0 and 'pulse' in dev['name'].lower():
            return i
    return None

_INPUT_DEVICE  = _find_input_device()
_OUTPUT_DEVICE = _find_output_device()

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
        segs, _ = self._model.transcribe(buf, beam_size=1, initial_prompt="Nova,")
        return " ".join(s.text.strip() for s in segs).strip()

    def transcribe_file(self, path: str) -> str:
        segs, _ = self._model.transcribe(path, beam_size=1, initial_prompt="Nova,")
        return " ".join(s.text.strip() for s in segs).strip()


class TTSEngine:
    """Piper-based TTS. Interface unchanged: speak(text, verbose=True)."""

    def __init__(self, model_path: str = None):
        import wave
        import io as _io
        from piper.voice import PiperVoice
        from config import PIPER_MODEL_PATH
        _path = model_path or PIPER_MODEL_PATH
        print(f"Loading Piper TTS ({_path}) ...")
        self._voice = PiperVoice.load(_path)
        self._wave = wave
        self._io = _io
        print("TTS ready.")

    def speak(self, text: str, verbose: bool = True) -> None:
        if verbose:
            print("[TTS]", text)
        try:
            import numpy as np
            wav_buf = self._io.BytesIO()
            with self._wave.open(wav_buf, 'wb') as wf:
                self._voice.synthesize(text, wf)
            wav_buf.seek(0)
            with self._wave.open(wav_buf, 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                rate = wf.getframerate()
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            # duplicate to stereo for BT speaker compatibility
            stereo = np.stack([audio, audio], axis=1)
            sd.play(stereo, rate, device=_OUTPUT_DEVICE)
            sd.wait()
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
            device=_INPUT_DEVICE,
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
                device=_INPUT_DEVICE,
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
