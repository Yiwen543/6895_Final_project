#!/usr/bin/env python3
"""
Bluetooth audio diagnostic script for Nova on Raspberry Pi 5.

Usage:
    python3 bt_diag.py wifi      # Test WiFi/BT coexistence
    python3 bt_diag.py quantum   # Test PipeWire quantum size
    python3 bt_diag.py rssi      # Test Bluetooth signal strength
    python3 bt_diag.py cpu       # Test CPU competition
"""

import argparse
import os
import subprocess
import sys
import threading
import time

import numpy as np
import sounddevice as sd

BT_MAC = "68:59:32:F5:D3:BC"
XDG = {"XDG_RUNTIME_DIR": "/run/user/1000"}
ENV = {**os.environ, **XDG}


def make_signal(duration: float = 5.0, rate: int = 22050) -> np.ndarray:
    """440 Hz sine wave, float32, mono."""
    t = np.linspace(0, duration, int(duration * rate), endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)


def play_and_count_xruns(signal: np.ndarray, rate: int = 22050) -> int:
    """Play signal via sounddevice OutputStream; return number of output underflows."""
    xruns = 0
    pos = 0

    def callback(outdata, frames, time_info, status):
        nonlocal xruns, pos
        if status.output_underflow:
            xruns += 1
        remaining = len(signal) - pos
        chunk = min(frames, remaining)
        outdata[:chunk, 0] = signal[pos:pos + chunk]
        if chunk < frames:
            outdata[chunk:] = 0
        pos += chunk

    with sd.OutputStream(
        samplerate=rate,
        channels=1,
        dtype="float32",
        callback=callback,
    ):
        duration = len(signal) / rate
        time.sleep(duration + 0.2)

    return xruns


def raw(label: str, value):
    print(f"[RAW]  {label}: {value}")


def result(passed: bool, reason: str):
    tag = "PASS" if passed else "FAIL"
    print(f"[RESULT] {tag} — {reason}")
