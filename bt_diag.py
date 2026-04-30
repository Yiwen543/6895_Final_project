#!/usr/bin/env python3
"""
Bluetooth audio diagnostic script for Nova on Raspberry Pi 5.

Usage:
    python3 bt_diag.py wifi       # Test WiFi/BT coexistence
    python3 bt_diag.py quantum    # Test PipeWire quantum size
    python3 bt_diag.py rssi       # Test Bluetooth signal strength
    python3 bt_diag.py cpu        # Test CPU competition (busy-loop threads)
    python3 bt_diag.py codec      # Check A2DP codec negotiated with speaker
    python3 bt_diag.py inference  # Test audio under LLM-like cache-intensive load
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
    done = threading.Event()

    def callback(outdata, frames, time_info, status):
        nonlocal xruns, pos
        if status.output_underflow:
            xruns += 1
        remaining = len(signal) - pos
        chunk = min(frames, remaining)
        outdata[:chunk, 0] = signal[pos:pos + chunk]
        if chunk < frames:
            outdata[chunk:] = 0
            done.set()
            raise sd.CallbackStop
        pos += chunk

    with sd.OutputStream(
        samplerate=rate,
        channels=1,
        dtype="float32",
        callback=callback,
    ):
        done.wait(timeout=len(signal) / rate + 2.0)

    return xruns


def raw(label: str, value):
    print(f"[RAW]  {label}: {value}")


def result(passed: bool, reason: str):
    tag = "PASS" if passed else "FAIL"
    print(f"[RESULT] {tag} — {reason}")


def cmd_wifi():
    print("\n=== TEST: WiFi/BT Coexistence ===")
    sig = make_signal()

    out = subprocess.run(
        ["nmcli", "dev", "status"], capture_output=True, text=True, env=ENV
    )
    wifi_on = "wlan0" in out.stdout and "connected" in out.stdout
    raw("wifi_active", wifi_on)

    print("[INFO] Playing with WiFi ON...")
    xrun_on = play_and_count_xruns(sig)
    raw("xrun_wifi_on", xrun_on)

    subprocess.run(["sudo", "nmcli", "dev", "disconnect", "wlan0"],
                   capture_output=True, env=ENV)
    time.sleep(1)

    print("[INFO] Playing with WiFi OFF...")
    xrun_off = play_and_count_xruns(sig)
    raw("xrun_wifi_off", xrun_off)

    subprocess.run(["sudo", "nmcli", "dev", "connect", "wlan0"],
                   capture_output=True, env=ENV)

    passed = abs(xrun_on - xrun_off) <= 2
    if passed:
        result(True, "xrun count unchanged with WiFi on/off — WiFi is NOT the cause")
    else:
        result(False, f"xruns dropped {xrun_on}→{xrun_off} when WiFi disconnected — WiFi IS the root cause")


def cmd_quantum():
    print("\n=== TEST: PipeWire Quantum Size ===")
    sig = make_signal()

    out = subprocess.run(
        ["pw-metadata", "-n", "settings"],
        capture_output=True, text=True, env=ENV
    )
    current_quantum = "1024"
    for line in out.stdout.splitlines():
        if "clock.force-quantum" in line or "clock.quantum" in line:
            parts = line.split("'")
            if len(parts) >= 4:
                current_quantum = parts[3]
                break
    raw("quantum_current", current_quantum)

    print(f"[INFO] Playing at quantum={current_quantum}...")
    xrun_default = play_and_count_xruns(sig)
    raw("xrun_default_quantum", xrun_default)

    subprocess.run(
        ["pw-metadata", "-n", "settings", "0", "clock.force-quantum", "2048"],
        capture_output=True, env=ENV
    )
    time.sleep(0.5)

    print("[INFO] Playing at quantum=2048...")
    xrun_large = play_and_count_xruns(sig)
    raw("xrun_quantum_2048", xrun_large)

    subprocess.run(
        ["pw-metadata", "-n", "settings", "0", "clock.force-quantum", current_quantum],
        capture_output=True, env=ENV
    )

    passed = abs(xrun_default - xrun_large) <= 2
    if passed:
        result(True, "xrun count unchanged with larger quantum — quantum is NOT the cause")
    else:
        result(False, f"xruns dropped {xrun_default}→{xrun_large} with quantum=2048 — quantum IS the root cause")


def _read_load() -> str:
    with open("/proc/loadavg") as f:
        return f.read().split()[0]


def cmd_rssi():
    print("\n=== TEST: Bluetooth Signal Strength ===")
    sig = make_signal()
    samples = []
    failures = 0

    def poll_rssi():
        nonlocal failures
        for _ in range(10):
            out = subprocess.run(
                ["hcitool", "rssi", BT_MAC],
                capture_output=True, text=True, env=ENV
            )
            line = out.stdout.strip()
            try:
                val = int(line.split()[-1])
                samples.append(val)
                raw("rssi_sample", val)
            except (ValueError, IndexError):
                failures += 1
                raw("rssi_sample", f"FAIL ({line!r})")
            time.sleep(0.5)

    poller = threading.Thread(target=poll_rssi, daemon=True)
    poller.start()
    play_and_count_xruns(sig)
    poller.join()

    if samples:
        avg = sum(samples) / len(samples)
        raw("rssi_min", min(samples))
        raw("rssi_max", max(samples))
        raw("rssi_avg", f"{avg:.1f}")
        raw("poll_failures", failures)
        passed = avg > -70 and failures < 3
        if passed:
            result(True, f"avg RSSI {avg:.1f} dBm > -70 — signal is NOT the cause")
        else:
            reason = []
            if avg <= -70:
                reason.append(f"avg RSSI {avg:.1f} dBm ≤ -70")
            if failures >= 3:
                reason.append(f"{failures} poll failures")
            result(False, "; ".join(reason) + " — weak signal / interference IS likely a cause")
    else:
        result(False, f"all {failures} RSSI polls failed — cannot measure signal")


def cmd_cpu():
    print("\n=== TEST: CPU Competition ===")
    sig = make_signal()

    load_before = _read_load()
    raw("load_avg_before", load_before)
    print("[INFO] Playing at idle...")
    xrun_idle = play_and_count_xruns(sig)
    raw("xrun_idle", xrun_idle)

    stop_event = threading.Event()

    def busy():
        while not stop_event.is_set():
            pass

    threads = [threading.Thread(target=busy, daemon=True) for _ in range(3)]
    for t in threads:
        t.start()
    time.sleep(0.5)

    load_during = _read_load()
    raw("load_avg_during", load_during)
    print("[INFO] Playing under CPU load (3 busy threads)...")
    xrun_loaded = play_and_count_xruns(sig)
    raw("xrun_loaded", xrun_loaded)

    stop_event.set()
    for t in threads:
        t.join(timeout=1)

    passed = xrun_loaded <= xrun_idle + 2
    if passed:
        result(True, f"xruns {xrun_idle}→{xrun_loaded} under load — CPU is NOT the cause")
    else:
        result(False, f"xruns jumped {xrun_idle}→{xrun_loaded} under load — CPU starvation IS the root cause")


def cmd_codec():
    print("\n=== TEST: A2DP Codec ===")

    codec_hits = []

    # 1. pw-dump — PipeWire native, always available
    out = subprocess.run(["pw-dump"], capture_output=True, text=True, env=ENV)
    for line in out.stdout.splitlines():
        ls = line.strip()
        if ("bluez" in ls.lower() or "a2dp" in ls.lower()) and "codec" in ls.lower():
            raw("pw_dump_codec", ls)
            codec_hits.append(ls.lower())

    # 2. wpctl status — lists active BT sink with codec info
    out2 = subprocess.run(["wpctl", "status"], capture_output=True, text=True, env=ENV)
    in_audio = False
    for line in out2.stdout.splitlines():
        ls = line.strip()
        if "Audio" in line:
            in_audio = True
        if in_audio and ("bluez" in ls.lower() or BT_MAC.lower().replace(":", "_") in ls.lower()):
            raw("wpctl_bt_sink", ls)
        if in_audio and "codec" in ls.lower():
            raw("wpctl_codec", ls)
            codec_hits.append(ls.lower())

    # 3. bluetoothctl info — shows negotiated profile
    out3 = subprocess.run(
        ["bluetoothctl", "info", BT_MAC],
        capture_output=True, text=True, env=ENV
    )
    for line in out3.stdout.splitlines():
        ls = line.strip()
        if ls:
            raw("btctl", ls)
            if any(k in ls.lower() for k in ["codec", "profile", "a2dp", "sbc", "aac", "aptx"]):
                codec_hits.append(ls.lower())

    # Verdict
    combined = " ".join(codec_hits)
    if any(c in combined for c in ["aac", "aptx", "ldac"]):
        result(True, "High-quality codec active — codec is NOT the likely cause")
    elif "sbc" in combined:
        result(False,
               "SBC codec in use — SBC has low fixed bitrate and causes choppy audio; "
               "fix: sudo apt install libldac bluez-plugins, then re-pair the speaker")
    else:
        result(False,
               "Could not confirm codec from pw-dump/wpctl/bluetoothctl; "
               "check the btctl lines above for 'Codec' or 'UUID' entries")


def cmd_inference():
    print("\n=== TEST: Audio under LLM-like Inference Load ===")
    print("[INFO] This test uses numpy matrix ops to simulate OpenBLAS cache pressure,")
    print("       which is closer to real llama.cpp behaviour than busy-loop threads.\n")
    sig = make_signal(duration=10.0)

    print("[INFO] Baseline: playing at idle...")
    xrun_idle = play_and_count_xruns(sig)
    raw("xrun_idle", xrun_idle)

    stop_event = threading.Event()

    def matrix_worker():
        # 512×512 float32 matmul: ~0.5 GB/s memory bandwidth per thread,
        # matching a single llama.cpp matrix multiply step on Pi 5.
        A = np.random.randn(512, 512).astype(np.float32)
        B = np.random.randn(512, 512).astype(np.float32)
        while not stop_event.is_set():
            np.dot(A, B)

    threads = [threading.Thread(target=matrix_worker, daemon=True) for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(0.5)

    print("[INFO] Playing under 4-thread matrix-multiply load...")
    xrun_loaded = play_and_count_xruns(sig)
    raw("xrun_loaded", xrun_loaded)

    stop_event.set()
    for t in threads:
        t.join(timeout=2)

    passed = xrun_loaded <= xrun_idle + 2
    if passed:
        result(True,
               f"xruns {xrun_idle}→{xrun_loaded} under matrix load — "
               "inference cache pressure is NOT the cause")
    else:
        result(False,
               f"xruns jumped {xrun_idle}→{xrun_loaded} under matrix load — "
               "cache-intensive LLM inference IS starving the BT audio thread; "
               "fix: sudo chrt -f -p 99 $(pidof pipewire)  OR  "
               "pw-metadata -n settings 0 clock.force-quantum 4096")


def main():
    parser = argparse.ArgumentParser(
        description="Nova Bluetooth audio diagnostic tool"
    )
    parser.add_argument(
        "test",
        choices=["wifi", "quantum", "rssi", "cpu", "codec", "inference"],
        help="Which diagnostic test to run",
    )
    args = parser.parse_args()
    {
        "wifi":      cmd_wifi,
        "quantum":   cmd_quantum,
        "rssi":      cmd_rssi,
        "cpu":       cmd_cpu,
        "codec":     cmd_codec,
        "inference": cmd_inference,
    }[args.test]()


if __name__ == "__main__":
    main()
