# Bluetooth Audio Diagnostic Script Design
**Project:** Nova Smart Home Assistant — EECS 6895 Final Project  
**Date:** 2026-04-30  
**Goal:** Diagnose root cause of choppy Bluetooth A2DP audio on Raspberry Pi 5.

---

## Background

Nova's TTS output via ULT FIELD 1 Bluetooth speaker is intermittently choppy. The issue persists across playback methods (sounddevice, pw-play) and after switching from HFP to A2DP profile. Four candidate root causes have been identified.

---

## Script

**File:** `bt_diag.py` — placed in `~/nova/` on the Pi  
**Invocation:** `python3 bt_diag.py <subcommand>`

```
python3 bt_diag.py wifi      # Test 1: WiFi/BT coexistence
python3 bt_diag.py quantum   # Test 2: PipeWire quantum size
python3 bt_diag.py rssi      # Test 3: Bluetooth signal strength
python3 bt_diag.py cpu       # Test 4: CPU competition
```

---

## Shared Infrastructure

**Audio playback with xrun counting:**  
Each test uses `sounddevice.OutputStream` with a callback. The callback increments an `xrun_count` counter on `status.output_underflow`. A fixed 5-second mono float32 test signal (440 Hz sine wave) is synthesized in Python at 22050 Hz, then played.

**Output format per test:**
```
[RAW]  <metric>: <value>
...
[RESULT] PASS — <reason>
         or
[RESULT] FAIL — <reason> (likely root cause)
```

---

## Test 1: `wifi` — WiFi/BT Coexistence

**Hypothesis:** Pi 5's BCM4345C0 chip shares a 2.4 GHz antenna for WiFi and Bluetooth. Active WiFi causes periodic BT packet loss.

**Steps:**
1. Confirm WiFi status via `nmcli dev status`
2. Play 5s test audio → record `xrun_wifi_on`
3. `nmcli dev disconnect wlan0`
4. Play 5s test audio → record `xrun_wifi_off`
5. `nmcli dev connect wlan0` (restore)

**Raw output:** xrun count for each run, WiFi state  
**Verdict:**
- PASS: `abs(xrun_wifi_on - xrun_wifi_off) <= 2` — WiFi not the cause
- FAIL: `xrun_wifi_off < xrun_wifi_on - 2` — WiFi is the root cause

---

## Test 2: `quantum` — PipeWire Quantum Size

**Hypothesis:** Default PipeWire quantum (1024 frames @ 48 kHz = 21 ms) is too small for Bluetooth A2DP buffering, causing periodic underruns.

**Steps:**
1. Read current quantum via `pw-metadata -n settings`
2. Play 5s test audio → record `xrun_default`
3. `XDG_RUNTIME_DIR=/run/user/1000 pw-metadata -n settings 0 clock.force-quantum 2048`
4. Play 5s test audio → record `xrun_large`
5. Restore original quantum via `pw-metadata`

**Raw output:** quantum values, xrun counts  
**Verdict:**
- PASS: `abs(xrun_default - xrun_large) <= 2` — quantum not the cause
- FAIL: `xrun_large < xrun_default - 2` — quantum too small is the root cause

---

## Test 3: `rssi` — Bluetooth Signal Strength

**Hypothesis:** Weak signal or interference causes A2DP packet loss.

**Steps:**
1. Start audio playback (5s test signal) in a background thread
2. Poll `hcitool rssi 68:59:32:F5:D3:BC` every 500 ms → collect ~10 samples
3. Compute min / max / avg RSSI

**Raw output:** all RSSI samples, min/max/avg  
**Verdict:**
- PASS: avg RSSI > -70 dBm and no polling failures — signal is fine
- FAIL: avg RSSI ≤ -70 dBm or ≥ 3 polling failures — weak signal / interference

---

## Test 4: `cpu` — CPU Competition

**Hypothesis:** llama.cpp runs 4 threads saturating all cores; audio thread gets preempted, causing underruns.

**Steps:**
1. Play 5s test audio at idle → record `xrun_idle`
2. Start 3 Python busy-loop threads (`while True: pass`)
3. Play 5s test audio under load → record `xrun_loaded`
4. Stop busy-loop threads

**Raw output:** CPU load (from `/proc/loadavg`), xrun counts  
**Verdict:**
- PASS: `abs(xrun_idle - xrun_loaded) <= 2` — CPU not the cause
- FAIL: `xrun_loaded > xrun_idle + 2` — CPU starvation is the root cause

---

## Dependencies

- `sounddevice` — xrun counting via OutputStream callback
- `numpy` — test signal generation
- `subprocess` — nmcli, pw-metadata, hcitool
- No new pip installs required (all already in Nova's environment)

---

## Success Criteria

Each subcommand:
1. Runs to completion without crashing
2. Prints raw data for every metric
3. Prints a clear PASS/FAIL verdict with explanation
4. Restores system state (WiFi reconnected, quantum restored, threads stopped)
