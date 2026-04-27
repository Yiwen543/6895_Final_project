# Nova Pi 5 Deployment Design

**Date:** 2026-04-26
**Project:** Nova Smart Home Assistant — EECS 6895 Final Project
**Scope:** Deploy the Nova pipeline to Raspberry Pi 5 (8GB) as a systemd service with GPIO hardware control and latency optimizations. No model architecture changes in this phase; GGUF quantization is planned separately.

---

## 1. Target Hardware

| Component | Spec |
|-----------|------|
| Board | Raspberry Pi 5 (8GB RAM) |
| OS | Raspberry Pi OS 64-bit (Bookworm) |
| Microphone | SunFounder USB 2.0 Mini Microphone |
| Speaker | Bluetooth speaker (PipeWire sink) |
| Light | Grove Chainable RGB LED (P9813) |
| Curtain | 28BYJ-48 stepper motor + ULN2003 driver board |

---

## 2. File Structure

```
nova/
├── nova.py                          # Main pipeline (refactored from Nova_4_16.ipynb)
├── gpio_executor.py                 # GPIO control layer (LED + stepper)
├── requirements_pi.txt              # Pi-specific dependencies
├── nova.service                     # systemd main service
├── bt-speaker.service               # Bluetooth auto-connect service
├── deploy.sh                        # One-command deploy script (run from dev machine)
└── voices/
    └── en_US-lessac-medium.onnx     # Piper TTS voice model (~60MB)
```

**Layer boundaries:**
- `nova.py` owns pipeline logic (STT → rule engine → LLM → TTS). It calls `gpio_executor.execute(cmd)` but knows nothing about GPIO internals.
- `gpio_executor.py` owns hardware. It receives a standard `cmd` dict (same schema as the existing `execute_command`) and knows nothing about the LLM.
- Swapping the model later (GGUF) only touches `nova.py`. Adding a new device only touches `gpio_executor.py`.

---

## 3. GPIO Wiring

### Pi 5 Note
Pi 5 uses the RP1 southbridge chip. **`RPi.GPIO` is not supported.** Use `lgpio` (native) or `rpi-lgpio` (drop-in compatibility shim).

### Pin Assignments (BCM numbering, verified against Pi 5 40-pin header)

| Device | Signal | Pi Physical Pin | GPIO |
|--------|--------|----------------|------|
| Grove RGB LED | DATA | Pin 11 | GPIO 17 |
| Grove RGB LED | CLK | Pin 13 | GPIO 27 |
| ULN2003 | IN1 | Pin 29 | GPIO 5 |
| ULN2003 | IN2 | Pin 31 | GPIO 6 |
| ULN2003 | IN3 | Pin 33 | GPIO 13 |
| ULN2003 | IN4 | Pin 35 | GPIO 19 |

**供电说明：** 所有设备（RGB LED、ULN2003 驱动板、步进马达）的 VCC 和 GND 均接外部独立电源，不走 Pi GPIO header 的 5V/GND 引脚。

**⚠️ 共地要求：** 外部电源的 GND 必须与 Pi 的任意 GND 引脚（如 Pin 6 或 Pin 9）短接，确保信号电平参考一致，否则 GPIO 输出无法可靠驱动设备。

### Action Mapping

| Command | GPIO Behavior |
|---------|---------------|
| `light turn_on` | P9813 → white (255, 255, 255) |
| `light turn_off` | P9813 → off (0, 0, 0) |
| `light set_brightness` | White scaled by percentage: `v = int(2.55 * value)` |
| `light rgb_cycle` | Background thread cycles hue (red→orange→…→red, 100ms/step) |
| `curtain open` | Stepper forward to 100% |
| `curtain close` | Stepper reverse to 0% |
| `curtain set_position` | Stepper moves from `current_pos` to `value`%; `current_pos` state maintained in `GPIOExecutor` |
| `window` / `ac` | Log only (reserved interface, no hardware wired) |

**Stepper motion:** half-step sequence (8 steps/cycle), smoother and quieter than full-step. One full curtain travel = fixed step count (calibrated at first run).

---

## 4. Audio

### USB Microphone (SunFounder USB 2.0 Mini Microphone)
标准 USB Audio Class 设备，Raspberry Pi OS 免驱即插即用。采样率支持 16000 Hz，与 Nova pipeline 一致。

Pi 上存在多个音频输入设备时（板载 + USB 麦克风），`sounddevice` 不一定自动选中 USB 麦克风。`deploy.sh` 执行时加入以下步骤，将其设为系统默认输入：

```bash
# 找到 SunFounder USB 麦克风对应的 PipeWire source
USB_SOURCE=$(pactl list sources short | grep -i usb | awk '{print $2}' | head -1)
pactl set-default-source "$USB_SOURCE"
```

`nova.py` 中 `sounddevice` 使用系统默认输入，无需硬编码设备索引，重新插拔后仍有效。

### Bluetooth Speaker (one-time manual pairing)
```bash
bluetoothctl
  power on
  agent on
  scan on          # note the speaker MAC, e.g. AA:BB:CC:DD:EE:FF
  pair   AA:BB:CC:DD:EE:FF
  trust  AA:BB:CC:DD:EE:FF
  connect AA:BB:CC:DD:EE:FF
  exit
```
Run this once over SSH before deploying. The `trust` command enables auto-reconnect across reboots.

### TTS: Piper
Replaces `pyttsx3/espeak`. Piper is the Raspberry Pi Foundation's recommended offline TTS — lower first-token latency and significantly better voice quality.

- Voice model: `en_US-lessac-medium` (~60MB ONNX, downloaded once by `deploy.sh`)
- Integration: `piper.voice.PiperVoice.load(model_path)` → `voice.synthesize_stream_raw(text)` → `sounddevice.play()`
- Output routes through PipeWire to the Bluetooth sink automatically once the speaker is the default audio output.

---

## 5. Latency Optimizations

All four optimizations are included in this deployment. None require model changes.

### 5.1 Rule-Based Fast Path
A `try_rule_based(text)` function runs before the LLM. It uses regex to handle unambiguous direct commands (turn on/off light, open/close curtain/window, turn on/off AC, set AC temperature, set brightness, set position). If it matches, the LLM is skipped entirely.

- Covers ~70% of real-world home control requests
- Latency for matched commands: **~3s → <5ms**
- Unmatched commands (ambiguous / general QA) still go to the LLM

Implementation follows the pattern already documented in `OPTIMIZATION_SUGGESTIONS.md` Issue 3 Fix A.

### 5.2 GPIO + TTS Parallel Execution
After intent resolution, device execution and voice response are launched as two concurrent threads:
```python
threading.Thread(target=gpio_executor.execute, args=(cmd,)).start()
threading.Thread(target=speak, args=(reply_text,)).start()
```
The LED turns on and the speaker starts talking at the same time instead of sequentially.

- Saves ~200–500ms per command (TTS first-token latency)

### 5.3 `max_new_tokens=96`
LLM generation is capped at 96 tokens. The longest valid JSON output (a `needs_clarification` with two options) fits within 80 tokens. Eliminates unnecessary generation cycles.

### 5.4 Piper TTS (see §4)
Lower first-token latency than espeak, no subprocess overhead.

---

## 6. Systemd Services

### `bt-speaker.service`
```ini
[Unit]
Description=Auto-connect Bluetooth Speaker
After=bluetooth.service
Wants=bluetooth.service

[Service]
Type=oneshot
ExecStart=/usr/bin/bluetoothctl connect AA:BB:CC:DD:EE:FF
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

### `nova.service`
```ini
[Unit]
Description=Nova Smart Home Assistant
After=sound.target bt-speaker.service
Wants=bt-speaker.service

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/nova
ExecStart=/usr/bin/python3 nova.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

`nova.service` depends on `bt-speaker.service` so the speaker is connected before the pipeline starts.

---

## 7. deploy.sh Flow

Run from the development machine:
```bash
./deploy.sh pi@<PI_IP> <BT_MAC>
```

Steps executed:
1. `rsync -avz --exclude '__pycache__' ./ pi@<PI_IP>:~/nova/`
2. SSH into Pi and run:
   - `apt-get install -y espeak-ng libportaudio2 python3-pip`
   - `pip3 install -r requirements_pi.txt`
   - Download Piper voice model to `~/nova/voices/` if not present
   - Set SunFounder USB mic as default PipeWire input (`pactl set-default-source`)
   - Write `bt-speaker.service` with the provided MAC address
   - Write `nova.service`
   - `systemctl daemon-reload && systemctl enable bt-speaker nova`
3. Print pairing reminder if this is the first deploy
4. Ask whether to `systemctl start nova` immediately

Re-running `deploy.sh` is safe — rsync is incremental, pip skips already-installed packages, systemctl enable is idempotent.

---

## 8. Out of Scope (This Phase)

- GGUF quantization / llama-cpp-python migration (planned separately)
- LoRA adapter loading
- Real AC or window hardware
- Multi-room or networked device control
- Web UI or remote control interface
