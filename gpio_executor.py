import lgpio
import os
import random
import subprocess
import time
import threading
import colorsys
from config import WINDOW_PINS, WINDOW_TOTAL_STEPS

try:
    from pi5neo import Pi5Neo
    _HAS_PI5NEO = True
except ImportError:
    _HAS_PI5NEO = False
    print("[GPIO] pi5neo not found — LED ring disabled (install: pip install pi5neo)")

# ── WS2812B LED Ring (pi5neo via SPI0) ───────────────────────────────────────
LED_SPI_DEV  = "/dev/spidev0.0"   # SPI0 MOSI = GPIO 10 / Pin 19
LED_COUNT    = 12                  # 12-LED ring
LED_FREQ     = 800                 # kHz

# ── Fan / AC simulation ───────────────────────────────────────────────────────
FAN_PIN      = 12         # GPIO 12 / Pin 32 (hardware PWM0)
FAN_FREQ_HZ  = 1000       # 1 kHz — lgpio max is 10 kHz; most fans accept ≥100 Hz
FAN_TACH_PIN = 16         # GPIO 16 / Pin 36 (optional tachometer input)

# ── Stepper motor (curtain) ───────────────────────────────────────────────────
MOTOR_PINS          = [5, 6, 13, 26]   # ULN2003 IN1-IN4 → Pins 29,31,33,37
CURTAIN_TOTAL_STEPS = 4096              # 2 revolutions full travel
STEP_DELAY          = 0.002
WINDOW_STEP_DELAY   = 0.002

# ── RGB cycle parameters ──────────────────────────────────────────────────────
RGB_HUE_STEP   = 0.01
RGB_CYCLE_TICK = 0.1


# Brightness level 1-3 → percentage (1=dim, 2=mid, 3=full)
_BRIGHTNESS_PCT = {1: 20, 2: 50, 3: 100}

# Color temperature lookup: level 1-5 → (R, G, B)
# Scale: 1=coldest(6500K/daylight) … 5=warmest(2700K/candlelight)
# Direction matches LLM intuition: small number = cold, large = warm.
_COLOR_TEMP_RGB = {
    1: (180, 210, 255),   # 6500K — cool blue-white / daylight
    2: (255, 255, 255),   # 5000K — neutral white / reading
    3: (255, 200,  80),   # 4000K — warm yellow
    4: (255, 120,   0),   # 3000K — orange
    5: (255,  50,   0),   # 2700K — deep orange
}


def _temp_to_duty(temp: int) -> float:
    return 20.0 if temp <= 22 else 100.0


class GPIOExecutor:
    HALF_STEP_SEQ = [
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 1],
    ]

    def __init__(self):
        self._h = lgpio.gpiochip_open(0)

        # ── Stepper motor ─────────────────────────────────────────────────────
        for pin in MOTOR_PINS:
            lgpio.gpio_claim_output(self._h, pin)
        self._step_index  = 0
        self._curtain_pos = 0
        self._release_motor()

        # ── Window stepper motor ──────────────────────────────────────────────
        for pin in WINDOW_PINS:
            lgpio.gpio_claim_output(self._h, pin)
        self._window_step_index = 0
        self._window_pos = 0
        self._release_window_motor()

        # ── Device state ──────────────────────────────────────────────────────
        self._color_temp_level = 3
        self._brightness_level = 2   # 1-3, default mid brightness

        # ── Fan (PWM) ─────────────────────────────────────────────────────────
        lgpio.gpio_claim_output(self._h, FAN_PIN)
        self._fan_duty = 0.0
        lgpio.tx_pwm(self._h, FAN_PIN, FAN_FREQ_HZ, 0)   # start stopped

        # ── WS2812B LED ring (pi5neo via SPI0) ───────────────────────────────
        self._strip = None
        if _HAS_PI5NEO:
            try:
                self._strip = Pi5Neo(LED_SPI_DEV, LED_COUNT, LED_FREQ)
                self._fill(0, 0, 0)
                print("[GPIO] LED ring ready (pi5neo SPI0)")
            except Exception as e:
                print(f"[GPIO] LED ring init failed ({e}) — LED disabled")
                self._strip = None

        self._rgb_stop   = threading.Event()
        self._rgb_thread = None
        self._rgb_lock   = threading.Lock()

        self._party_stop    = threading.Event()
        self._party_thread  = None
        self._party_music   = None
        self._party_lock    = threading.Lock()

    # ── LED helpers ───────────────────────────────────────────────────────────

    def _fill(self, r: int, g: int, b: int) -> None:
        if self._strip is None:
            return
        self._strip.fill_strip(r, g, b)
        self._strip.update_strip()

    def _apply_light(self) -> None:
        """Render current color_temp_level × brightness_level to LEDs."""
        scale = _BRIGHTNESS_PCT.get(self._brightness_level, 100) / 100.0
        r, g, b = _COLOR_TEMP_RGB.get(self._color_temp_level, (255, 255, 255))
        self._fill(int(r * scale), int(g * scale), int(b * scale))

    def _start_rgb_cycle(self) -> None:
        with self._rgb_lock:
            self._stop_rgb_cycle_unsafe()
            self._rgb_stop.clear()
            self._rgb_thread = threading.Thread(
                target=self._rgb_cycle_loop, daemon=True
            )
            self._rgb_thread.start()

    def _stop_rgb_cycle(self) -> None:
        with self._rgb_lock:
            self._stop_rgb_cycle_unsafe()

    def _stop_rgb_cycle_unsafe(self) -> None:
        if self._rgb_thread and self._rgb_thread.is_alive():
            self._rgb_stop.set()
            self._rgb_thread.join(timeout=0.5)

    def _rgb_cycle_loop(self) -> None:
        """Rotate a rainbow around the 12-LED ring."""
        while not self._rgb_stop.is_set():
            for offset in range(LED_COUNT):
                if self._rgb_stop.is_set() or self._strip is None:
                    break
                for i in range(LED_COUNT):
                    hue = ((i + offset) % LED_COUNT) / LED_COUNT
                    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
                    self._strip.set_led_color(
                        i, int(r * 255), int(g * 255), int(b * 255)
                    )
                self._strip.update_strip()
                time.sleep(RGB_CYCLE_TICK)

    # ── Party mode helpers ────────────────────────────────────────────────────

    _DISCO_COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (255, 128, 0), (128, 0, 255), (255, 0, 128),
    ]

    def _start_party(self) -> None:
        with self._party_lock:
            self._stop_party_unsafe()
            self._party_stop.clear()
            self._party_thread = threading.Thread(
                target=self._party_loop, daemon=True
            )
            self._party_thread.start()
            env = {**os.environ, "XDG_RUNTIME_DIR": "/run/user/1000"}
            try:
                self._party_music = subprocess.Popen(
                    ["mpg123", "-q", "/home/tl3461/Downloads/Rick-roll.mp3"],
                    env=env
                )
            except FileNotFoundError:
                self._party_music = None

    def _stop_party(self) -> None:
        with self._party_lock:
            self._stop_party_unsafe()

    def _stop_party_unsafe(self) -> None:
        self._party_stop.set()
        if self._party_thread and self._party_thread.is_alive():
            self._party_thread.join(timeout=0.5)
        if self._party_music and self._party_music.poll() is None:
            self._party_music.terminate()
        self._party_music = None

    def _party_loop(self) -> None:
        """Disco strobe: random saturated colors flashing at ~5 fps."""
        while not self._party_stop.is_set():
            r, g, b = random.choice(self._DISCO_COLORS)
            self._fill(r, g, b)
            time.sleep(0.15)
            if self._party_stop.is_set():
                break
            self._fill(0, 0, 0)
            time.sleep(0.05)

    # ── Fan helpers ───────────────────────────────────────────────────────────

    def _set_fan(self, duty: float) -> None:
        duty = max(0.0, min(100.0, duty))
        self._fan_duty = duty
        lgpio.tx_pwm(self._h, FAN_PIN, FAN_FREQ_HZ, duty)

    # ── Stepper motor helpers ─────────────────────────────────────────────────

    def _do_step(self, direction: int) -> None:
        self._step_index = (self._step_index + direction) % 8
        for i, pin in enumerate(MOTOR_PINS):
            lgpio.gpio_write(self._h, pin, self.HALF_STEP_SEQ[self._step_index][i])
        time.sleep(STEP_DELAY)

    def _release_motor(self) -> None:
        for pin in MOTOR_PINS:
            lgpio.gpio_write(self._h, pin, 0)

    def _do_window_step(self, direction: int) -> None:
        self._window_step_index = (self._window_step_index + direction) % 8
        for i, pin in enumerate(WINDOW_PINS):
            lgpio.gpio_write(self._h, pin, self.HALF_STEP_SEQ[self._window_step_index][i])
        time.sleep(WINDOW_STEP_DELAY)

    def _release_window_motor(self) -> None:
        for pin in WINDOW_PINS:
            lgpio.gpio_write(self._h, pin, 0)

    def _move_window(self, target_pct: int) -> None:
        target_pct = 0 if target_pct <= 50 else 100
        steps     = int((target_pct - self._window_pos) / 100 * WINDOW_TOTAL_STEPS)
        direction = 1 if steps >= 0 else -1
        for _ in range(abs(steps)):
            self._do_window_step(direction)
        self._window_pos = target_pct
        self._release_window_motor()

    def get_device_state(self) -> dict:
        return {
            "color_temp": self._color_temp_level,
            "brightness": self._brightness_level,
            "curtain_pos": self._curtain_pos,
            "window_pos": self._window_pos,
        }

    def _move_to_position(self, target_pct: int) -> None:
        target_pct = max(0, min(100, target_pct))
        steps     = int((target_pct - self._curtain_pos) / 100 * CURTAIN_TOTAL_STEPS)
        direction = 1 if steps >= 0 else -1
        for _ in range(abs(steps)):
            self._do_step(direction)
        self._curtain_pos = target_pct
        self._release_motor()

    # ── Main dispatch ─────────────────────────────────────────────────────────

    def execute(self, cmd: dict) -> str:
        device = cmd.get("device")
        action = cmd.get("action")
        value  = cmd.get("value")

        if device == "light":
            if action == "party_mode":
                self._stop_rgb_cycle()
                self._start_party()
                return "LIGHT -> PARTY MODE"
            if action == "turn_on":
                self._stop_rgb_cycle(); self._stop_party()
                self._brightness_level = 2
                self._apply_light()
                return "LIGHT -> ON"
            if action == "turn_off":
                self._stop_rgb_cycle(); self._stop_party()
                self._fill(0, 0, 0)
                return "LIGHT -> OFF"
            if action == "set_brightness":
                self._stop_rgb_cycle(); self._stop_party()
                pct = int(value)
                self._brightness_level = 1 if pct <= 33 else (2 if pct <= 66 else 3)
                self._apply_light()
                return f"LIGHT -> BRIGHTNESS level {self._brightness_level} ({pct}%)"
            if action == "rgb_cycle":
                self._stop_party()
                self._start_rgb_cycle()
                return "LIGHT -> RGB CYCLE"
            if action == "set_color_temp":
                self._stop_rgb_cycle(); self._stop_party()
                self._color_temp_level = int(value)
                self._apply_light()
                return f"LIGHT -> COLOR TEMP {value}"

        if device == "curtain":
            if action == "open":
                self._move_to_position(100)
                return "CURTAIN -> OPEN"
            if action == "close":
                self._move_to_position(0)
                return "CURTAIN -> CLOSE"
            if action == "set_position":
                self._move_to_position(int(value))
                return f"CURTAIN -> POSITION {value}%"

        if device == "window":
            if action == "open":
                self._move_window(100)
                return "WINDOW -> OPEN"
            if action == "close":
                self._move_window(0)
                return "WINDOW -> CLOSE"
            if action == "set_position":
                self._move_window(int(value))
                return f"WINDOW -> POSITION {value}%"

        if device == "ac":
            if action == "turn_on":
                self._set_fan(50.0)
                return "AC -> ON (fan 50%)"
            if action == "turn_off":
                self._set_fan(0.0)
                return "AC -> OFF"
            if action == "set_temperature":
                duty = _temp_to_duty(int(value))
                self._set_fan(duty)
                return f"AC -> TEMPERATURE {value}C (fan {duty:.0f}%)"

        return f"[STUB] {device} {action}"

    def cleanup(self) -> None:
        self._stop_rgb_cycle()
        self._stop_party()
        self._fill(0, 0, 0)
        self._set_fan(0.0)
        lgpio.tx_pwm(self._h, FAN_PIN, FAN_FREQ_HZ, 0)
        self._release_motor()
        self._release_window_motor()
        lgpio.gpiochip_close(self._h)
