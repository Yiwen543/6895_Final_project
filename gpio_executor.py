import lgpio
import time
import threading
import colorsys

# Signal pins (Pi 5 BCM numbering, verified against 40-pin header)
DATA_PIN = 17   # Grove RGB LED DATA  → Pin 11
CLK_PIN  = 27   # Grove RGB LED CLK   → Pin 13
MOTOR_PINS = [5, 6, 13, 19]  # ULN2003 IN1-IN4 → Pins 29,31,33,35

# ⚠️ VCC and GND for LED and motor board come from an external PSU.
# The external PSU GND must share a common ground with a Pi GND pin
# (e.g. Pin 6 or Pin 9) for GPIO signal levels to be valid.

CURTAIN_TOTAL_STEPS = 2048  # half-revolution; calibrate after first wiring test
STEP_DELAY = 0.002          # 2 ms between half-steps


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
        lgpio.gpio_claim_output(self._h, DATA_PIN)
        lgpio.gpio_claim_output(self._h, CLK_PIN)
        for pin in MOTOR_PINS:
            lgpio.gpio_claim_output(self._h, pin)
        self._step_index = 0
        self._curtain_pos = 0
        self._rgb_stop = threading.Event()
        self._rgb_thread = None
        self._set_color(0, 0, 0)
        self._release_motor()

    def _send_bit(self, bit: int) -> None:
        lgpio.gpio_write(self._h, DATA_PIN, bit)
        lgpio.gpio_write(self._h, CLK_PIN, 1)
        lgpio.gpio_write(self._h, CLK_PIN, 0)

    def _send_byte(self, byte: int) -> None:
        for i in range(7, -1, -1):
            self._send_bit((byte >> i) & 1)

    def _set_color(self, r: int, g: int, b: int) -> None:
        for _ in range(32):
            self._send_bit(0)
        prefix = 0xC0
        prefix |= ((~b) & 0xC0) >> 2
        prefix |= ((~g) & 0xC0) >> 4
        prefix |= ((~r) & 0xC0) >> 6
        self._send_byte(prefix & 0xFF)
        self._send_byte(b & 0xFF)
        self._send_byte(g & 0xFF)
        self._send_byte(r & 0xFF)
        for _ in range(32):
            self._send_bit(0)

    def _start_rgb_cycle(self) -> None:
        self._stop_rgb_cycle()
        self._rgb_stop.clear()
        self._rgb_thread = threading.Thread(target=self._rgb_cycle_loop, daemon=True)
        self._rgb_thread.start()

    def _stop_rgb_cycle(self) -> None:
        if self._rgb_thread and self._rgb_thread.is_alive():
            self._rgb_stop.set()
            self._rgb_thread.join(timeout=0.5)

    def _rgb_cycle_loop(self) -> None:
        hue = 0.0
        while not self._rgb_stop.is_set():
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            self._set_color(int(r * 255), int(g * 255), int(b * 255))
            hue = (hue + 0.01) % 1.0
            time.sleep(0.1)

    def _do_step(self, direction: int) -> None:
        self._step_index = (self._step_index + direction) % 8
        for i, pin in enumerate(MOTOR_PINS):
            lgpio.gpio_write(self._h, pin, self.HALF_STEP_SEQ[self._step_index][i])
        time.sleep(STEP_DELAY)

    def _release_motor(self) -> None:
        for pin in MOTOR_PINS:
            lgpio.gpio_write(self._h, pin, 0)

    def _move_to_position(self, target_pct: int) -> None:
        steps = int((target_pct - self._curtain_pos) / 100 * CURTAIN_TOTAL_STEPS)
        direction = 1 if steps >= 0 else -1
        for _ in range(abs(steps)):
            self._do_step(direction)
        self._curtain_pos = target_pct
        self._release_motor()

    def execute(self, cmd: dict) -> str:
        device = cmd.get('device')
        action = cmd.get('action')
        value  = cmd.get('value')

        if device == 'light':
            if action == 'turn_on':
                self._stop_rgb_cycle()
                self._set_color(255, 255, 255)
                return 'LIGHT -> ON'
            if action == 'turn_off':
                self._stop_rgb_cycle()
                self._set_color(0, 0, 0)
                return 'LIGHT -> OFF'
            if action == 'set_brightness':
                self._stop_rgb_cycle()
                v = int(round(2.55 * value))
                self._set_color(v, v, v)
                return f'LIGHT -> BRIGHTNESS {value}%'
            if action == 'rgb_cycle':
                self._start_rgb_cycle()
                return 'LIGHT -> RGB CYCLE'

        if device == 'curtain':
            if action == 'open':
                self._move_to_position(100)
                return 'CURTAIN -> OPEN'
            if action == 'close':
                self._move_to_position(0)
                return 'CURTAIN -> CLOSE'
            if action == 'set_position':
                self._move_to_position(value)
                return f'CURTAIN -> POSITION {value}%'

        return f'[STUB] {device} {action}'

    def cleanup(self) -> None:
        self._stop_rgb_cycle()
        self._set_color(0, 0, 0)
        self._release_motor()
        lgpio.gpiochip_close(self._h)
