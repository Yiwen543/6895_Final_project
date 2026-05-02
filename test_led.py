#!/usr/bin/env python3
"""
LED ring test for WS2812B 12-LED via pi5neo (SPI0).
Run: python3 test_led.py
Each step pauses 2s so you can observe the LED.
"""

import time
import colorsys
from pi5neo import Pi5Neo

LED_COUNT = 12
SPI_DEV   = "/dev/spidev0.0"
FREQ      = 800   # kHz

def main():
    print(f"Connecting to LED ring ({LED_COUNT} LEDs) via {SPI_DEV}...")
    strip = Pi5Neo(SPI_DEV, LED_COUNT, FREQ)
    print("OK\n")

    tests = [
        ("RED   (all)",   (255,   0,   0)),
        ("GREEN (all)",   (  0, 255,   0)),
        ("BLUE  (all)",   (  0,   0, 255)),
        ("WHITE (all)",   (255, 255, 255)),
        ("DIM   (10%)",   ( 25,  25,  25)),
        ("OFF",           (  0,   0,   0)),
    ]

    # ── Solid colour tests ────────────────────────────────────────────────────
    for label, (r, g, b) in tests:
        print(f"[TEST] {label} → rgb({r},{g},{b})")
        strip.fill_strip(r, g, b)
        strip.update_strip()
        time.sleep(2)

    # ── One pixel at a time ───────────────────────────────────────────────────
    print("[TEST] Scanning one pixel at a time (white)...")
    strip.fill_strip(0, 0, 0)
    strip.update_strip()
    for i in range(LED_COUNT):
        strip.set_led_color(i, 255, 255, 255)
        strip.update_strip()
        time.sleep(0.15)
        strip.set_led_color(i, 0, 0, 0)
        strip.update_strip()

    # ── Rainbow spin ─────────────────────────────────────────────────────────
    print("[TEST] Rainbow spin (5 rounds)...")
    for _ in range(5 * LED_COUNT):
        for i in range(LED_COUNT):
            hue = (i / LED_COUNT + _ / LED_COUNT) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            strip.set_led_color(i, int(r * 255), int(g * 255), int(b * 255))
        strip.update_strip()
        time.sleep(0.05)

    # ── Off ───────────────────────────────────────────────────────────────────
    strip.fill_strip(0, 0, 0)
    strip.update_strip()
    print("\nAll tests done. If nothing lit up, check:")
    print("  1. BreadVolt GND → Pi Pin 6 (shared ground)")
    print("  2. LED VCC from BreadVolt 3.3V rail (not 5V, avoids logic level issue)")
    print("  3. LED DI → Pi Pin 19 (GPIO 10 / SPI0 MOSI)")

if __name__ == "__main__":
    main()
