"""
RGB common-cathode status LED for mount motor activity (Raspberry Pi GPIO).

Idle: purple (red + blue). When movement commands are active: brief red flash,
then solid red until all activity ends.

Wiring (BCM numbering): connect each LED anode through a resistor to the GPIO
listed below; common cathode to GND.

Environment (optional):
  ASTRA_MOTOR_RGB_LED   — set to "0" / "false" / "off" to disable
  ASTRA_MOTOR_RGB_R   — BCM pin for red   (default 17)
  ASTRA_MOTOR_RGB_G   — BCM pin for green (default 27)
  ASTRA_MOTOR_RGB_B   — BCM pin for blue  (default 22)

Requires RPi.GPIO on the Pi (typically preinstalled on Raspberry Pi OS).
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable, List, Optional

_lock = threading.Lock()
_depth = 0
_gpio = None  # type: ignore[var-annotated]
pin_r = pin_g = pin_b = -1
_initialized = False
_disabled = False
_pending_cancel: List[Callable[[], None]] = []


def _env_bool(name: str, default: bool = True) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "off", "no", "")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)).strip(), 10)
    except ValueError:
        return default


def _lazy_init() -> bool:
    global _initialized, _disabled, _gpio, pin_r, pin_g, pin_b
    if _initialized:
        return not _disabled
    _initialized = True
    if not _env_bool("ASTRA_MOTOR_RGB_LED", True):
        _disabled = True
        return False
    pin_r = _env_int("ASTRA_MOTOR_RGB_R", 17)
    pin_g = _env_int("ASTRA_MOTOR_RGB_G", 27)
    pin_b = _env_int("ASTRA_MOTOR_RGB_B", 22)
    try:
        import RPi.GPIO as GPIO  # type: ignore[import-untyped]

        _gpio = GPIO
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        for p in (pin_r, pin_g, pin_b):
            GPIO.setup(p, GPIO.OUT, initial=GPIO.LOW)
        _set_pins(1, 0, 1)  # purple idle
    except Exception:
        _gpio = None
        _disabled = True
        return False
    return True


def _set_pins(r: int, g: int, b: int) -> None:
    if _gpio is None:
        return
    _gpio.output(pin_r, _gpio.HIGH if r else _gpio.LOW)
    _gpio.output(pin_g, _gpio.HIGH if g else _gpio.LOW)
    _gpio.output(pin_b, _gpio.HIGH if b else _gpio.LOW)


def _purple() -> None:
    _set_pins(1, 0, 1)


def _red() -> None:
    _set_pins(1, 0, 0)


def _off() -> None:
    _set_pins(0, 0, 0)


def _flash_then_red() -> None:
    """Blocking flash (~0.56 s) then solid red; caller must hold _lock."""
    if _gpio is None:
        return
    for _ in range(4):
        _red()
        time.sleep(0.07)
        _off()
        time.sleep(0.07)
    _red()


def _remove_cancel(fn: Callable[[], None]) -> None:
    try:
        _pending_cancel.remove(fn)
    except ValueError:
        pass


def ensure_idle_display() -> None:
    """Set up GPIO when available and show purple if no movement is active."""
    if not _lazy_init():
        return
    with _lock:
        if _depth <= 0:
            _depth = 0
            _purple()


def begin_movement() -> None:
    """Call when a motor movement command starts (nudge press or slew command)."""
    global _depth
    if not _lazy_init():
        return
    with _lock:
        was = _depth
        _depth += 1
        if was == 0:
            _flash_then_red()
        else:
            _red()


def end_movement() -> None:
    """Call when a movement session ends (nudge release, timer, etc.)."""
    global _depth
    if not _initialized or _disabled or _gpio is None:
        return
    with _lock:
        if _depth <= 0:
            _depth = 0
            _purple()
            return
        _depth -= 1
        if _depth == 0:
            _purple()
        else:
            _red()


def schedule_end_movement(delay_s: float) -> None:
    """
    Pair with begin_movement() for commands that return before the slew finishes
    (e.g. GoTo). After delay_s, end_movement() runs once on a timer thread.
    """
    if not _lazy_init():
        return

    done = threading.Lock()
    ended = False

    def _end_once() -> None:
        nonlocal ended
        with done:
            if ended:
                return
            ended = True
        end_movement()

    timer_holder: List[Optional[threading.Timer]] = [None]

    def _cancel() -> None:
        with done:
            if ended:
                return
            t = timer_holder[0]
            if t is not None:
                t.cancel()
        _end_once()
        with _lock:
            _remove_cancel(_cancel)

    def _timer_cb() -> None:
        _end_once()
        with _lock:
            _remove_cancel(_cancel)

    t = threading.Timer(max(0.05, float(delay_s)), _timer_cb)
    t.daemon = True
    timer_holder[0] = t
    with _lock:
        _pending_cancel.append(_cancel)
    t.start()


def cancel_scheduled_movement_ends() -> None:
    """Cancel pending slew timers and apply one end per cancelled session (e.g. STOP)."""
    with _lock:
        cancels = list(_pending_cancel)
        _pending_cancel.clear()
    for c in cancels:
        c()


def cleanup() -> None:
    """Release GPIO (e.g. on app exit)."""
    global _gpio, _initialized, _disabled, _depth
    cancel_scheduled_movement_ends()
    with _lock:
        _depth = 0
        if _gpio is not None:
            try:
                _gpio.cleanup()
            except Exception:
                pass
            _gpio = None
        _initialized = False
        _disabled = False
