"""Layout helpers for 1024x600-class panels (e.g. Raspberry Pi WSVGA touch displays)."""

from __future__ import annotations

import os
import tkinter as tk

# Typical 7" DSI / WSVGA; allow a few extra scanlines for overscan-safe modes.
SMALL_DISPLAY_MAX_W = 1024
SMALL_DISPLAY_MAX_H = 640


def ui_compact_by_env() -> bool | None:
    v = (os.environ.get("ASTRA_UI_COMPACT") or "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return None


def ui_compact_for_screen(sw: int, sh: int) -> bool:
    return sw <= SMALL_DISPLAY_MAX_W and sh <= SMALL_DISPLAY_MAX_H


def ui_compact(root: tk.Misc) -> bool:
    """True for small displays, or when ASTRA_UI_COMPACT=1 (force on/off with 0/false)."""
    o = ui_compact_by_env()
    if o is not None:
        return o
    try:
        sw = int(root.winfo_screenwidth())
        sh = int(root.winfo_screenheight())
    except tk.TclError:
        return False
    return ui_compact_for_screen(sw, sh)


def main_window_geometry_minsize(*, compact: bool) -> tuple[str, tuple[int, int]]:
    if compact:
        return "1024x600", (720, 480)
    return "1680x860", (1280, 700)


def camgui_pane_minsizes(*, compact: bool) -> tuple[int, int, int, int]:
    """(left_pane, pan_inner, center_column, mount_column) minimum widths for tk.PanedWindow."""
    if compact:
        return (200, 540, 240, 240)
    return (260, 980, 380, 520)


def mount_standalone_geometry_minsize(*, compact: bool) -> tuple[str, tuple[int, int]]:
    if compact:
        return "1008x580", (640, 400)
    return "980x820", (860, 700)
