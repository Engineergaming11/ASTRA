"""Bidirectional scroll regions using tk.Canvas + horizontal and vertical scrollbars.

CustomTkinter's ``CTkScrollableFrame`` only supports one orientation at a time; this
module provides both axes so narrow or short windows remain usable.
"""
from __future__ import annotations

import sys
import tkinter as tk
from dataclasses import dataclass


@dataclass
class XYScrollArea:
    outer: tk.Frame
    inner: tk.Frame
    canvas: tk.Canvas
    vsb: tk.Scrollbar
    hsb: tk.Scrollbar
    corner: tk.Frame


def _is_descendant(widget: tk.Misc | None, ancestor: tk.Misc | None) -> bool:
    w = widget
    while w is not None:
        if w is ancestor:
            return True
        w = getattr(w, "master", None)
    return False


def _should_skip_wheel_target(widget: tk.Misc | None) -> bool:
    """Leave wheel to widgets that manage their own scrolling."""
    w = widget
    while w is not None:
        if isinstance(w, (tk.Text, tk.Listbox, tk.Entry)):
            return True
        cls_name = w.__class__.__name__
        if cls_name in ("CTkTextbox", "CTkScrollableFrame"):
            return True
        w = getattr(w, "master", None)
    return False


def create_xy_scroll_area(parent: tk.Misc, *, bg: str = "#f0f0f0") -> XYScrollArea:
    """Create a frame with bottom and side scrollbars; pack/grid children on ``inner``."""
    outer = tk.Frame(parent, bg=bg)
    outer.grid_rowconfigure(0, weight=1)
    outer.grid_columnconfigure(0, weight=1)

    canvas = tk.Canvas(outer, highlightthickness=0, borderwidth=0, bg=bg)
    vsb = tk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
    hsb = tk.Scrollbar(outer, orient=tk.HORIZONTAL, command=canvas.xview)
    inner = tk.Frame(canvas, bg=bg)
    corner = tk.Frame(outer, bg=bg, width=17, height=17)

    canvas.create_window((0, 0), window=inner, anchor="nw")

    def _update_scrollregion(_event=None):
        canvas.update_idletasks()
        bbox = canvas.bbox("all")
        if bbox:
            canvas.configure(scrollregion=bbox)

    inner.bind("<Configure>", lambda e: _update_scrollregion())

    canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

    canvas.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")
    corner.grid(row=1, column=1, sticky="nsew")

    return XYScrollArea(
        outer=outer,
        inner=inner,
        canvas=canvas,
        vsb=vsb,
        hsb=hsb,
        corner=corner,
    )


def _yscroll_units(event: tk.Event) -> int:
    """Tk units to pass to ``yview_scroll`` (positive = scroll down)."""
    if sys.platform == "darwin":
        d = int(getattr(event, "delta", 0) or 0)
        return -d
    d = int(getattr(event, "delta", 0) or 0)
    if not d:
        return 0
    return int(-d / 120)


def _xscroll_units(event: tk.Event) -> int:
    if sys.platform == "darwin":
        d = int(getattr(event, "delta", 0) or 0)
        return -d
    d = int(getattr(event, "delta", 0) or 0)
    if not d:
        return 0
    return int(-d / 120)


def attach_mousewheel_dispatch(
    root: tk.Misc,
    areas: list[XYScrollArea],
    *,
    extra_area_lists: list[list[XYScrollArea]] | None = None,
) -> None:
    """Route mouse wheel to the XY scroll area under the cursor."""

    def _all_areas() -> list[XYScrollArea]:
        out: list[XYScrollArea] = list(areas)
        if extra_area_lists:
            for lst in extra_area_lists:
                out.extend(lst)
        return out

    def _on_mousewheel(event: tk.Event):
        if _should_skip_wheel_target(getattr(event, "widget", None)):
            return
        w = getattr(event, "widget", None)
        du = _yscroll_units(event)
        if not du:
            return
        for a in _all_areas():
            if _is_descendant(w, a.inner):
                a.canvas.yview_scroll(du, "units")
                return

    def _on_shift_mousewheel(event: tk.Event):
        if _should_skip_wheel_target(getattr(event, "widget", None)):
            return
        w = getattr(event, "widget", None)
        du = _xscroll_units(event)
        if not du:
            return
        for a in _all_areas():
            if _is_descendant(w, a.inner):
                a.canvas.xview_scroll(du, "units")
                return

    root.bind_all("<MouseWheel>", _on_mousewheel, add="+")
    root.bind_all("<Shift-MouseWheel>", _on_shift_mousewheel, add="+")

    def _linux_up(event: tk.Event):
        if _should_skip_wheel_target(getattr(event, "widget", None)):
            return
        w = getattr(event, "widget", None)
        for a in _all_areas():
            if _is_descendant(w, a.inner):
                a.canvas.yview_scroll(-1, "units")
                return

    def _linux_down(event: tk.Event):
        if _should_skip_wheel_target(getattr(event, "widget", None)):
            return
        w = getattr(event, "widget", None)
        for a in _all_areas():
            if _is_descendant(w, a.inner):
                a.canvas.yview_scroll(1, "units")
                return

    root.bind_all("<Button-4>", _linux_up, add="+")
    root.bind_all("<Button-5>", _linux_down, add="+")


def theme_xy_area(area: XYScrollArea | None, bg: str, *, trough: str | None = None) -> None:
    """Apply a single background colour to native Tk pieces (for day/night theme).

    ``trough`` colours the scrollbar trough so it does not flash light grey/white
    against dark panels (falls back to ``bg`` when omitted).
    """
    if area is None:
        return
    trough_c = trough if trough is not None else bg
    try:
        area.outer.configure(bg=bg)
        area.inner.configure(bg=bg)
        area.canvas.configure(bg=bg)
        area.corner.configure(bg=bg)
        for sb in (area.vsb, area.hsb):
            try:
                sb.configure(
                    bg=trough_c,
                    troughcolor=trough_c,
                    activebackground=trough_c,
                    highlightthickness=0,
                )
            except tk.TclError:
                pass
    except tk.TclError:
        pass
