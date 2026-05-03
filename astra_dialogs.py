"""Dark-red modal dialogs for ASTRA (night-vision friendly; avoids bright native messagebox)."""

from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
from typing import Any, Literal

IconKind = Literal["info", "warning", "error"]

# Matches CamGUI PLATE_RED_PAL / night theme — low luminance, red-shifted text.
_PAL: dict[str, str] = {
    "bg": "#1a0a05",
    "bg_alt": "#2d1410",
    "fg": "#ff6666",
    "fg_dim": "#cc5555",
    "accent": "#d11515",
    "accent_hover": "#ff4444",
    "btn_text": "#ffe6e6",
    "border": "#4d2020",
}

_PREFIX: dict[IconKind, str] = {"info": "", "warning": "", "error": ""}


def _master(parent: tk.Misc | None) -> tk.Misc:
    if parent is None:
        r = tk._default_root
        if r is None:
            raise RuntimeError("astra_dialogs: no parent and no Tk default root")
        return r.winfo_toplevel()
    return parent.winfo_toplevel()


def _show_modal(
    parent: tk.Misc | None,
    title: str,
    body: str,
    kind: IconKind,
    *,
    yesno: bool,
    default: bool = True,
) -> bool | None:
    master = _master(parent)
    top = tk.Toplevel(master)
    top.title(title)
    top.configure(bg=_PAL["bg"])
    try:
        top.transient(master)
    except tk.TclError:
        pass
    top.resizable(True, True)
    top.minsize(340, 140)

    result: list[bool | None] = [None if yesno else True]

    def finish(val: bool | None) -> None:
        result[0] = val
        try:
            top.grab_release()
        except tk.TclError:
            pass
        top.destroy()

    top.protocol("WM_DELETE_WINDOW", lambda: finish(False if yesno else True))

    pad = tk.Frame(top, bg=_PAL["bg"])
    pad.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

    msg_font = tkfont.nametofont("TkDefaultFont").copy()
    try:
        msg_font.configure(size=max(10, int(msg_font.cget("size"))))
    except tk.TclError:
        pass

    lbl = tk.Label(
        pad,
        text=_PREFIX[kind] + body,
        bg=_PAL["bg"],
        fg=_PAL["fg_dim"] if kind == "error" else _PAL["fg"],
        font=msg_font,
        justify=tk.LEFT,
        wraplength=420,
        anchor="w",
    )
    lbl.pack(fill=tk.BOTH, expand=True)

    btn_row = tk.Frame(pad, bg=_PAL["bg"])
    btn_row.pack(fill=tk.X, pady=(12, 0))

    def mk_btn(text: str, val: bool | None, is_primary: bool) -> tk.Button:
        bg = _PAL["accent"] if is_primary else _PAL["bg_alt"]
        fg = _PAL["btn_text"]
        return tk.Button(
            btn_row,
            text=text,
            command=lambda: finish(val),
            bg=bg,
            fg=fg,
            activebackground=_PAL["accent_hover"] if is_primary else _PAL["border"],
            activeforeground=fg,
            relief=tk.FLAT,
            padx=14,
            pady=6,
            highlightthickness=1,
            highlightbackground=_PAL["border"],
            cursor="hand2",
        )

    if yesno:
        # No on left, Yes on right (matches common confirmation layout).
        mk_btn("No", False, is_primary=not default).pack(side=tk.LEFT, padx=(0, 8))
        mk_btn("Yes", True, is_primary=default).pack(side=tk.RIGHT)
    else:
        mk_btn("OK", True, is_primary=True).pack(side=tk.RIGHT)

    top.update_idletasks()
    try:
        px = master.winfo_rootx() + max(20, (master.winfo_width() - top.winfo_reqwidth()) // 2)
        py = master.winfo_rooty() + max(20, (master.winfo_height() - top.winfo_reqheight()) // 2)
        top.geometry(f"+{px}+{py}")
    except tk.TclError:
        pass

    top.grab_set()
    try:
        top.focus_force()
    except tk.TclError:
        pass
    top.wait_window(top)
    return result[0]


def showinfo(title: str, message: str, *, parent: tk.Misc | None = None, **_kw: Any) -> str:
    """Same role as ``tkinter.messagebox.showinfo``; returns ``"ok"``."""
    _show_modal(parent, title, str(message), "info", yesno=False)
    return "ok"


def showwarning(title: str, message: str, *, parent: tk.Misc | None = None, **_kw: Any) -> str:
    _show_modal(parent, title, str(message), "warning", yesno=False)
    return "ok"


def showerror(title: str, message: str, *, parent: tk.Misc | None = None, **_kw: Any) -> str:
    _show_modal(parent, title, str(message), "error", yesno=False)
    return "ok"


def askyesno(
    title: str,
    message: str,
    *,
    parent: tk.Misc | None = None,
    default: bool = True,
    **_kw: Any,
) -> bool:
    """Same role as ``tkinter.messagebox.askyesno``."""
    r = _show_modal(parent, title, str(message), "warning", yesno=True, default=default)
    return bool(r)
