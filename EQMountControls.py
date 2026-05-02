"""
iOptron HAE16C Mount Controller — standalone window.
Re-exports driver symbols from ioptron_mount for backward compatibility.
"""

import customtkinter as ctk

from ui_display_profile import mount_standalone_geometry_minsize, ui_compact

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

from ioptron_mount import (  # noqa: F401
    BAUD_RATE,
    TIMEOUT,
    WIFI_DEFAULT_IP,
    WIFI_DEFAULT_PORT,
    TRACK_SIDEREAL,
    TRACK_LUNAR,
    TRACK_SOLAR,
    IOptronHAE,
    IOptronHAEWifi,
    IOptronMount,
    _deg_to_dms,
    _deg_to_hms,
    _encode_dec,
    _encode_ra,
    _parse_angle_dms,
    _parse_dec_input,
    _parse_ra_input,
)
from mount_controls_frame import MountControlsFrame, STANDALONE_PAL


class MountControlApp(ctk.CTk):
    """Standalone top-level window hosting MountControlsFrame."""

    def __init__(self):
        super().__init__()
        self.title("iOptron HAE16C  ·  EQ Controller")
        _g, _mn = mount_standalone_geometry_minsize(compact=ui_compact(self))
        self.geometry(_g)
        self.minsize(_mn[0], _mn[1])
        self.configure(fg_color=STANDALONE_PAL["bg"])
        self._frame = MountControlsFrame(self, standalone=True)
        self._frame.pack(fill="both", expand=True)
        self.protocol("WM_DELETE_WINDOW", self._frame.on_window_close)


if __name__ == "__main__":
    app = MountControlApp()
    app.mainloop()
