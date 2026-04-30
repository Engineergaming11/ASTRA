import tkinter as tk
from tkinter import ttk
import customtkinter as ctk
import numpy as np
from skyfield.api import Star
from datetime import datetime
import sys

from sky_ephemeris import (
    DEFAULT_ELEV_M,
    DEFAULT_LAT,
    DEFAULT_LON,
    DESCRIPTIONS,
    NEBULAE,
    SOLAR_SYSTEM_BODIES,
    TYCHO_PRESET_STARS,
    catalog,
    eph,
    get_messier_coordinates,
    is_above_horizon,
    ts,
    write_livewcs,
    livewcs_path,
)

# THEME CONFIGURATION
THEMES = {
    "day": {
        "bg_primary": "#f5f5f5",
        "bg_secondary": "#ebebeb",
        "bg_tertiary": "#e0e0e0",
        "fg_primary": "#1a1a1a",
        "fg_secondary": "#4a4a4a",
        "fg_tertiary": "#7a7a7a",
        "accent": "#0b5c8f",
        "accent_light": "#1a7ab5",
        "accent_dark": "#083d5e",
        "canvas_bg": "#ffffff",
        "border": "#cccccc",
    },
    "night": {
        "bg_primary": "#1a0a05",
        "bg_secondary": "#2d1410",
        "bg_tertiary": "#3d1f1a",
        "fg_primary": "#ff6666",
        "fg_secondary": "#ff6666",
        "fg_tertiary": "#ff6666",
        "accent": "#ff4444",
        "accent_light": "#ff6666",
        "accent_dark": "#cc0000",
        "canvas_bg": "#0d0503",
        "border": "#4d2020",
    }
}

WCS_PATH = str(livewcs_path())


class StarTrackerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ASTRA Sky Tracker")
        self.root.geometry("700x700")

        self.night_mode = False
        self.theme = THEMES["day"]

        self.active_mode = None
        self.active_target = None
        self.cached_star = None

        self.setup_ui()
        self.apply_theme()
        self.live_update()

    def setup_ui(self):
        # TOP BAR
        self.top_bar = tk.Frame(self.root)
        self.top_bar.pack(fill=tk.X, padx=12, pady=8)

        self.title_label = tk.Label(
            self.top_bar, text="ASTRA SKY TRACKER", font=("Segoe UI", 16, "bold")
        )
        self.title_label.pack(side=tk.LEFT)

        spacer = tk.Frame(self.top_bar)
        spacer.pack(side=tk.LEFT, expand=True)

        self.theme_btn = ctk.CTkButton(
            self.top_bar,
            text="☀ Day Mode",
            command=self.toggle_theme,
            font=("Segoe UI", 9),
            width=120,
            height=32,
            fg_color="#e0e0e0",
            text_color="#1a1a1a",
            hover_color="#2d7a3e",
            corner_radius=6
        )
        self.theme_btn.pack(side=tk.RIGHT)

        # MAIN CONTAINER
        self.main = tk.Frame(self.root)
        self.main.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        # LEFT PANEL
        self.left = tk.Frame(self.main)
        self.left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        tk.Label(
            self.left, text="LIVE OUTPUT", font=("Segoe UI", 11, "bold")
        ).pack(anchor="w", pady=(0, 8))

        # Height=7 to fit RA, DEC, AZ, ALT lines comfortably
        self.output_text = tk.Text(
            self.left, height=7, font=("Courier New", 14), width=28,
            relief=tk.FLAT, borderwidth=2
        )
        self.output_text.pack(pady=4, anchor="w", fill=tk.X)
        self.output_text.insert("1.0", "Idle. Select a\ntarget.\n")
        self.output_text.config(state=tk.DISABLED)

        tk.Label(
            self.left, text="OBJECT DESCRIPTION", font=("Segoe UI", 11, "bold")
        ).pack(anchor="w", pady=(12, 4))

        self.description_text = tk.Text(
            self.left, font=("Segoe UI", 11), wrap=tk.WORD, width=28,
            relief=tk.FLAT, borderwidth=2
        )
        self.description_text.pack(pady=4, fill=tk.BOTH, expand=True)
        self.description_text.insert("1.0", "No object selected.\n")
        self.description_text.config(state=tk.DISABLED)

        # RIGHT PANEL
        self.right = tk.Frame(self.main)
        self.right.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Label(
            self.right, text="TARGET TRACKING", font=("Segoe UI", 11, "bold")
        ).pack(anchor="w", pady=(0, 8))

        def create_combo(label_text, values, command=None):
            if label_text:
                tk.Label(self.right, text=label_text, font=("Segoe UI", 9)).pack(anchor="w", pady=(8, 0))
            combo = ctk.CTkComboBox(
                self.right,
                values=values,
                state="readonly",
                width=240,
                fg_color="white",
                button_color="#2d7a3e",
                button_hover_color="#4a9f54",
                border_color="#cccccc",
                dropdown_fg_color="white",
                dropdown_hover_color="#f0f0f0",
                corner_radius=6,
                command=command
            )
            combo.pack(pady=4)
            return combo

        self.target_type = create_combo(
            "Target Type:",
            ["Star", "Solar System Body", "Messier Object"],
            self.on_type_change
        )
        self.target_type.set("Star")

        self.preset_combo = create_combo(
            "Preset Objects:",
            list(TYCHO_PRESET_STARS.keys()),
            self.on_preset_select
        )
        self.preset_combo.set("")

        tk.Label(self.right, text="Custom Tycho ID:", font=("Segoe UI", 9)).pack(anchor="w", pady=(8, 0))

        self.custom_entry = ctk.CTkEntry(
            self.right, width=240, height=32,
            fg_color="white", text_color="#1a1a1a",
            border_color="#cccccc", corner_radius=6
        )
        self.custom_entry.pack(pady=4)

        self.track_btn = ctk.CTkButton(
            self.right,
            text="Track Target",
            command=self.track_target,
            font=("Segoe UI", 10, "bold"),
            height=40,
            fg_color="#2d7a3e",
            text_color="white",
            hover_color="#4a9f54",
            corner_radius=6
        )
        self.track_btn.pack(pady=12, fill=tk.X)

        ttk.Separator(self.right).pack(fill=tk.X, pady=10)

        tk.Label(self.right, text="STATUS", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 4))

        self.status_label = tk.Label(
            self.right,
            text="No target selected",
            font=("Segoe UI", 9, "italic"),
            wraplength=240,
            justify=tk.LEFT,
            anchor=tk.W
        )
        self.status_label.pack(fill=tk.X, pady=4)

        ttk.Separator(self.right).pack(fill=tk.X, pady=10)

        tk.Label(self.right, text="RELATED STARS", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 4))

        self.related_label = tk.Label(self.right, text="Closest Stars:", font=("Segoe UI", 9))
        self.related_combo = create_combo("", [], self.on_related_select)

        self.related_label.pack_forget()
        self.related_combo.pack_forget()

    def apply_theme(self):
        t = self.theme

        self.root.configure(bg=t["bg_primary"])
        for frame in [self.top_bar, self.main, self.left, self.right]:
            try:
                frame.configure(bg=t["bg_primary"])
            except:
                pass

        for label in [self.title_label, self.status_label, self.related_label]:
            label.configure(bg=t["bg_primary"], fg=t["fg_primary"])

        for text_widget in [self.output_text, self.description_text]:
            text_widget.configure(
                bg=t["bg_secondary"],
                fg=t["fg_primary"],
                insertbackground=t["fg_primary"]
            )

        for container in [self.left, self.right]:
            for widget in container.winfo_children():
                if isinstance(widget, tk.Label) and widget not in (self.status_label, self.related_label, self.title_label):
                    try:
                        widget.configure(bg=t["bg_primary"], fg=t["fg_primary"])
                    except:
                        pass

        self.track_btn.configure(fg_color=t["accent"], hover_color=t["accent_light"])

        if self.night_mode:
            self.theme_btn.configure(fg_color=t["accent"], text_color="white", hover_color=t["accent_light"])
            combo_settings = {
                "fg_color": t["bg_secondary"],
                "text_color": t["fg_primary"],
                "border_color": t["border"],
                "button_color": t["accent"],
                "button_hover_color": t["accent_light"],
                "dropdown_fg_color": t["bg_tertiary"],
                "dropdown_hover_color": t["bg_secondary"]
            }
            self.custom_entry.configure(
                fg_color=t["bg_secondary"],
                text_color=t["fg_primary"],
                border_color=t["border"]
            )
        else:
            self.theme_btn.configure(fg_color=t["bg_tertiary"], text_color=t["fg_primary"], hover_color=t["accent"])
            combo_settings = {
                "fg_color": "white",
                "text_color": "#1a1a1a",
                "border_color": "#cccccc",
                "button_color": t["accent"],
                "button_hover_color": t["accent_light"],
                "dropdown_fg_color": "white",
                "dropdown_hover_color": "#f0f0f0"
            }
            self.custom_entry.configure(fg_color="white", text_color="#1a1a1a", border_color="#cccccc")

        for combo in [self.target_type, self.preset_combo, self.related_combo]:
            combo.configure(**combo_settings)

    def toggle_theme(self):
        self.night_mode = not self.night_mode
        self.theme = THEMES["night"] if self.night_mode else THEMES["day"]
        self.theme_btn.configure(text="🌙 Night Mode" if self.night_mode else "☀ Day Mode")
        self.apply_theme()

    def on_type_change(self, choice=None):
        target_type = self.target_type.get()
        presets_map = {
            "Star": TYCHO_PRESET_STARS,
            "Solar System Body": SOLAR_SYSTEM_BODIES,
            "Messier Object": NEBULAE
        }
        self.preset_combo.configure(values=list(presets_map[target_type].keys()))
        self.preset_combo.set("")
        if target_type == "Star":
            self.custom_entry.configure(state="normal")
        else:
            self.custom_entry.delete(0, tk.END)
            self.custom_entry.configure(state="disabled")

    def on_preset_select(self, choice=None):
        if self.target_type.get() == "Star":
            preset_name = self.preset_combo.get()
            tycho_id = TYCHO_PRESET_STARS.get(preset_name, "")
            self.custom_entry.delete(0, tk.END)
            self.custom_entry.insert(0, tycho_id)

    def on_related_select(self, choice=None):
        selected = self.related_combo.get()
        if not selected:
            return
        tycho_id = selected.split(" - ")[1].strip() if " - " in selected else \
            selected.replace("TYC ", "").split(" (mag")[0].strip()
        if tycho_id:
            self.custom_entry.delete(0, tk.END)
            self.custom_entry.insert(0, tycho_id)
            self.track_target()

    def tycho_lookup(self, tyc_id):
        match = catalog.loc[catalog["TYC_ID"].str.strip() == tyc_id]
        if match.empty:
            return None
        r = match.iloc[0]
        return Star(
            ra_hours=r["RA_deg"] / 15.0,
            dec_degrees=r["Dec_deg"],
            ra_mas_per_year=r["pmRA_masyr"],
            dec_mas_per_year=r["pmDec_masyr"],
        )

    @staticmethod
    def _to_dms(degrees, signed=False):
        """Convert decimal degrees to ±DDDd MM' SS.S\" string."""
        sign = "-" if degrees < 0 else ("+" if signed else "")
        d = abs(degrees)
        deg = int(d)
        minutes = int((d - deg) * 60)
        seconds = (d - deg - minutes / 60) * 3600
        return f"{sign}{deg:03d}\u00b0{minutes:02d}'{seconds:04.1f}\""

    def write_output(self, ra, dec, object_name="Unknown"):
        """Write coordinates to file atomically as JSON and update GUI"""
        write_livewcs(WCS_PATH, ra, dec, object_name)

        # Compute Az/Alt and display in live output box (default Steward site)
        az_alt_lines = ""
        try:
            if self.active_mode in ("star", "messier") and self.cached_star is not None:
                above, alt, az = is_above_horizon(
                    DEFAULT_LAT, DEFAULT_LON, DEFAULT_ELEV_M, target_star=self.cached_star
                )
            elif self.active_mode == "body" and self.active_target:
                above, alt, az = is_above_horizon(
                    DEFAULT_LAT, DEFAULT_LON, DEFAULT_ELEV_M, body_key=self.active_target
                )
            else:
                above, alt, az = None, None, None

            if alt is not None:
                horizon_indicator = "\n▲ Above Horizon" if above else "\n▼ Below Horizon"
                az_alt_lines = (
                    f"\nAZ : {self._to_dms(az)}"
                    f"\nALT: {self._to_dms(alt, signed=True)} {horizon_indicator}"
                )
        except Exception:
            pass

        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", f"RA : {ra.hstr()}\nDEC: {dec.dstr()}{az_alt_lines}\n")
        self.output_text.config(state=tk.DISABLED)

        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {object_name} | RA: {ra.hstr()} | DEC: {dec.dstr()}", flush=True)

    def update_description(self, object_name):
        """Update the description text box."""
        description = DESCRIPTIONS.get(object_name, "No description available for this object.")
        self.description_text.config(state=tk.NORMAL)
        self.description_text.delete("1.0", tk.END)
        self.description_text.insert("1.0", description)
        self.description_text.config(state=tk.DISABLED)

    def find_closest_stars(self, tycho_id, num_stars=5):
        target_match = catalog.loc[catalog["TYC_ID"].str.strip() == tycho_id]
        if target_match.empty:
            return []

        target = target_match.iloc[0]
        target_ra = target["RA_deg"]
        target_dec = target["Dec_deg"]

        dec_rad = np.radians(target_dec)
        ra_diff = (catalog["RA_deg"] - target_ra) * np.cos(dec_rad)
        dec_diff = catalog["Dec_deg"] - target_dec
        distances = np.sqrt(ra_diff ** 2 + dec_diff ** 2)

        catalog_copy = catalog.copy()
        catalog_copy["distance"] = distances

        mag_column = "VT_mag" if "VT_mag" in catalog_copy.columns else "BT_mag"
        filtered = catalog_copy[
            (catalog_copy["distance"] > 0) &
            (catalog_copy[mag_column] < 5)
        ]

        closest = filtered.nsmallest(num_stars, "distance")

        results = []
        for _, star in closest.iterrows():
            tyc = star["TYC_ID"].strip()
            mag = star[mag_column]
            star_name = next((name for name, tid in TYCHO_PRESET_STARS.items() if tid == tyc), None)
            if star_name:
                results.append(f"{star_name} (mag {mag:.1f}) - {tyc}")
            else:
                results.append(f"TYC {tyc} (mag {mag:.1f})")

        return results

    def show_related_stars(self, tycho_id):
        closest = self.find_closest_stars(tycho_id)
        if closest:
            self.related_combo.configure(values=closest)
            self.related_label.pack(anchor="w", pady=(4, 0))
            self.related_combo.pack(pady=4, anchor="w")
        else:
            self.hide_related_stars()

    def hide_related_stars(self):
        self.related_label.pack_forget()
        self.related_combo.pack_forget()
        self.related_combo.set("")

    def track_target(self):
        target_type = self.target_type.get()
        if target_type == "Star":
            self._track_star()
        elif target_type == "Solar System Body":
            self._track_body()
        else:
            self._view_messier()

    def _track_star(self):
        tyc_id = self.custom_entry.get().strip()
        if not tyc_id:
            self.status_label.config(text="Error: Please enter a Tycho ID or select a preset star")
            return

        star = self.tycho_lookup(tyc_id)
        if not star:
            self.status_label.config(text=f"Error: Star '{tyc_id}' not found in catalog")
            return

        self.active_mode = "star"
        self.active_target = tyc_id
        self.cached_star = star

        preset_name = next((name for name, tid in TYCHO_PRESET_STARS.items() if tid == tyc_id), None)

        if preset_name:
            self.status_label.config(text=f"Tracking: {preset_name} ({tyc_id})")
            self.update_description(preset_name)
            print(f"\n>>> Now tracking star: {preset_name} ({tyc_id})", flush=True)
        else:
            self.status_label.config(text=f"Tracking: Star {tyc_id}")
            self.update_description("Unknown Star")
            print(f"\n>>> Now tracking star: {tyc_id}", flush=True)

        self.show_related_stars(tyc_id)

    def _track_body(self):
        preset_name = self.preset_combo.get()
        if not preset_name:
            self.status_label.config(text="Error: Please select a solar system body")
            return

        body_key = SOLAR_SYSTEM_BODIES.get(preset_name)
        if not body_key:
            self.status_label.config(text="Error: Invalid body selection")
            return

        self.active_mode = "body"
        self.active_target = body_key
        self.cached_star = None
        self.status_label.config(text=f"Tracking: {preset_name}")
        self.update_description(preset_name)
        print(f"\n>>> Now tracking body: {preset_name}", flush=True)
        self.hide_related_stars()

    def _view_messier(self):
        preset_name = self.preset_combo.get()
        if not preset_name:
            self.status_label.config(text="Error: Please select a Messier object")
            return

        nebula_id = NEBULAE.get(preset_name)
        if not nebula_id:
            self.status_label.config(text="Error: Invalid Messier object selection")
            return

        ra_deg, dec_deg = get_messier_coordinates(nebula_id)

        if ra_deg is None or dec_deg is None:
            self.status_label.config(text=f"Error: Coordinates not available for {nebula_id}")
            return

        messier_star = Star(ra_hours=ra_deg / 15.0, dec_degrees=dec_deg)

        self.active_mode = "messier"
        self.active_target = nebula_id
        self.cached_star = messier_star

        self.status_label.config(text=f"Tracking: {preset_name} ({nebula_id})")
        self.update_description(preset_name)
        print(f"\n>>> Now tracking Messier object: {preset_name} ({nebula_id})", flush=True)
        self.hide_related_stars()

    def live_update(self):
        """Continuous update loop for tracking"""
        if self.active_mode in ("star", "messier") and self.cached_star:
            t = ts.now()
            ra, dec, _ = eph["earth"].at(t).observe(self.cached_star).apparent().radec("date")

            if self.active_mode == "star":
                object_name = next(
                    (name for name, tid in TYCHO_PRESET_STARS.items() if tid == self.active_target),
                    f"TYC {self.active_target}"
                )
            else:
                object_name = next(
                    (display for display, mid in NEBULAE.items() if mid == self.active_target),
                    self.active_target
                )

            self.write_output(ra, dec, object_name)

        elif self.active_mode == "body":
            body = eph[self.active_target]
            t = ts.now()
            ra, dec, _ = eph["earth"].at(t).observe(body).apparent().radec("date")

            object_name = next(
                (name for name, key in SOLAR_SYSTEM_BODIES.items() if key == self.active_target),
                self.active_target
            )

            self.write_output(ra, dec, object_name)

        self.root.after(500, self.live_update)


if __name__ == "__main__":
    print("=" * 50)
    print("ASTRA Sky Tracker Starting...")
    print("=" * 50)
    sys.stdout.flush()

    root = tk.Tk()
    app = StarTrackerGUI(root)
    root.mainloop()