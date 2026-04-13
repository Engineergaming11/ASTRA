import zwoasi as asi
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import threading
import os
import subprocess
import json
import time
from datetime import datetime
from astropy.io import fits



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
        "canvas_bg": "#0d0503",
        "border": "#4d2020",
    }
}


class ZWOCameraGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera Control")
        self.root.geometry("1100x750")

        # Theme settings
        self.night_mode = False
        self.theme = THEMES["day"]

        # Initialize camera
        self.camera = None
        self.is_capturing = False
        self.is_recording = False
        self.video_writer = None
        self.save_path = "/Users/alexis/ASICAP/CapGUI"
        self.available_cameras = []
        self.camera_initialized = False

        # Create save directory if it doesn't exist
        try:
            os.makedirs(self.save_path, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create save directory {self.save_path}: {e}")
            self.save_path = os.path.expanduser("~")

        # Store full sensor dimensions
        self.full_sensor_width = None
        self.full_sensor_height = None

        # LiveWCS polling
        self.livewcs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LiveWCS.txt")
        self._livewcs_last_mtime = None
        self._livewcs_poll_running = False

        # Initialize ZWO library
        try:
            asi.init("libASICamera2.dylib")
            num_cameras = asi.get_num_cameras()
            if num_cameras > 0:
                for i in range(num_cameras):
                    camera_info = asi.Camera(i).get_camera_property()
                    self.available_cameras.append((i, camera_info['Name']))
                print(f"Cameras found: {num_cameras}")
            else:
                print("No cameras found - GUI will run in demo mode")
        except Exception as e:
            print(f"Failed to initialize camera library: {e}")
            print("GUI will run in demo mode")

        self.setup_ui()
        self.apply_theme()

        # Start polling LiveWCS.txt for sky tracker data
        self._livewcs_poll_running = True
        self._poll_livewcs()

        # Show viewing mode selection popup on startup
        self.viewing_mode = None
        self.root.after(100, self._show_viewing_mode_popup)

        # Auto-select first camera if available
        if self.available_cameras:
            self.camera_select_var.set(f"{self.available_cameras[0][0]}: {self.available_cameras[0][1]}")
            self.connect_camera()

    def _show_viewing_mode_popup(self):
        """Show a startup dialog asking the user what type of viewing they are doing."""
        popup = ctk.CTkToplevel(self.root)
        popup.title("Viewing Mode")
        popup.geometry("360x200")
        popup.resizable(False, False)
        popup.grab_set()  # Make it modal

        # Center over the main window
        self.root.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 180
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 100
        popup.geometry(f"+{x}+{y}")

        ctk.CTkLabel(
            popup,
            text="What type of viewing are you doing?",
            font=("Segoe UI", 14, "bold"),
            wraplength=320,
        ).pack(pady=(28, 20))

        btn_frame = ctk.CTkFrame(popup, fg_color="transparent")
        btn_frame.pack(pady=(0, 20))

        def select_mode(mode):
            self.viewing_mode = mode
            print(f"Viewing mode selected: {mode}")
            popup.destroy()
            if mode == "day":
                self.root.after(50, self._show_solar_filter_popup)
            elif mode == "night":
                self.root.after(50, lambda: self._show_polar_motors_popup(viewing_mode="night"))
            elif mode == "deep_sky":
                self.root.after(50, self._show_polar_motors_on_popup)

        for label, mode in [("☀  Day Time", "day"), ("🌙  Night Time", "night"), ("🔭  Deep Sky", "deep_sky")]:
            ctk.CTkButton(
                btn_frame,
                text=label,
                command=lambda m=mode: select_mode(m),
                font=("Segoe UI", 11, "bold"),
                width=90,
                height=38,
                fg_color="#0b5c8f",
                hover_color="#1a7ab5",
                corner_radius=8,
            ).pack(side=tk.LEFT, padx=6)

    def _show_solar_filter_popup(self, first_time=True):
        """Warn the user to attach a solar filter before daytime solar viewing."""
        popup = ctk.CTkToplevel(self.root)
        popup.title("Safety Check")
        popup.resizable(False, False)
        popup.grab_set()

        # Center over the main window
        self.root.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 185
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 110
        popup.geometry(f"370x220+{x}+{y}")

        # Warning icon + message
        ctk.CTkLabel(
            popup,
            text="⚠️",
            font=("Segoe UI", 36),
        ).pack(pady=(22, 4))

        ctk.CTkLabel(
            popup,
            text="Do you have a solar filter on?",
            font=("Segoe UI", 14, "bold"),
        ).pack()

        if not first_time:
            ctk.CTkLabel(
                popup,
                text="A solar filter is required for daytime viewing.\nDo NOT point at the sun without one!",
                font=("Segoe UI", 10),
                text_color="#ff4444",
                wraplength=320,
                justify="center",
            ).pack(pady=(6, 0))

        btn_frame = ctk.CTkFrame(popup, fg_color="transparent")
        btn_frame.pack(pady=(16, 20))

        def on_yes():
            popup.destroy()
            self.root.after(50, lambda: self._show_polar_motors_popup(viewing_mode="day"))

        def on_no():
            popup.destroy()
            self.root.after(50, lambda: self._show_solar_filter_popup(first_time=False))

        ctk.CTkButton(
            btn_frame,
            text="Yes",
            command=on_yes,
            font=("Segoe UI", 12, "bold"),
            width=100,
            height=38,
            fg_color="#2e7d32",
            hover_color="#43a047",
            corner_radius=8,
        ).pack(side=tk.LEFT, padx=10)

        ctk.CTkButton(
            btn_frame,
            text="No",
            command=on_no,
            font=("Segoe UI", 12, "bold"),
            width=100,
            height=38,
            fg_color="#c62828",
            hover_color="#e53935",
            corner_radius=8,
        ).pack(side=tk.LEFT, padx=10)

    def _apply_mode_settings(self, night_mode, image_format):
        """Apply theme and image format based on the selected viewing mode."""
        # Apply theme if needed
        if night_mode and not self.night_mode:
            self.night_mode = True
            self.theme = THEMES["night"]
            self.theme_btn.configure(text="🌙 Night Mode")
            self.apply_theme()
        elif not night_mode and self.night_mode:
            self.night_mode = False
            self.theme = THEMES["day"]
            self.theme_btn.configure(text="☀ Day Mode")
            self.apply_theme()

        # Apply image format
        self.format_var.set(image_format)
        self.update_image_format()

    def _show_polar_motors_popup(self, first_time=True, viewing_mode="day"):
        """Check that polar alignment motors are off before daytime solar viewing."""
        popup = ctk.CTkToplevel(self.root)
        popup.title("Equipment Check")
        popup.resizable(False, False)
        popup.grab_set()

        # Center over the main window
        self.root.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 185
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 110
        popup.geometry(f"370x220+{x}+{y}")

        ctk.CTkLabel(
            popup,
            text="⚙️",
            font=("Segoe UI", 36),
        ).pack(pady=(22, 4))

        ctk.CTkLabel(
            popup,
            text="Are the polar alignment motors turned off?",
            font=("Segoe UI", 14, "bold"),
            wraplength=320,
            justify="center",
        ).pack()

        if not first_time:
            ctk.CTkLabel(
                popup,
                text="Please turn off the polar alignment motors\nbefore proceeding.",
                font=("Segoe UI", 10),
                text_color="#ff4444",
                wraplength=320,
                justify="center",
            ).pack(pady=(6, 0))

        btn_frame = ctk.CTkFrame(popup, fg_color="transparent")
        btn_frame.pack(pady=(16, 20))

        def on_yes():
            popup.destroy()
            if viewing_mode == "day":
                self._apply_mode_settings(night_mode=False, image_format="RAW8")
            elif viewing_mode == "night":
                self._apply_mode_settings(night_mode=True, image_format="RAW8")

        def on_no():
            popup.destroy()
            self.root.after(50, lambda: self._show_polar_motors_popup(first_time=False, viewing_mode=viewing_mode))

        ctk.CTkButton(
            btn_frame,
            text="Yes",
            command=on_yes,
            font=("Segoe UI", 12, "bold"),
            width=100,
            height=38,
            fg_color="#2e7d32",
            hover_color="#43a047",
            corner_radius=8,
        ).pack(side=tk.LEFT, padx=10)

        ctk.CTkButton(
            btn_frame,
            text="No",
            command=on_no,
            font=("Segoe UI", 12, "bold"),
            width=100,
            height=38,
            fg_color="#c62828",
            hover_color="#e53935",
            corner_radius=8,
        ).pack(side=tk.LEFT, padx=10)

    def _show_polar_motors_on_popup(self, first_time=True):
        """Check that polar alignment motors are ON for deep sky viewing."""
        popup = ctk.CTkToplevel(self.root)
        popup.title("Equipment Check")
        popup.resizable(False, False)
        popup.grab_set()

        # Center over the main window
        self.root.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 185
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 110
        popup.geometry(f"370x220+{x}+{y}")

        ctk.CTkLabel(
            popup,
            text="⚙️",
            font=("Segoe UI", 36),
        ).pack(pady=(22, 4))

        ctk.CTkLabel(
            popup,
            text="Are the polar alignment motors turned on?",
            font=("Segoe UI", 14, "bold"),
            wraplength=320,
            justify="center",
        ).pack()

        if not first_time:
            ctk.CTkLabel(
                popup,
                text="Please turn on the polar alignment motors\nbefore proceeding.",
                font=("Segoe UI", 10),
                text_color="#ff4444",
                wraplength=320,
                justify="center",
            ).pack(pady=(6, 0))

        btn_frame = ctk.CTkFrame(popup, fg_color="transparent")
        btn_frame.pack(pady=(16, 20))

        def on_yes():
            popup.destroy()
            self._apply_mode_settings(night_mode=True, image_format="RAW16")

        def on_no():
            popup.destroy()
            self.root.after(50, lambda: self._show_polar_motors_on_popup(first_time=False))

        ctk.CTkButton(
            btn_frame,
            text="Yes",
            command=on_yes,
            font=("Segoe UI", 12, "bold"),
            width=100,
            height=38,
            fg_color="#2e7d32",
            hover_color="#43a047",
            corner_radius=8,
        ).pack(side=tk.LEFT, padx=10)

        ctk.CTkButton(
            btn_frame,
            text="No",
            command=on_no,
            font=("Segoe UI", 12, "bold"),
            width=100,
            height=38,
            fg_color="#c62828",
            hover_color="#e53935",
            corner_radius=8,
        ).pack(side=tk.LEFT, padx=10)

    def _get_exposure_seconds(self):
        """Return current exposure as seconds (float), used for timeout and FPS calculations."""
        range_text = self.exposure_range_var.get()
        exp_val = self.exposure_var.get()
        if range_text == "32-1000us":
            return exp_val / 1_000_000
        elif range_text in ("1-10s", "10-60s"):
            return exp_val
        else:
            return exp_val / 1000.0

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')

        # TOP BAR
        self.top_bar = tk.Frame(self.root)
        self.top_bar.pack(fill=tk.X, padx=12, pady=8)

        self.title_label = tk.Label(
            self.top_bar, text="CAMERA CONTROL", font=("Segoe UI", 16, "bold")
        )
        self.title_label.pack(side=tk.LEFT)

        spacer = tk.Frame(self.top_bar)
        spacer.pack(side=tk.LEFT, expand=True)

        # Temperature display
        self.temp_label = tk.Label(
            self.top_bar,
            text="Temp: --°C",
            font=("Segoe UI", 10),
            padx=12
        )
        self.temp_label.pack(side=tk.RIGHT, padx=(0, 12))

        self.theme_btn = ctk.CTkButton(
            self.top_bar,
            text="☀ Day Mode",
            command=self.toggle_theme,
            font=("Segoe UI", 9),
            width=120,
            height=32,
            fg_color="#e0e0e0",
            text_color="#1a1a1a",
            hover_color="#0b5c8f",
            corner_radius=6
        )
        self.theme_btn.pack(side=tk.RIGHT)

        self.plate_solve_btn = ctk.CTkButton(
            self.top_bar,
            text="Plate Solver",
            command=self.open_plate_solver,
            font=("Segoe UI", 9, "bold"),
            width=130,
            height=32,
            fg_color="#0b5c8f",
            text_color="white",
            hover_color="#1a7ab5",
            corner_radius=6
        )
        self.plate_solve_btn.pack(side=tk.RIGHT, padx=(0, 8))

        self.skytrack_top_btn = ctk.CTkButton(
            self.top_bar,
            text="Sky Tracker",
            command=self.open_skytrack,
            font=("Segoe UI", 9, "bold"),
            width=130,
            height=32,
            fg_color="#0b5c8f",
            text_color="white",
            hover_color="#1a7ab5",
            corner_radius=6
        )
        self.skytrack_top_btn.pack(side=tk.RIGHT, padx=(0, 8))

        # MAIN CONTAINER
        self.main = tk.Frame(self.root)
        self.main.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        # LEFT PANEL - Camera Controls (scrollable)
        self.left_scroll = ctk.CTkScrollableFrame(
            self.main,
            width=290,
            fg_color="transparent",
            scrollbar_button_color="#999999",
            scrollbar_button_hover_color="#777777",
        )
        self.left_scroll.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        self.left = self.left_scroll  # alias so all existing widget code works unchanged

        tk.Label(
            self.left, text="CAMERA CONTROLS", font=("Segoe UI", 11, "bold")
        ).pack(anchor="w", pady=(0, 8))

        # Camera selection
        tk.Label(
            self.left, text="Camera:", font=("Segoe UI", 9)
        ).pack(anchor="w")

        camera_frame = tk.Frame(self.left)
        camera_frame.pack(fill=tk.X, pady=4)

        self.camera_select_var = tk.StringVar()
        camera_options = [f"{idx}: {name}" for idx, name in self.available_cameras]
        if not camera_options:
            camera_options = ["No cameras detected"]

        self.camera_dropdown = ctk.CTkComboBox(
            camera_frame,
            variable=self.camera_select_var,
            values=camera_options,
            state="readonly",
            width=180,
            fg_color="white",
            button_color="#0b5c8f",
            button_hover_color="#1a7ab5",
            border_color="#cccccc",
            dropdown_fg_color="white",
            dropdown_hover_color="#f0f0f0",
            corner_radius=6
        )
        self.camera_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True)
        if camera_options[0] != "No cameras detected":
            self.camera_dropdown.set(camera_options[0])

        self.connect_btn = ctk.CTkButton(
            camera_frame,
            text="Connect",
            command=self.connect_camera,
            font=("Segoe UI", 8),
            width=70,
            height=28,
            fg_color="#ebebeb",
            text_color="#1a1a1a",
            hover_color="#e0e0e0",
            corner_radius=6
        )
        self.connect_btn.pack(side=tk.LEFT, padx=(4, 0))

        self.refresh_btn = ctk.CTkButton(
            camera_frame,
            text="⟳",
            command=self.refresh_cameras,
            font=("Segoe UI", 10),
            width=35,
            height=28,
            fg_color="#ebebeb",
            text_color="#1a1a1a",
            hover_color="#e0e0e0",
            corner_radius=6
        )
        self.refresh_btn.pack(side=tk.LEFT, padx=(2, 0))

        # Helper function to create dropdown controls
        def create_dropdown_control(label_text, var, options, command=None):
            tk.Label(self.left, text=label_text, font=("Segoe UI", 9)).pack(anchor="w", pady=(8, 0))
            dropdown = ctk.CTkComboBox(
                self.left,
                variable=var if var else None,
                values=options,
                state="readonly",
                width=240,
                fg_color="white",
                button_color="#0b5c8f",
                button_hover_color="#1a7ab5",
                border_color="#cccccc",
                dropdown_fg_color="white",
                dropdown_hover_color="#f0f0f0",
                corner_radius=6,
                command=command
            )
            dropdown.pack(fill=tk.X, pady=4)
            return dropdown

        # BINNING
        self.binning_dropdown = create_dropdown_control(
            "Binning:",
            None,
            ["1x1", "2x2", "3x3", "4x4"],
            self.update_binning
        )
        self.binning_dropdown.set("1x1")

        # IMAGE FORMAT
        self.format_var = tk.StringVar(value="RAW16")
        self.format_dropdown = create_dropdown_control(
            "Image Format:",
            self.format_var,
            ["RAW8", "RAW16"],
            self.update_image_format
        )

        # CAPTURE COUNT
        self.capture_count_dropdown = create_dropdown_control(
            "Capture Count:",
            None,
            ["1", "5", "10", "25", "50"]
        )
        self.capture_count_dropdown.set("1")

        # EXPOSURE RANGE — includes new 32-1000µs option
        self.exposure_range_var = tk.StringVar(value="1-100ms")
        self.exposure_range_dropdown = create_dropdown_control(
            "Exposure Range:",
            self.exposure_range_var,
            ["32-1000us", "1-100ms", "100ms-1000ms", "1-10s", "10-60s"],
            self.update_exposure_range
        )

        # Helper function to create slider controls
        def create_slider_control(label_text, var, from_, to, command, display_width=10):
            tk.Label(self.left, text=label_text, font=("Segoe UI", 9)).pack(anchor="w", pady=(8, 0))
            frame = tk.Frame(self.left)
            frame.pack(fill=tk.X, pady=4)

            slider = ctk.CTkSlider(
                frame,
                from_=from_,
                to=to,
                variable=var,
                command=command,
                button_color="#0b5c8f",
                button_hover_color="#1a7ab5",
                progress_color="#0b5c8f",
                fg_color="#e0e0e0",
                width=160,
                height=16
            )
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

            label = tk.Label(frame, text="", width=display_width, font=("Segoe UI", 9))
            label.pack(side=tk.LEFT, padx=(4, 0))

            return slider, label

        # EXPOSURE
        self.exposure_var = tk.DoubleVar(value=10)
        self.exposure_slider, self.exposure_label = create_slider_control(
            "Exposure:",
            self.exposure_var,
            1,
            100,
            self.update_exposure
        )
        self.exposure_label.configure(text="10.0 ms")

        # GAIN
        self.gain_var = tk.IntVar(value=0)
        self.gain_slider, self.gain_label = create_slider_control(
            "Gain:",
            self.gain_var,
            0,
            500,
            self.update_gain
        )
        self.gain_label.configure(text="0")

        # FILE PATH
        tk.Label(
            self.left, text="Save Path:", font=("Segoe UI", 9)
        ).pack(anchor="w", pady=(8, 0))

        path_frame = tk.Frame(self.left)
        path_frame.pack(fill=tk.X, pady=4)

        self.path_entry = ctk.CTkEntry(
            path_frame,
            font=("Segoe UI", 8),
            height=28,
            fg_color="white",
            text_color="#1a1a1a",
            border_color="#cccccc",
            border_width=1,
            corner_radius=6
        )
        self.path_entry.insert(0, self.save_path)
        self.path_entry.configure(state="readonly")
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.browse_btn = ctk.CTkButton(
            path_frame,
            text="Browse...",
            command=self.select_save_path,
            font=("Segoe UI", 8),
            width=80,
            height=28,
            fg_color="#ebebeb",
            text_color="#1a1a1a",
            hover_color="#e0e0e0",
            corner_radius=6
        )
        self.browse_btn.pack(side=tk.LEFT, padx=(4, 0))

        # PROJECT NAME
        tk.Label(
            self.left, text="Project Name:", font=("Segoe UI", 9)
        ).pack(anchor="w", pady=(8, 0))

        self.project_name_entry = ctk.CTkEntry(
            self.left,
            font=("Segoe UI", 8),
            height=28,
            fg_color="white",
            text_color="#1a1a1a",
            placeholder_text_color="#1a1a1a",
            border_color="#cccccc",
            border_width=1,
            corner_radius=6,
            placeholder_text="e.g. Jupiter_2026"
        )
        self.project_name_entry.pack(fill=tk.X, pady=4)

        # FRAME TYPE
        tk.Label(
            self.left, text="Frame Type:", font=("Segoe UI", 9)
        ).pack(anchor="w", pady=(8, 0))

        self.frame_type_var = tk.StringVar(value="Light")
        self.frame_type_dropdown = ctk.CTkComboBox(
            self.left,
            variable=self.frame_type_var,
            values=["Light", "Dark", "Flat", "Bias"],
            font=("Segoe UI", 9),
            height=28,
            fg_color="white",
            text_color="#1a1a1a",
            border_color="#cccccc",
            border_width=1,
            corner_radius=6,
            button_color="#0b5c8f",
            button_hover_color="#1a7ab5",
            state="readonly",
            dropdown_fg_color="white",
            dropdown_hover_color="#f0f0f0",
            dropdown_text_color="#1a1a1a",
        )
        self.frame_type_dropdown.pack(fill=tk.X, pady=4)

        # CAPTURE BUTTONS
        ttk.Separator(self.left).pack(fill=tk.X, pady=10)

        tk.Label(
            self.left, text="CAPTURE", font=("Segoe UI", 11, "bold")
        ).pack(anchor="w", pady=(0, 8))

        # Helper function to create action buttons
        def create_action_button(text, command):
            return ctk.CTkButton(
                self.left,
                text=text,
                command=command,
                font=("Segoe UI", 9, "bold"),
                height=36,
                fg_color="#0b5c8f",
                text_color="white",
                hover_color="#1a7ab5",
                corner_radius=6
            )

        self.start_preview_btn = create_action_button("Start Preview", self.toggle_preview)
        self.start_preview_btn.pack(fill=tk.X, pady=4)

        self.capture_img_btn = create_action_button("Capture Image", self.capture_image)
        self.capture_img_btn.pack(fill=tk.X, pady=4)

        self.record_video_btn = create_action_button("Start Recording", self.toggle_recording)
        self.record_video_btn.pack(fill=tk.X, pady=4)

        # SKY TRACKER SECTION
        ttk.Separator(self.left).pack(fill=tk.X, pady=10)

        tk.Label(
            self.left, text="SKY TRACKER", font=("Segoe UI", 11, "bold")
        ).pack(anchor="w", pady=(0, 8))

        # Sky tracker status info
        skytrack_info_frame = tk.Frame(self.left)
        skytrack_info_frame.pack(fill=tk.X, pady=(6, 2))

        self.skytrack_signal_dot = tk.Label(
            skytrack_info_frame,
            text="●",
            font=("Segoe UI", 10),
            fg="#999999"
        )
        self.skytrack_signal_dot.pack(side=tk.LEFT, padx=(2, 4))

        self.skytrack_signal_label = tk.Label(
            skytrack_info_frame,
            text="Not tracking",
            font=("Segoe UI", 9, "italic"),
            fg="#999999"
        )
        self.skytrack_signal_label.pack(side=tk.LEFT)

        self.skytrack_target_label = tk.Label(
            self.left,
            text="Target:",
            font=("Segoe UI", 9),
            anchor="w"
        )
        self.skytrack_target_label.pack(fill=tk.X, padx=4)

        self.skytrack_ra_label = tk.Label(
            self.left,
            text="RA :",
            font=("Segoe UI", 9, "bold"),
            anchor="w"
        )
        self.skytrack_ra_label.pack(fill=tk.X, padx=4)

        self.skytrack_dec_label = tk.Label(
            self.left,
            text="DEC :",
            font=("Segoe UI", 9, "bold"),
            anchor="w"
        )
        self.skytrack_dec_label.pack(fill=tk.X, padx=4)

        ttk.Separator(self.left).pack(fill=tk.X, pady=10)

        tk.Label(
            self.left, text="STATUS", font=("Segoe UI", 11, "bold")
        ).pack(anchor="w", pady=(0, 4))

        self.status_label = tk.Label(
            self.left,
            text="Ready - No camera connected",
            font=("Segoe UI", 9, "italic"),
            wraplength=240,
            justify=tk.LEFT,
            anchor=tk.W
        )
        self.status_label.pack(fill=tk.X)

        # RIGHT PANEL - Preview
        self.right = tk.Frame(self.main)
        self.right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(
            self.right, text="CAMERA PREVIEW", font=("Segoe UI", 11, "bold")
        ).pack(anchor="w", pady=(0, 8))

        self.preview_label = tk.Label(
            self.right,
            text="No preview available\nConnect camera and start preview",
            font=("Segoe UI", 12),
            justify=tk.CENTER
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True)

    def apply_theme(self):
        """Apply the current theme to all UI elements"""
        t = self.theme

        # Root and main containers
        self.root.configure(bg=t["bg_primary"])

        for frame in [self.top_bar, self.main, self.right]:
            try:
                frame.configure(bg=t["bg_primary"])
            except:
                pass

        # Style the scrollable panel
        try:
            self.left_scroll.configure(
                fg_color=t["bg_primary"],
                scrollbar_button_color=t["bg_tertiary"],
                scrollbar_button_hover_color=t["bg_secondary"],
            )
        except:
            pass

        # Labels
        label_configs = [
            (self.title_label, t["bg_primary"], t["fg_primary"]),
            (self.temp_label, t["bg_primary"], t["fg_primary"]),
            (self.status_label, t["bg_primary"], t["fg_primary"]),
            (self.exposure_label, t["bg_primary"], t["fg_primary"]),
            (self.gain_label, t["bg_primary"], t["fg_primary"]),
            (self.preview_label, t["canvas_bg"], t["fg_tertiary"])
        ]

        for label, bg, fg in label_configs:
            label.configure(bg=bg, fg=fg)

        # Update all other labels in containers
        configured_labels = [lbl for lbl, _, _ in label_configs]
        for container in [self.left, self.right, self.top_bar]:
            try:
                children = container.winfo_children()
            except:
                children = []
            for widget in children:
                if isinstance(widget, tk.Label) and widget not in configured_labels:
                    try:
                        widget.configure(bg=t["bg_primary"], fg=t["fg_primary"])
                    except:
                        pass
                try:
                    for sub in widget.winfo_children():
                        if isinstance(sub, tk.Label) and sub not in configured_labels:
                            try:
                                sub.configure(bg=t["bg_primary"], fg=t["fg_primary"])
                            except:
                                pass
                except:
                    pass

        # Update all tk.Frame widgets inside left panel
        try:
            for widget in self.left.winfo_children():
                if isinstance(widget, tk.Frame):
                    try:
                        widget.configure(bg=t["bg_primary"])
                    except:
                        pass
        except:
            pass

        # Update CTkButtons
        action_buttons = [self.start_preview_btn, self.capture_img_btn, self.record_video_btn]
        small_buttons = [self.connect_btn, self.refresh_btn, self.browse_btn]

        btn_text_color = "#fccfcf" if self.night_mode else "white"
        for btn in action_buttons:
            btn.configure(fg_color=t["accent"], hover_color=t["accent_light"], text_color=btn_text_color)

        for btn in small_buttons:
            btn.configure(
                fg_color=t["bg_secondary"],
                text_color=t["fg_primary"],
                hover_color=t["bg_tertiary"],
                border_color=t["border"],
                border_width=1
            )

        # Path entry
        self.path_entry.configure(
            fg_color=t["bg_secondary"] if self.night_mode else "white",
            text_color=t["fg_primary"],
            border_color=t["border"]
        )

        # Project name entry
        self.project_name_entry.configure(
            fg_color=t["bg_secondary"] if self.night_mode else "white",
            text_color=t["fg_primary"],
            placeholder_text_color=t["fg_primary"],
            border_color=t["border"]
        )

        # Plate Solver and Sky Tracker top bar buttons — same style as action buttons
        for btn in [self.plate_solve_btn, self.skytrack_top_btn]:
            btn.configure(fg_color=t["accent"], hover_color=t["accent_light"], text_color=btn_text_color)

        process = getattr(self, "_skytrack_process", None)
        is_tracking = process is not None and process.poll() is None
        signal_color = t["accent"] if is_tracking else t["fg_tertiary"]
        signal_text  = "Tracking" if is_tracking else "Not tracking"
        self.skytrack_signal_dot.configure(bg=t["bg_primary"], fg=signal_color)
        self.skytrack_signal_label.configure(bg=t["bg_primary"], fg=signal_color, text=signal_text)
        for lbl in [self.skytrack_target_label, self.skytrack_ra_label, self.skytrack_dec_label]:
            lbl.configure(bg=t["bg_primary"], fg=t["fg_secondary"])

        if self.night_mode:
            self.theme_btn.configure(fg_color=t["accent"], text_color="white", hover_color=t["accent_light"])
        else:
            self.theme_btn.configure(fg_color=t["bg_tertiary"], text_color=t["fg_primary"], hover_color=t["accent"])

        combo_settings = {
            "fg_color": t["bg_secondary"] if self.night_mode else "white",
            "text_color": t["fg_primary"] if self.night_mode else "#1a1a1a",
            "border_color": t["border"] if self.night_mode else "#cccccc",
            "button_color": t["accent"],
            "button_hover_color": t["accent_light"],
            "dropdown_fg_color": t["bg_tertiary"] if self.night_mode else "white",
            "dropdown_hover_color": t["bg_secondary"] if self.night_mode else "#f0f0f0"
        }

        for combo in [self.camera_dropdown, self.binning_dropdown, self.format_dropdown,
                      self.capture_count_dropdown, self.exposure_range_dropdown, self.frame_type_dropdown]:
            combo.configure(**combo_settings)

        slider_settings = {
            "button_color": t["accent"],
            "button_hover_color": t["accent_light"],
            "progress_color": t["accent"],
            "fg_color": t["bg_tertiary"]
        }

        self.exposure_slider.configure(**slider_settings)
        self.gain_slider.configure(**slider_settings)

    def toggle_theme(self):
        self.night_mode = not self.night_mode
        self.theme = THEMES["night"] if self.night_mode else THEMES["day"]
        self.theme_btn.configure(text="🌙 Night Mode" if self.night_mode else "☀ Day Mode")
        self.apply_theme()

    def setup_camera(self):
        if not self.camera or not self.camera_initialized:
            self.update_status("No camera connected")
            return

        try:
            controls = self.camera.get_controls()
            camera_property = self.camera.get_camera_property()
            self.full_sensor_width = camera_property['MaxWidth']
            self.full_sensor_height = camera_property['MaxHeight']

            print(f"Full sensor dimensions: {self.full_sensor_width}x{self.full_sensor_height}")

            self.camera.set_control_value(asi.ASI_GAIN, self.gain_var.get())
            self.camera.set_control_value(asi.ASI_EXPOSURE, int(self.exposure_var.get() * 1000))
            self.camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, controls['BandWidth']['DefaultValue'])

            format_map = {"RAW8": asi.ASI_IMG_RAW8, "RAW16": asi.ASI_IMG_RAW16}
            selected_format = format_map.get(self.format_var.get(), asi.ASI_IMG_RAW16)
            self.camera.set_image_type(selected_format)

            self.update_status("Camera initialized successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to setup camera: {e}")

    def refresh_cameras(self):
        try:
            if self.is_capturing:
                self.stop_preview()

            if self.camera:
                try:
                    self.camera.close()
                except:
                    pass
                self.camera = None
                self.camera_initialized = False

            self.available_cameras = []
            num_cameras = asi.get_num_cameras()

            if num_cameras > 0:
                for i in range(num_cameras):
                    camera_info = asi.Camera(i).get_camera_property()
                    self.available_cameras.append((i, camera_info['Name']))

                camera_options = [f"{idx}: {name}" for idx, name in self.available_cameras]
                self.camera_dropdown.configure(values=camera_options)
                self.camera_dropdown.set(camera_options[0])
                self.update_status(f"Found {num_cameras} camera(s)")
            else:
                self.camera_dropdown.configure(values=["No cameras detected"])
                self.camera_dropdown.set("No cameras detected")
                self.update_status("No cameras found")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh cameras: {e}")
            self.update_status("Error refreshing cameras")

    def connect_camera(self):
        try:
            if not self.available_cameras:
                messagebox.showwarning("Warning", "No cameras available to connect")
                return

            if self.is_capturing:
                self.stop_preview()

            if self.camera:
                try:
                    self.camera.close()
                except:
                    pass

            selection = self.camera_select_var.get()
            if selection == "No cameras detected":
                return

            camera_idx = int(selection.split(":")[0])

            self.camera = asi.Camera(camera_idx)
            self.camera_info = self.camera.get_camera_property()
            self.camera_initialized = True

            self.setup_camera()

            self.update_status(f"Connected to {self.camera_info['Name']}")
            messagebox.showinfo("Success", f"Connected to {self.camera_info['Name']}")

        except Exception as e:
            self.camera = None
            self.camera_initialized = False
            messagebox.showerror("Error", f"Failed to connect to camera: {e}")
            self.update_status("Failed to connect to camera")

    def update_binning(self, choice=None):
        if not self.camera or not self.camera_initialized:
            return

        try:
            bin_text = self.binning_dropdown.get()
            bin_value = int(bin_text.split('x')[0])

            was_capturing = self.is_capturing
            if was_capturing:
                self.camera.stop_video_capture()

            binned_width = self.full_sensor_width // bin_value
            binned_height = self.full_sensor_height // bin_value

            self.camera.set_roi(
                start_x=0,
                start_y=0,
                width=binned_width,
                height=binned_height,
                bins=bin_value
            )

            print(f"Set ROI: {binned_width}x{binned_height} with {bin_value}x{bin_value} binning")

            if was_capturing:
                self.camera.start_video_capture()

            self.update_status(f"Binning set to {bin_value}x{bin_value}")
        except Exception as e:
            self.update_status(f"Error setting binning: {e}")
            import traceback
            traceback.print_exc()

    def update_exposure_range(self, choice=None):
        """Update exposure slider range based on selected range."""
        range_text = self.exposure_range_var.get()

        range_map = {
            "32-1000us":     (32,   1000,  "us"),
            "1-100ms":       (1,    100,   "ms"),
            "100ms-1000ms":  (100,  1000,  "ms"),
            "1-10s":         (1,    10,    "s"),
            "10-60s":        (10,   60,    "s"),
        }

        if range_text not in range_map:
            return

        min_val, max_val, unit = range_map[range_text]

        self.exposure_slider.configure(from_=min_val, to=max_val)

        current = self.exposure_var.get()
        if current < min_val or current > max_val:
            self.exposure_var.set(min_val)
            self.exposure_slider.set(min_val)
            self.update_exposure(min_val)

    def update_exposure(self, value):
        """Update camera exposure."""
        val = float(value)
        range_text = self.exposure_range_var.get()

        if range_text == "32-1000us":
            self.exposure_label.configure(text=f"{val:.0f} µs")
            exposure_us = int(val)
        elif range_text in ("1-10s", "10-60s"):
            self.exposure_label.configure(text=f"{val:.1f} s")
            exposure_us = int(val * 1_000_000)
        else:
            self.exposure_label.configure(text=f"{val:.1f} ms")
            exposure_us = int(val * 1000)

        if not self.camera or not self.camera_initialized:
            return

        try:
            self.camera.set_control_value(asi.ASI_EXPOSURE, exposure_us)
        except Exception as e:
            pass

    def update_gain(self, value):
        gain = int(float(value))
        self.gain_label.configure(text=str(gain))

        if not self.camera or not self.camera_initialized:
            return

        try:
            self.camera.set_control_value(asi.ASI_GAIN, gain)
        except Exception as e:
            self.update_status(f"Error setting gain: {e}")

    def update_image_format(self, choice=None):
        if not self.camera or not self.camera_initialized:
            return

        try:
            was_capturing = self.is_capturing
            if was_capturing:
                self.stop_preview()

            format_map = {"RAW8": asi.ASI_IMG_RAW8, "RAW16": asi.ASI_IMG_RAW16}
            selected_format = format_map.get(self.format_var.get(), asi.ASI_IMG_RAW16)
            self.camera.set_image_type(selected_format)

            self.update_status(f"Image format set to {self.format_var.get()}")

            if was_capturing:
                self.start_preview()

        except Exception as e:
            self.update_status(f"Error setting format: {e}")

    def select_save_path(self):
        path = filedialog.askdirectory(initialdir=self.save_path)
        if path:
            self.save_path = path
            self.path_entry.configure(state="normal")
            self.path_entry.delete(0, "end")
            self.path_entry.insert(0, path)
            self.path_entry.configure(state="readonly")

    def toggle_preview(self):
        if not self.is_capturing:
            self.start_preview()
            self.start_preview_btn.configure(text="Stop Preview")
        else:
            self.stop_preview()
            self.start_preview_btn.configure(text="Start Preview")

    def start_preview(self):
        if not self.camera or not self.camera_initialized:
            messagebox.showwarning("Warning", "Please connect to a camera first")
            return

        try:
            self.camera.start_video_capture()
            self.is_capturing = True
            self.update_status("Preview started")
            threading.Thread(target=self.preview_loop, daemon=True).start()
            threading.Thread(target=self.temperature_loop, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start preview: {e}")

    def stop_preview(self):
        try:
            self.is_capturing = False
            if self.is_recording:
                self.toggle_recording()
            if self.camera and self.camera_initialized:
                self.camera.stop_video_capture()
            self.update_status("Preview stopped")
        except Exception as e:
            self.update_status(f"Error stopping preview: {e}")

    def temperature_loop(self):
        while self.camera and self.camera_initialized:
            try:
                controls = self.camera.get_controls()
                if 'Temperature' in controls:
                    temp = self.camera.get_control_value(asi.ASI_TEMPERATURE)[0] / 10.0
                    self.temp_label.configure(text=f"Temp: {temp:.1f}°C")
            except Exception as e:
                self.temp_label.configure(text="Temp: --°C")

            time.sleep(2)

    def preview_loop(self):
        while self.is_capturing:
            try:
                # FIX: Compute a timeout that accounts for the actual exposure duration.
                # The ZWO API default timeout is too short for exposures longer than ~1s,
                # causing capture_video_frame() to raise an error before the frame arrives.
                # Formula: exposure_ms * 2 + 500ms gives a safe margin in all ranges.
                exposure_s = self._get_exposure_seconds()
                exposure_ms = exposure_s * 1000
                timeout_ms = int(exposure_ms * 2 + 500)

                frame = self.camera.capture_video_frame(timeout=timeout_ms)

                if frame is not None:
                    current_format = self.format_var.get()

                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    elif len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
                        frame_2d = frame if len(frame.shape) == 2 else frame.reshape(frame.shape[0], frame.shape[1])

                        if current_format == "RAW16":
                            frame_8bit = (frame_2d / 256).astype(np.uint8)
                            img = cv2.cvtColor(frame_8bit, cv2.COLOR_BayerRG2RGB)
                        elif current_format == "RAW8":
                            img = cv2.cvtColor(frame_2d, cv2.COLOR_BayerRG2RGB)
                        else:
                            img = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                    else:
                        img = frame

                    if self.is_recording and self.video_writer:
                        self.video_writer.write(img)

                    height, width = img.shape[:2]
                    scale = min(640 / width, 480 / height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)

                    img_resized = cv2.resize(img, (new_width, new_height))
                    img_pil = Image.fromarray(img_resized)
                    img_tk = ImageTk.PhotoImage(image=img_pil)

                    self.preview_label.configure(image=img_tk, text="")
                    self.preview_label.image = img_tk

            except Exception as e:
                print(f"Preview error: {e}")
                import traceback
                traceback.print_exc()
                continue

    def capture_image(self):
        if not self.camera or not self.camera_initialized:
            messagebox.showwarning("Warning", "Please connect to a camera first")
            return

        was_previewing = self.is_capturing
        current_format = self.format_var.get()
        bin_value = int(self.binning_dropdown.get().split('x')[0])
        capture_count = int(self.capture_count_dropdown.get())

        session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        capture_folder = f"capture_{session_timestamp}"
        project_name = self.project_name_entry.get().strip()
        frame_type = self.frame_type_var.get()
        if project_name:
            base_path = os.path.join(self.save_path, project_name, frame_type)
        else:
            base_path = os.path.join(self.save_path, frame_type)
        capture_path = os.path.join(base_path, capture_folder)

        try:
            os.makedirs(capture_path, exist_ok=True)
            print(f"Created capture folder: {capture_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create capture folder: {e}")
            return

        try:
            if was_previewing:
                print("Stopping preview for still capture...")
                self.is_capturing = False
                time.sleep(0.2)
                try:
                    self.camera.stop_video_capture()
                except:
                    pass
                self.update_status("Preview stopped for capture")

            binned_width = self.full_sensor_width // bin_value
            binned_height = self.full_sensor_height // bin_value

            print(f"Setting ROI: {binned_width}x{binned_height} with {bin_value}x{bin_value} binning")
            self.camera.set_roi(
                start_x=0,
                start_y=0,
                width=binned_width,
                height=binned_height,
                bins=bin_value
            )

            format_map = {"RAW8": asi.ASI_IMG_RAW8, "RAW16": asi.ASI_IMG_RAW16}
            self.camera.set_image_type(format_map.get(current_format, asi.ASI_IMG_RAW16))

            # Re-apply exposure using the same unit-aware logic as update_exposure
            range_text = self.exposure_range_var.get()
            if range_text == "32-1000us":
                exposure_us = int(self.exposure_var.get())
                exposure_display = f"{self.exposure_var.get():.0f} µs"
                exposure_s = self.exposure_var.get() / 1_000_000
            elif range_text in ("1-10s", "10-60s"):
                exposure_us = int(self.exposure_var.get() * 1_000_000)
                exposure_display = f"{self.exposure_var.get():.1f} s"
                exposure_s = self.exposure_var.get()
            else:
                exposure_us = int(self.exposure_var.get() * 1000)
                exposure_display = f"{self.exposure_var.get():.1f} ms"
                exposure_s = self.exposure_var.get() / 1000.0
            self.camera.set_control_value(asi.ASI_EXPOSURE, exposure_us)
            self.camera.set_control_value(asi.ASI_GAIN, self.gain_var.get())

            all_saved_files = []

            for img_num in range(1, capture_count + 1):
                print(f"\n{'=' * 60}")
                print(f"CAPTURING IMAGE {img_num}/{capture_count}")
                print(f"{'=' * 60}")

                self.update_status(f"Capturing image {img_num}/{capture_count}...")
                self.root.update()

                frame = self.camera.capture()

                if frame is None:
                    raise Exception(f"Failed to capture frame {img_num}")

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

                try:
                    temp = self.camera.get_control_value(asi.ASI_TEMPERATURE)[0] / 10.0
                except:
                    temp = None

                metadata = {
                    'image_number': img_num,
                    'total_images': capture_count,
                    'timestamp': timestamp,
                    'datetime': datetime.now().isoformat(),
                    'camera': self.camera_info['Name'],
                    'resolution': f"{frame.shape[1]}x{frame.shape[0]}",
                    'bins': bin_value,
                    'format': current_format,
                    'bit_depth': 16 if current_format == "RAW16" else 8,
                    'dtype': str(frame.dtype),
                    'buffer_size': frame.nbytes,
                    'shape': frame.shape,
                    'exposure_display': exposure_display,
                    'exposure_s': exposure_s,
                    'gain': self.gain_var.get(),
                    'temperature_c': temp
                }

                print(f"Capture metadata for image {img_num}:")
                for key, value in metadata.items():
                    print(f"  {key}: {value}")

                fits_filename = f"image_{img_num:03d}_{timestamp}_{current_format}.fits"
                fits_filepath = os.path.join(capture_path, fits_filename)

                self.update_status(f"Saving FITS {img_num}/{capture_count}...")
                self.root.update()

                header = fits.Header()
                header['INSTRUME'] = self.camera_info['Name']
                header['EXPOSURE'] = (exposure_s, 'Exposure time in seconds')
                header['GAIN'] = (self.gain_var.get(), 'Gain value')
                header['BINNING'] = (bin_value, 'Binning factor')
                header['XBINNING'] = (bin_value, 'X-axis binning')
                header['YBINNING'] = (bin_value, 'Y-axis binning')
                header['XPIXSZ'] = (2.9, 'Pixel size in microns (unbinned)')
                header['YPIXSZ'] = (2.9, 'Pixel size in microns (unbinned)')
                header['IMAGETYP'] = ('Light Frame', 'Type of image')
                header['BAYERPAT'] = ('RGGB', 'Bayer pattern')
                header['COLORTYP'] = ('RGGB', 'Color type')
                header['DATE-OBS'] = (datetime.now().isoformat(), 'Date of observation')
                header['IMGNUM'] = (img_num, f'Image number in sequence of {capture_count}')
                header['BITDEPTH'] = (metadata['bit_depth'], 'Bit depth')
                if temp is not None:
                    header['CCD-TEMP'] = (temp, 'CCD temperature in Celsius')

                hdu = fits.PrimaryHDU(data=frame, header=header)
                hdu.writeto(fits_filepath, overwrite=True)

                print(f"Saved FITS file: {fits_filepath}")
                all_saved_files.append(fits_filename)

                if current_format == "RAW8":
                    self.update_status(f"Creating PNG {img_num}/{capture_count}...")
                    self.root.update()

                    if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
                        debayered = cv2.cvtColor(frame, cv2.COLOR_BayerRG2RGB)
                    else:
                        debayered = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    png_filename = f"image_{img_num:03d}_{timestamp}_debayered.png"
                    png_filepath = os.path.join(capture_path, png_filename)
                    cv2.imwrite(png_filepath, cv2.cvtColor(debayered, cv2.COLOR_RGB2BGR))

                    print(f"Saved debayered PNG: {png_filepath}")
                    all_saved_files.append(png_filename)

                metadata_filename = f"image_{img_num:03d}_{timestamp}_metadata.txt"
                metadata_filepath = os.path.join(capture_path, metadata_filename)

                with open(metadata_filepath, 'w') as f:
                    f.write("=" * 60 + "\n")
                    f.write(f"CAPTURE METADATA - IMAGE {img_num}/{capture_count}\n")
                    f.write("=" * 60 + "\n\n")

                    for key, value in [
                        ('Image Number', f"{metadata['image_number']} of {metadata['total_images']}"),
                        ('Camera', metadata['camera']),
                        ('Format', metadata['format']),
                        ('Timestamp', metadata['datetime']),
                        ('Resolution', metadata['resolution']),
                        ('Binning', f"{metadata['bins']}x{metadata['bins']}"),
                        ('Bit Depth', metadata['bit_depth']),
                        ('Data Type', metadata['dtype']),
                        ('Buffer Size', f"{metadata['buffer_size']} bytes"),
                        ('Array Shape', metadata['shape']),
                        ('Exposure', metadata['exposure_display']),
                        ('Gain', metadata['gain'])
                    ]:
                        f.write(f"{key}: {value}\n")

                    if metadata['temperature_c'] is not None:
                        f.write(f"CCD Temperature: {metadata['temperature_c']:.1f}°C\n")

                    f.write("\n" + "=" * 60 + "\n")
                    f.write("FILES FOR THIS IMAGE\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(f"  - {fits_filename}\n")
                    if current_format == "RAW8":
                        f.write(f"  - {png_filename}\n")
                    f.write(f"  - {metadata_filename}\n")

                print(f"Saved metadata: {metadata_filepath}")
                all_saved_files.append(metadata_filename)
                print(f"Completed image {img_num}/{capture_count}")

            summary_filename = "_session_summary.txt"
            summary_filepath = os.path.join(capture_path, summary_filename)

            with open(summary_filepath, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("CAPTURE SESSION SUMMARY\n")
                f.write("=" * 60 + "\n\n")

                for key, value in [
                    ('Session ID', session_timestamp),
                    ('Total Images', capture_count),
                    ('Format', current_format),
                    ('Camera', self.camera_info['Name']),
                    ('Session Started', session_timestamp),
                    ('Exposure', exposure_display),
                    ('Gain', self.gain_var.get()),
                    ('Binning', f"{bin_value}x{bin_value}")
                ]:
                    f.write(f"{key}: {value}\n")

                f.write("\n" + "=" * 60 + "\n")
                f.write(f"ALL FILES ({len(all_saved_files)} total)\n")
                f.write("=" * 60 + "\n\n")
                for fname in sorted(all_saved_files):
                    f.write(f"  - {fname}\n")
                f.write(f"  - {summary_filename}\n")

            all_saved_files.append(summary_filename)
            print(f"\nSaved session summary: {summary_filepath}")

            self.update_status(f"Completed: {capture_count} images saved to {capture_folder}")

            files_description = "- FITS file\n- PNG file (debayered)\n- Metadata file" if current_format == "RAW8" else "- FITS file\n- Metadata file"

            messagebox.showinfo("Success",
                                f"Capture session completed!\n\n"
                                f"Format: {current_format}\n"
                                f"Images captured: {capture_count}\n"
                                f"Folder: {capture_folder}\n\n"
                                f"Files per image:\n{files_description}\n\n"
                                f"Total files: {len(all_saved_files)}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to capture images: {e}")
            import traceback
            traceback.print_exc()

        finally:
            if was_previewing:
                self.update_status("Restarting preview...")
                self.root.update()
                time.sleep(0.1)
                self.start_preview()

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        if not self.camera or not self.camera_initialized:
            messagebox.showwarning("Warning", "Please connect to a camera first")
            return

        try:
            if not self.is_capturing:
                self.start_preview()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"video_{timestamp}.avi"
            project_name = self.project_name_entry.get().strip()
            frame_type = self.frame_type_var.get()
            if project_name:
                video_dir = os.path.join(self.save_path, project_name, frame_type)
            else:
                video_dir = os.path.join(self.save_path, frame_type)
            os.makedirs(video_dir, exist_ok=True)
            filepath = os.path.join(video_dir, filename)

            info = self.camera.get_roi_format()
            width = info[0]
            height = info[1]

            # FIX: Derive FPS from actual exposure time instead of hardcoding 30.
            # A hardcoded 30 FPS causes the AVI to play back wildly accelerated for
            # long exposures (e.g. 5s exposure at 30 FPS = 150x real speed), and the
            # write buffer backs up or drops frames. We cap at 30 FPS for short
            # exposures and floor at 1 FPS so the VideoWriter always stays valid.
            exposure_s = self._get_exposure_seconds()
            fps = min(30.0, 1.0 / exposure_s) if exposure_s > 0 else 30.0
            fps = max(fps, 1.0)

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))

            self.is_recording = True
            self.record_video_btn.configure(text="Stop Recording")
            self.update_status(f"Recording to: {filename} @ {fps:.1f} FPS")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start recording: {e}")

    def stop_recording(self):
        try:
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.record_video_btn.configure(text="Start Recording")
            self.update_status("Recording stopped")
            messagebox.showinfo("Success", "Video saved successfully")
        except Exception as e:
            self.update_status(f"Error stopping recording: {e}")

    def update_status(self, message):
        self.status_label.configure(text=message)
        print(message)

    def open_plate_solver(self):
        try:
            plate_solve_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Formal_PlateSolve_Gui.py")
            subprocess.Popen(["python3", plate_solve_path])
            self.update_status("Plate Solver opened")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open Formal_PlateSolve_Gui.py:\n{e}")

    def open_skytrack(self):
        try:
            skytrack_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SkyTracker2.py")
            self._skytrack_process = subprocess.Popen(["python3", skytrack_path])
            self.update_status("SkyTrack opened")
            self._update_skytrack_status()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open SkyTrackTest.py:\n{e}")

    def _update_skytrack_status(self):
        process = getattr(self, "_skytrack_process", None)
        if process and process.poll() is None:
            t = self.theme
            self.skytrack_signal_dot.configure(text="●", fg=t["accent"])
            self.skytrack_signal_label.configure(text="Tracking", fg=t["accent"])
            self.root.after(1000, self._update_skytrack_status)
        else:
            t = self.theme
            not_tracking_color = t["fg_tertiary"]
            self.skytrack_signal_dot.configure(text="●", fg=not_tracking_color)
            self.skytrack_signal_label.configure(text="Not tracking", fg=not_tracking_color)
            self.skytrack_target_label.configure(text="Target:")
            self.skytrack_ra_label.configure(text="RA :")
            self.skytrack_dec_label.configure(text="DEC :")

    def _poll_livewcs(self):
        if not self._livewcs_poll_running:
            return
        try:
            if os.path.exists(self.livewcs_path):
                mtime = os.path.getmtime(self.livewcs_path)
                if mtime != self._livewcs_last_mtime:
                    self._livewcs_last_mtime = mtime
                    with open(self.livewcs_path, "r") as f:
                        data = json.load(f)
                    self._apply_livewcs(data)
            else:
                self._clear_livewcs()
        except Exception as e:
            print(f"LiveWCS read error: {e}")
            self._clear_livewcs()

        self.root.after(1000, self._poll_livewcs)

    def _apply_livewcs(self, data):
        target = data.get("object", "--")
        ra     = data.get("ra_WCS", "--")
        dec    = data.get("dec_WCS", "--")

        t = self.theme
        process = getattr(self, "_skytrack_process", None)
        is_tracking = process is not None and process.poll() is None
        status_text  = "Tracking" if is_tracking else "Not tracking"
        status_color = t["accent"] if is_tracking else t["fg_tertiary"]

        self.skytrack_signal_dot.configure(text="●", fg=status_color)
        self.skytrack_signal_label.configure(text=status_text, fg=status_color)
        self.skytrack_target_label.configure(text=f"Target:  {target}")
        self.skytrack_ra_label.configure(text=f"RA :    {ra}")
        self.skytrack_dec_label.configure(text=f"DEC :  {dec}")

    def _clear_livewcs(self):
        t = self.theme
        not_tracking_color = t["fg_tertiary"]
        self.skytrack_signal_dot.configure(text="●", fg=not_tracking_color)
        self.skytrack_signal_label.configure(text="Not tracking", fg=not_tracking_color)
        self.skytrack_target_label.configure(text="Target:")
        self.skytrack_ra_label.configure(text="RA :")
        self.skytrack_dec_label.configure(text="DEC :")

    def cleanup(self):
        self._livewcs_poll_running = False
        try:
            if self.is_recording:
                self.stop_recording()
            if self.is_capturing:
                self.stop_preview()
            if self.camera:
                self.camera.close()
        except:
            pass


def main():
    root = tk.Tk()
    app = ZWOCameraGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: [app.cleanup(), root.destroy()])
    root.mainloop()


if __name__ == "__main__":
    main()
