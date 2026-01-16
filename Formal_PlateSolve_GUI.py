#!/usr/bin/env python3
"""
test_solve_local.py - Professional Edition

Modified GUI that:
 - Professional, formal appearance (greys, blacks, greens)
 - Night mode with red-scaling for night viewing
 - Clean, minimal design
 - Only accepts RA, Dec, and FOV (degrees) in the settings pane
 - Calls local astrometry.net `solve-field` via subprocess
 - Provides annotated PNG and solved image viewing
 - Uses threads for responsive UI

Dependencies (same as before + astropy, sep):
 - tkinter (builtin)
 - numpy
 - PIL (Pillow)
 - astropy
 - sep
 - scipy
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import threading
import subprocess
import glob
import os
from pathlib import Path
import time
import traceback

from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
import sep

SOLVE_FIELD_CMD = "solve-field"

# Color schemes
THEMES = {
    "day": {
        "bg_primary": "#f5f5f5",
        "bg_secondary": "#ebebeb",
        "bg_tertiary": "#e0e0e0",
        "fg_primary": "#1a1a1a",
        "fg_secondary": "#4a4a4a",
        "fg_tertiary": "#7a7a7a",
        "accent": "#2d7a3e",  # Forest green
        "accent_light": "#4a9f54",
        "accent_dark": "#1f5428",
        "canvas_bg": "#ffffff",
        "border": "#cccccc",
    },
    "night": {
        "bg_primary": "#1a0a05",
        "bg_secondary": "#2d1410",
        "bg_tertiary": "#3d1f1a",
        "fg_primary": "#ff6666",
        "fg_secondary": "#cc5555",
        "fg_tertiary": "#994444",
        "accent": "#ff4444",
        "accent_light": "#ff6666",
        "accent_dark": "#cc0000",
        "canvas_bg": "#0d0503",
        "border": "#4d2020",
    }
}

# ---------- Utility functions ----------

def run_solve_field_local(image_path, ra, dec, fov_deg, out_basename, solve_field_cmd=SOLVE_FIELD_CMD, timeout=300):
    """Run local astrometry.net solve-field with RA/DEC/FOV hints."""
    image_path = Path(image_path).resolve()
    out_dir = image_path.parent
    out_basename_path = out_dir / out_basename

    cmd = [
        solve_field_cmd,
        str(image_path),
        "--ra", str(float(ra)),
        "--dec", str(float(dec)),
        "--radius", str(float(fov_deg)),
        "--overwrite",
    ]

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(out_dir), timeout=timeout)
    except subprocess.TimeoutExpired as te:
        return {'success': False, 'stdout': te.stdout or "", 'stderr': "Timeout running solve-field", 'annotated_png': None, 'solved_fits': None, 'out_basename': str(out_basename_path)}

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    fits_candidates = []
    fits_candidates += glob.glob(str(out_basename_path) + "*.fits")
    fits_candidates += [p for p in glob.glob(str(out_dir / ("*" + image_path.stem + "*.fits"))) if out_basename in p or image_path.stem in p]
    fits_candidates = list(dict.fromkeys(fits_candidates))

    solved_fits = None
    for f in fits_candidates:
        if f.endswith(".new.fits") or f.endswith(".fits"):
            solved_fits = f
            break

    annotated_candidates = []
    annotated_candidates += glob.glob(str(out_basename_path) + "*.png")
    annotated_candidates += glob.glob(str(out_basename_path) + "*.jpg")
    annotated_candidates += glob.glob(str(out_dir / (out_basename + "-*.*")))
    annotated_candidates += glob.glob(str(out_dir / (image_path.stem + "*annot*.png")))
    annotated_candidates = [p for p in annotated_candidates if p.lower().endswith((".png", ".jpg", ".jpeg"))]
    annotated_candidates = list(dict.fromkeys(annotated_candidates))

    annotated_png = annotated_candidates[0] if annotated_candidates else None

    return {
        'success': (proc.returncode == 0),
        'stdout': stdout,
        'stderr': stderr,
        'annotated_png': annotated_png,
        'solved_fits': solved_fits,
        'out_basename': str(out_basename_path)
    }


def create_annotated_from_fits(solved_fits_path, out_png_path, max_stars=200):
    """Create an annotated PNG from a solved FITS."""
    hd = fits.open(solved_fits_path)
    data = hd[0].data
    if data is None:
        raise RuntimeError("Solved FITS has no image data")
    if data.ndim > 2:
        data = data.squeeze()
        if data.ndim > 2:
            data = data[0]

    arr = np.array(data, dtype=np.float32)

    try:
        bkg = sep.Background(arr)
        data_sub = arr - bkg.back()
        thresh = 3.0 * np.std(data_sub)
        sources = sep.extract(data_sub, thresh)
    except Exception:
        sources = np.empty((0,))

    vmin, vmax = np.percentile(arr, [2, 99.5])
    if vmax <= vmin:
        vmax = arr.max()
        vmin = arr.min()
    scaled = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    img8 = (scaled * 255).astype(np.uint8)
    pil = Image.fromarray(img8)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    draw = ImageDraw.Draw(pil)

    if isinstance(sources, np.ndarray) and sources.size:
        if 'flux' in sources.dtype.names:
            idx_sorted = np.argsort(sources['flux'])[::-1]
        else:
            idx_sorted = np.arange(len(sources))
        for i_idx, sidx in enumerate(idx_sorted[:max_stars]):
            sx = float(sources['x'][sidx])
            sy = float(sources['y'][sidx])
            r = 4 if i_idx < 15 else 2
            color = (0, 255, 0) if i_idx < 15 else (0, 200, 0)
            draw.ellipse([sx - r, sy - r, sx + r, sy + r], outline=color, width=2)

    pil.save(out_png_path)
    return out_png_path


from scipy import ndimage

class StarDetector:
    """Light wrapper for local detection."""
    def __init__(self, threshold_sigma=3.0, min_area=5):
        self.threshold_sigma = threshold_sigma
        self.min_area = min_area

    def detect_stars(self, image_data):
        arr = np.array(image_data, dtype=float)
        background = ndimage.median_filter(arr, size=20)
        img_sub = arr - background
        sigma = np.std(img_sub)
        thresh = self.threshold_sigma * sigma
        binary = img_sub > thresh
        labeled, num = ndimage.label(binary)
        stars = []
        for i in range(1, num+1):
            mask = labeled == i
            area = np.sum(mask)
            if area < self.min_area: continue
            coords = np.argwhere(mask)
            y, x = coords.mean(axis=0)
            brightness = np.sum(img_sub[mask])
            stars.append((x, y, brightness))
        stars.sort(key=lambda s: s[2], reverse=True)
        return stars[:100]


class StarFinderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Eosnyx Sky Tracker")
        self.root.geometry("1200x800")
        
        self.star_detector = StarDetector()
        self.current_image_path = None
        self.current_image = None
        self.detected_stars = []
        self.solving = False
        self.solved_fits_path = None
        self.annotated_png_path = None
        
        self.night_mode = False
        self.theme = THEMES["day"]
        
        self.setup_ui()
        self.apply_theme()

    def setup_ui(self):
        # Top control bar
        control_bar = tk.Frame(self.root)
        control_bar.pack(side=tk.TOP, fill=tk.X, padx=12, pady=8)
        
        title_label = tk.Label(control_bar, text="SKY FINDER", font=("Segoe UI", 16, "bold"))
        title_label.pack(side=tk.LEFT, padx=6)
        
        spacer = tk.Frame(control_bar)
        spacer.pack(side=tk.LEFT, expand=True)
        
        self.night_mode_btn = tk.Button(control_bar, text="☀ Day Mode", command=self.toggle_night_mode, 
                                        font=("Segoe UI", 9), padx=12, pady=4)
        self.night_mode_btn.pack(side=tk.RIGHT, padx=6)

        # Main container
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        # Left: image preview
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        preview_label = tk.Label(left_frame, text="IMAGE PREVIEW", font=("Segoe UI", 11, "bold"))
        preview_label.pack(pady=(0, 8))

        self.image_canvas = tk.Canvas(left_frame, width=550, height=550, highlightthickness=1)
        self.image_canvas.pack(padx=0, pady=0)
        
        self.image_label = tk.Label(self.image_canvas, text="No image loaded\nClick 'Load Image'",
                                    font=("Segoe UI", 11), justify=tk.CENTER)
        self.image_canvas.create_window(275, 275, window=self.image_label)

        btn_frame = tk.Frame(left_frame)
        btn_frame.pack(pady=10, fill=tk.X)

        self.load_btn = tk.Button(btn_frame, text="Load Image", command=self.choose_image,
                                  font=("Segoe UI", 10, "bold"), padx=12, pady=6)
        self.load_btn.pack(side=tk.LEFT, padx=3)

        self.detect_local_btn = tk.Button(btn_frame, text="Detect Stars", command=self.detect_stars_only,
                                          font=("Segoe UI", 10, "bold"), padx=12, pady=6)
        self.detect_local_btn.pack(side=tk.LEFT, padx=3)

        # Right: controls & results
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Settings section
        settings_label = tk.Label(right_frame, text="PLATE SOLVE HINTS", font=("Segoe UI", 11, "bold"))
        settings_label.pack(pady=(0, 8))

        settings_inner = tk.Frame(right_frame)
        settings_inner.pack(padx=0, pady=0, fill=tk.X)

        # RA
        ra_frame = tk.Frame(settings_inner)
        ra_frame.pack(fill=tk.X, pady=5)
        tk.Label(ra_frame, text="RA (°):", font=("Segoe UI", 10), width=12, anchor="w").pack(side=tk.LEFT)
        self.ra_entry = tk.Entry(ra_frame, width=16, font=("Segoe UI", 10))
        self.ra_entry.pack(side=tk.RIGHT, padx=0)
        self.ra_entry.insert(0, "0.0")

        # Dec
        dec_frame = tk.Frame(settings_inner)
        dec_frame.pack(fill=tk.X, pady=5)
        tk.Label(dec_frame, text="Dec (°):", font=("Segoe UI", 10), width=12, anchor="w").pack(side=tk.LEFT)
        self.dec_entry = tk.Entry(dec_frame, width=16, font=("Segoe UI", 10))
        self.dec_entry.pack(side=tk.RIGHT, padx=0)
        self.dec_entry.insert(0, "0.0")

        # FOV
        fov_frame = tk.Frame(settings_inner)
        fov_frame.pack(fill=tk.X, pady=5)
        tk.Label(fov_frame, text="FOV (°):", font=("Segoe UI", 10), width=12, anchor="w").pack(side=tk.LEFT)
        self.fov_entry = tk.Entry(fov_frame, width=16, font=("Segoe UI", 10))
        self.fov_entry.pack(side=tk.RIGHT, padx=0)
        self.fov_entry.insert(0, "2.0")

        info_label = tk.Label(right_frame, text="Enter RA, Dec, FOV as hints for the solver.",
                              font=("Segoe UI", 8, "italic"), justify=tk.LEFT)
        info_label.pack(pady=(6, 12), anchor="w")

        # Solve button
        self.solve_btn = tk.Button(right_frame, text="SOLVE", command=self.start_solving,
                                   font=("Segoe UI", 12, "bold"), padx=16, pady=10)
        self.solve_btn.pack(pady=12, fill=tk.X)

        # Progress
        self.progress_label = tk.Label(right_frame, text="", font=("Segoe UI", 9))
        self.progress_label.pack()
        self.progress = ttk.Progressbar(right_frame, mode="indeterminate", length=350)

        # Results
        results_label = tk.Label(right_frame, text="OUTPUT", font=("Segoe UI", 11, "bold"))
        results_label.pack(pady=(12, 6))

        results_inner = tk.Frame(right_frame)
        results_inner.pack(padx=0, pady=0, fill=tk.BOTH, expand=True)

        self.results_text = tk.Text(results_inner, height=14, width=45, font=("Courier New", 9))
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        self.results_text.insert("1.0", "Ready. Load an image and enter hints.")
        self.results_text.config(state=tk.DISABLED)

        # Bottom buttons
        bottom_frame = tk.Frame(right_frame)
        bottom_frame.pack(pady=8, fill=tk.X)
        
        self.view_annot_btn = tk.Button(bottom_frame, text="View Annotated", command=self.view_annotated, 
                                        state=tk.DISABLED, font=("Segoe UI", 9), padx=10, pady=4)
        self.view_annot_btn.pack(side=tk.LEFT, padx=2)
        
        self.view_solved_btn = tk.Button(bottom_frame, text="View Solved", command=self.view_solved, 
                                         state=tk.DISABLED, font=("Segoe UI", 9), padx=10, pady=4)
        self.view_solved_btn.pack(side=tk.LEFT, padx=2)

    def apply_theme(self):
        """Apply color theme to all widgets."""
        self.root.configure(bg=self.theme["bg_primary"])
        
        for widget in self.root.winfo_children():
            self._apply_theme_recursive(widget)

    def _apply_theme_recursive(self, widget):
        """Recursively apply theme to widget and children."""
        try:
            if isinstance(widget, tk.Label):
                widget.configure(bg=self.theme["bg_primary"], fg=self.theme["fg_primary"])
            elif isinstance(widget, tk.Button):
                widget.configure(bg=self.theme["accent"], fg="white" if not self.night_mode else self.theme["fg_primary"],
                               activebackground=self.theme["accent_light"], relief=tk.FLAT, bd=0)
            elif isinstance(widget, tk.Entry):
                widget.configure(bg=self.theme["bg_secondary"], fg=self.theme["fg_primary"],
                               insertbackground=self.theme["fg_primary"], relief=tk.FLAT, bd=1, 
                               highlightcolor=self.theme["accent"], highlightbackground=self.theme["border"])
            elif isinstance(widget, tk.Frame):
                widget.configure(bg=self.theme["bg_primary"])
            elif isinstance(widget, tk.Canvas):
                widget.configure(bg=self.theme["canvas_bg"], highlightbackground=self.theme["border"])
            elif isinstance(widget, tk.Text):
                widget.configure(bg=self.theme["bg_secondary"], fg=self.theme["fg_primary"],
                               insertbackground=self.theme["fg_primary"], relief=tk.FLAT, bd=1)
        except:
            pass
        
        for child in widget.winfo_children():
            self._apply_theme_recursive(child)

    def toggle_night_mode(self):
        """Toggle between day and night mode."""
        self.night_mode = not self.night_mode
        self.theme = THEMES["night"] if self.night_mode else THEMES["day"]
        self.night_mode_btn.configure(text="🌙 Night Mode" if self.night_mode else "☀ Day Mode")
        self.apply_theme()

    def choose_image(self):
        filetypes = (("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.fits *.fit"), ("All files", "*.*"))
        filename = filedialog.askopenfilename(title="Load image", filetypes=filetypes)
        if filename:
            self.current_image_path = filename
            self.detected_stars = []
            self.solved_fits_path = None
            self.annotated_png_path = None
            self.display_image(filename)
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete("1.0", tk.END)
            self.results_text.insert("1.0", f"✓ Loaded: {Path(filename).name}\n\nDetect stars or enter hints and solve.")
            self.results_text.config(state=tk.DISABLED)
            self.view_annot_btn.config(state=tk.DISABLED)
            self.view_solved_btn.config(state=tk.DISABLED)

    def display_image(self, image_path, stars=None):
        try:
            img = Image.open(image_path)
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            self.current_image = img.copy()
            self._display_pil_image(img, stars=stars)
        except Exception as e:
            self.image_label.config(text="Error loading image")

    def _display_pil_image(self, img, stars=None):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        display_img = img.copy()

        if stars:
            draw = ImageDraw.Draw(display_img)
            for i, (x, y, b) in enumerate(stars[:200]):
                r = 4 if i < 12 else 2
                color = (0, 255, 0) if i < 12 else (0, 200, 0)
                draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=2)

        imgw, imgh = display_img.size
        canvas_size = 550
        scale = min(canvas_size / imgw, canvas_size / imgh)
        neww, newh = max(1, int(imgw * scale)), max(1, int(imgh * scale))
        display_img = display_img.resize((neww, newh), Image.Resampling.LANCZOS)
        
        canvas_img = Image.new("RGB", (canvas_size, canvas_size), 
                              tuple(int(self.theme["canvas_bg"].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)))
        offx = (canvas_size - neww) // 2
        offy = (canvas_size - newh) // 2
        canvas_img.paste(display_img, (offx, offy))

        photo = ImageTk.PhotoImage(canvas_img)
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo

    def detect_stars_only(self):
        if not self.current_image_path:
            messagebox.showwarning("No image", "Load an image first.")
            return
        self.progress_label.config(text="Detecting stars...")
        self.detect_local_btn.config(state=tk.DISABLED)

        def do_detect():
            try:
                img = Image.open(self.current_image_path)
                if img.mode != 'L':
                    img = img.convert('L')
                arr = np.array(img)
                try:
                    bkg = sep.Background(arr.astype(np.float32))
                    data_sub = arr.astype(np.float32) - bkg.back()
                    stars = sep.extract(data_sub, 3.0)
                    star_list = [(float(s['x']), float(s['y']), float(s.get('flux', 0.0))) for s in stars]
                except Exception:
                    sd = StarDetector()
                    star_list = sd.detect_stars(arr)
                self.detected_stars = star_list
                self.root.after(0, self._show_detected_stars, star_list)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
                self.detect_local_btn.config(state=tk.NORMAL)
                self.progress_label.config(text="")

        t = threading.Thread(target=do_detect, daemon=True)
        t.start()

    def _show_detected_stars(self, stars):
        self.detect_local_btn.config(state=tk.NORMAL)
        self.progress_label.config(text=f"Found {len(stars)} stars")
        if self.current_image:
            self._display_pil_image(self.current_image, stars=stars)
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete("1.0", tk.END)
        s = f"DETECTION RESULTS\nFound {len(stars)} stars\n\nTop positions:\n"
        for i, (x, y, b) in enumerate(stars[:10], 1):
            s += f"{i}. x={x:.0f} y={y:.0f}\n"
        self.results_text.insert("1.0", s)
        self.results_text.config(state=tk.DISABLED)

    def start_solving(self):
        if not self.current_image_path:
            messagebox.showwarning("No image", "Load an image first.")
            return
        if self.solving:
            return
        try:
            ra = float(self.ra_entry.get().strip())
            dec = float(self.dec_entry.get().strip())
            fov = float(self.fov_entry.get().strip())
        except Exception:
            messagebox.showerror("Invalid input", "RA, Dec, FOV must be numeric.")
            return

        self.solving = True
        self.solve_btn.config(state=tk.DISABLED)
        self.progress_label.config(text="Solving...")
        self.progress.pack(pady=6)
        self.progress.start(10)
        thread = threading.Thread(target=self._solve_thread, args=(ra, dec, fov), daemon=True)
        thread.start()

    def _solve_thread(self, ra, dec, fov):
        try:
            image_path = Path(self.current_image_path)
            base = image_path.stem
            out_basename = base + "-solved"
            self._append_result(f"Starting solve-field...\n")
            res = run_solve_field_local(str(image_path), ra, dec, fov, out_basename)
            self._append_result("--- Output ---\n" + (res['stdout'][-500:] if res['stdout'] else "(no output)") + "\n")
            
            if not res['success']:
                self.root.after(0, self._solve_failed, res)
                return

            solved_fits = res.get('solved_fits')
            annotated = res.get('annotated_png')

            if (not annotated) and solved_fits:
                try:
                    generated_png = str(Path(solved_fits).with_suffix(".annotated.png"))
                    self._append_result("Generating annotated image...\n")
                    create_annotated_from_fits(solved_fits, generated_png)
                    annotated = generated_png
                except Exception as e:
                    self._append_result(f"Could not generate: {str(e)}\n")

            self.solved_fits_path = solved_fits
            self.annotated_png_path = annotated
            self.root.after(0, self._solve_succeeded, res)

        except Exception as e:
            self._append_result(f"Exception: {str(e)}")
            self.root.after(0, self._solve_failed, {'error': str(e)})
        finally:
            self.solving = False
            self.root.after(0, lambda: self.progress.stop())
            self.root.after(0, lambda: self.progress.pack_forget())
            self.root.after(0, lambda: self.solve_btn.config(state=tk.NORMAL))

    def _append_result(self, txt):
        self.root.after(0, lambda: self._append_result_mainthread(txt))

    def _append_result_mainthread(self, txt):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.insert(tk.END, txt + "\n")
        self.results_text.see(tk.END)
        self.results_text.config(state=tk.DISABLED)

    def _solve_failed(self, res):
        self.progress_label.config(text="✗ Failed")
        self._append_result("Solve failed.")
        messagebox.showerror("Error", "Solving failed. Check output above.")

    def _solve_succeeded(self, res):
        self.progress_label.config(text="✓ Success")
        self._append_result("Solve complete.")
        if self.annotated_png_path and Path(self.annotated_png_path).exists():
            self.view_annot_btn.config(state=tk.NORMAL)
        if self.solved_fits_path and Path(self.solved_fits_path).exists():
            self.view_solved_btn.config(state=tk.NORMAL)

    def view_annotated(self):
        if not self.annotated_png_path or not Path(self.annotated_png_path).exists():
            messagebox.showwarning("Not available", "No annotated image.")
            return
        self.display_image(self.annotated_png_path)

    def view_solved(self):
        if not self.solved_fits_path or not Path(self.solved_fits_path).exists():
            messagebox.showwarning("Not available", "No solved FITS.")
            return
        out_png = str(Path(self.solved_fits_path).with_suffix(".solved.png"))
        try:
            hd = fits.open(self.solved_fits_path)
            arr = hd[0].data
            if arr is None:
                raise RuntimeError("No data in FITS")
            if arr.ndim > 2:
                arr = arr.squeeze()
                if arr.ndim > 2:
                    arr = arr[0]
            vmin, vmax = np.percentile(arr, [2, 99.5])
            if vmax <= vmin:
                vmax = arr.max()
                vmin = arr.min()
            scaled = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
            img8 = (scaled * 255).astype(np.uint8)
            pil = Image.fromarray(img8)
            pil.save(out_png)
            self.display_image(out_png)
        except Exception as e:
            messagebox.showerror("Error", f"Could not render: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = StarFinderGUI(root)
    root.mainloop()