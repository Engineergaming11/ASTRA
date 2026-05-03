import tkinter as tk
from tkinter import ttk

from astra_dialogs import askyesno, showerror
import serial
import serial.tools.list_ports

class MountDriver:
    def __init__(self, root):
        self.root = root
        self.root.title("Mount Driver")
        self.root.configure(bg="#2d2d2d")
        self.root.resizable(False, False)
        self.ser = None
        self._build()

    def _build(self):
        # --- Port row ---
        top = tk.Frame(self.root, bg="#2d2d2d")
        top.pack(fill="x", padx=10, pady=8)

        tk.Label(top, text="Port:", bg="#2d2d2d", fg="white").pack(side="left")
        ports = [p.device for p in serial.tools.list_ports.comports()] or ["/dev/ttyUSB0"]
        self.port_var = tk.StringVar(value=ports[0])
        self.port_menu = ttk.Combobox(top, textvariable=self.port_var, values=ports, width=14, state="readonly")
        self.port_menu.pack(side="left", padx=5)

        tk.Button(top, text="↺", bg="#3d3d3d", fg="white", command=self.refresh_ports).pack(side="left")
        self.btn_conn = tk.Button(top, text="Connect", bg="#4a4a4a", fg="white", width=10, command=self.toggle_connection)
        self.btn_conn.pack(side="left", padx=6)
        self.lbl_status = tk.Label(top, text="●  Disconnected", bg="#2d2d2d", fg="#888")
        self.lbl_status.pack(side="left")

        # --- Command Center ---
        cc = tk.LabelFrame(self.root, text="Command Center", bg="#2d2d2d", fg="white", padx=8, pady=8)
        cc.pack(padx=10, pady=(0, 10))

        # D-pad
        pad = tk.Frame(cc, bg="#2d2d2d")
        pad.grid(row=0, column=0, padx=(0, 16))

        def make_dir(txt, cmd, stop, r, c):
            b = tk.Button(pad, text=txt, width=4, height=2, bg="#4a4a4a", fg="white")
            b.grid(row=r, column=c, padx=2, pady=2)
            b.bind("<ButtonPress-1>",   lambda e: self.write(cmd))
            b.bind("<ButtonRelease-1>", lambda e: self.write(stop))

        make_dir("N", ":mn#", ":qD#", 0, 1)
        make_dir("W", ":mw#", ":qR#", 1, 0)
        make_dir("E", ":me#", ":qR#", 1, 2)
        make_dir("S", ":ms#", ":qD#", 2, 1)

        # Right side controls
        right = tk.Frame(cc, bg="#2d2d2d")
        right.grid(row=0, column=1, sticky="ns")

        tk.Button(right, text="STOP", bg="#cc0000", fg="white", font=("Arial", 10, "bold"),
                  width=12, command=lambda: self.write(":Q#")).pack(pady=(0, 4))

        tk.Button(right, text="Set Zero", bg="#3d3d3d", fg="white", width=12,
                  command=self.set_zero).pack(pady=2)
        tk.Button(right, text="Go to Zero", bg="#3d3d3d", fg="white", width=12,
                  command=lambda: self.write(":MH#")).pack(pady=2)

        tk.Label(right, text="Rate", bg="#2d2d2d", fg="white").pack(pady=(8, 0))
        self.rate = tk.IntVar(value=5)
        tk.Scale(right, from_=1, to=9, variable=self.rate, orient="horizontal",
                 bg="#2d2d2d", fg="white", troughcolor="#3d3d3d", showvalue=True,
                 command=lambda v: self.write(f":SR{v}#")).pack()

    # --- Serial helpers ---
    def write(self, cmd):
        if self.ser and self.ser.is_open:
            self.ser.write(cmd.encode("ascii"))

    def refresh_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()] or ["/dev/ttyUSB0"]
        self.port_menu["values"] = ports
        self.port_var.set(ports[0])

    def set_zero(self):
        if askyesno("Set Zero", "Mark current position as zero?", parent=self.root):
            self.write(":SZP#")

    def toggle_connection(self):
        if self.ser is None:
            try:
                self.ser = serial.Serial(self.port_var.get(), 115200, timeout=0.1)
                self.write(":MountInfo#"); self.ser.read(4)
                self.btn_conn.config(text="Disconnect", bg="#cc0000")
                self.lbl_status.config(text="●  Connected", fg="#00cc44")
            except Exception as e:
                showerror("Serial Error", str(e), parent=self.root)
        else:
            self.ser.close(); self.ser = None
            self.btn_conn.config(text="Connect", bg="#4a4a4a")
            self.lbl_status.config(text="●  Disconnected", fg="#888")

if __name__ == "__main__":
    root = tk.Tk()
    MountDriver(root)
    root.mainloop()