# Filename: modern_gui.py
# Created: 16-5-2026
# Description: Modern graphical user interface for running the simulation

import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import queue
import numpy as np
from datetime import datetime
import json
import sys
import os

# Import your QKD simulation modules
try:
    from key_distribution import run_experiment
    from key_sifting import CoincidenceDetection
    from info_reco import CascadeClient, CascadeServer
    from privacy_amp import PrivacyAmplificationServer, PrivacyAmplificationClient
    from qkd_logger import reset_logger, get_logger
    from quick import elliptic_beam_model
    import netsquid as ns
except ImportError as e:
    print(f"Warning: Could not import required modules: {e}")
    print("Make sure all QKD simulation files are in the same directory or in PYTHONPATH")

# Physics constants
h = 6.62607015e-34
c = 299792458

# ── Theme ──────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ACCENT       = "#3B8ED0"
ACCENT_HOVER = "#2F72B8"
BG_DARK      = "#1a1a2e"   # deep navy main background
BG_CARD      = "#16213e"   # slightly lighter card background
BG_PANEL     = "#0f3460"   # panel/section header background
TEXT_PRI     = "#e0e0e0"
TEXT_SEC     = "#a0a8b8"
SUCCESS      = "#2ecc71"
WARNING      = "#f39c12"
ERROR        = "#e74c3c"
TAG_SYSTEM   = "#60a0e0"
TAG_ALICE    = "#2ecc71"
TAG_BOB      = "#f39c12"
TAG_ERROR    = "#e74c3c"
TAG_SUCCESS  = "#2ecc71"


def make_section(parent, title, **grid_kwargs):
    """Rounded card with a coloured title bar."""
    outer = ctk.CTkFrame(parent, fg_color=BG_PANEL, corner_radius=12)
    outer.grid(**grid_kwargs, padx=10, pady=8)

    header = ctk.CTkFrame(outer, fg_color=BG_PANEL, corner_radius=0, height=36)
    header.pack(fill="x", padx=0, pady=0)
    ctk.CTkLabel(header, text=title, font=ctk.CTkFont(size=13, weight="bold"),
                 text_color=TEXT_PRI).pack(side="left", padx=14, pady=8)

    body = ctk.CTkFrame(outer, fg_color=BG_CARD, corner_radius=10)
    body.pack(fill="both", expand=True, padx=3, pady=(0, 3))
    return body


class ParamRow:
    """Label + entry pair inside a section body."""
    def __init__(self, parent, label, variable, row, width=160):
        ctk.CTkLabel(parent, text=label, text_color=TEXT_SEC,
                     font=ctk.CTkFont(size=12), anchor="w"
                     ).grid(row=row, column=0, sticky="w", padx=(14, 6), pady=6)
        ctk.CTkEntry(parent, textvariable=variable, width=width,
                     fg_color="#1e2a45", border_color="#2a3f6f",
                     text_color=TEXT_PRI, font=ctk.CTkFont(size=12)
                     ).grid(row=row, column=1, sticky="ew", padx=(0, 14), pady=6)
        parent.columnconfigure(1, weight=1)


class QKDSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Quantum Key Distribution Simulator")
        self.root.geometry("1280x820")
        self.root.configure(bg=BG_DARK)

        self.event_queue = queue.Queue()
        self.is_running   = False

        self.setup_config_vars()
        self._build_shell()
        self.process_queue()

    # ── Config vars (unchanged) ────────────────────────────────────────────────
    def setup_config_vars(self):
        import tkinter as tk
        self.num_pairs         = tk.IntVar(value=5000)
        self.avg_pair_rate     = tk.IntVar(value=100000)
        self.source_fidelity   = tk.DoubleVar(value=0.98)
        self.distance_a        = tk.DoubleVar(value=60.0)
        self.distance_b        = tk.DoubleVar(value=60.0)
        self.jitter_spad       = tk.DoubleVar(value=0.4)
        self.jitter_dispersion = tk.DoubleVar(value=0.5)
        self.jitter_timetagger = tk.DoubleVar(value=0.3)
        self.loss_data_path    = tk.StringVar(value="loss_data/day1_60km")
        self.use_loss_file     = tk.BooleanVar(value=False)
        self.condition         = tk.StringVar(value="day")
        self.Hb                = tk.DoubleVar(value=1.5e-3)
        self.Omega             = tk.DoubleVar(value=1e-10)
        self.Bf                = tk.DoubleVar(value=0.2)
        self.delta_t           = tk.DoubleVar(value=1e-9)
        self.receiver_aperture = tk.DoubleVar(value=0.55)
        self.wavelength        = tk.DoubleVar(value=785.0)
        self.calculated_noise  = tk.StringVar(value="0.0")

    # ── Shell: sidebar + content area ─────────────────────────────────────────
    def _build_shell(self):
        # ── Sidebar ──────────────────────────────────────────────────────────
        sidebar = ctk.CTkFrame(self.root, width=200, fg_color=BG_CARD,
                               corner_radius=0)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        # Logo / title block
        logo_block = ctk.CTkFrame(sidebar, fg_color=BG_PANEL, corner_radius=0, height=90)
        logo_block.pack(fill="x")
        ctk.CTkLabel(logo_block, text="⟨ψ|",
                     font=ctk.CTkFont(size=34, weight="bold"),
                     text_color=ACCENT).pack(pady=(14, 0))
        ctk.CTkLabel(logo_block, text="QKD Simulator",
                     font=ctk.CTkFont(size=11), text_color=TEXT_SEC).pack()

        ctk.CTkFrame(sidebar, height=1, fg_color="#2a3a5a").pack(fill="x", pady=12)

        # Nav buttons
        self._active_tab = "configure"
        self._nav_btns   = {}
        for key, icon, label in [
            ("configure", "⚙", "Configure"),
            ("run",       "▶", "Run"),
        ]:
            btn = ctk.CTkButton(
                sidebar, text=f"  {icon}  {label}",
                font=ctk.CTkFont(size=13),
                anchor="w", corner_radius=8,
                fg_color=ACCENT if key == "configure" else "transparent",
                hover_color=ACCENT_HOVER,
                text_color=TEXT_PRI,
                command=lambda k=key: self._switch_tab(k),
                height=42
            )
            btn.pack(fill="x", padx=10, pady=3)
            self._nav_btns[key] = btn

        # Bottom: appearance toggle
        ctk.CTkFrame(sidebar, height=1, fg_color="#2a3a5a").pack(fill="x", side="bottom", pady=0)
        mode_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        mode_frame.pack(side="bottom", fill="x", padx=10, pady=10)
        ctk.CTkLabel(mode_frame, text="Dark mode", text_color=TEXT_SEC,
                     font=ctk.CTkFont(size=11)).pack(side="left")
        ctk.CTkSwitch(mode_frame, text="", width=44,
                      command=self._toggle_mode,
                      onvalue="dark", offvalue="light",
                      variable=ctk.StringVar(value="dark")).pack(side="right")

        # ── Content pane ─────────────────────────────────────────────────────
        self.content = ctk.CTkFrame(self.root, fg_color=BG_DARK, corner_radius=0)
        self.content.pack(side="left", fill="both", expand=True)

        self._pages = {}
        for key, builder in [("configure", self.build_configure_tab),
                              ("run",       self.build_run_tab)]:
            page = ctk.CTkFrame(self.content, fg_color=BG_DARK, corner_radius=0)
            page.place(relx=0, rely=0, relwidth=1, relheight=1)
            self._pages[key] = page
            builder(page)

        self._pages["configure"].lift()

    def _switch_tab(self, key):
        self._active_tab = key
        for k, btn in self._nav_btns.items():
            btn.configure(fg_color=ACCENT if k == key else "transparent")
        self._pages[key].lift()

    def _toggle_mode(self):
        mode = ctk.get_appearance_mode()
        ctk.set_appearance_mode("light" if mode == "Dark" else "dark")

    # ── Configure tab ─────────────────────────────────────────────────────────
    def build_configure_tab(self, parent):
        # Page title
        ctk.CTkLabel(parent, text="Simulation Configuration",
                     font=ctk.CTkFont(size=20, weight="bold"),
                     text_color=TEXT_PRI).grid(row=0, column=0, columnspan=2,
                                               sticky="w", padx=20, pady=(18, 4))

        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.rowconfigure(1, weight=1)
        parent.rowconfigure(2, weight=1)

        # ── QKD Parameters ───────────────────────────────────────────────────
        qkd = make_section(parent, "QKD Parameters",
                           row=1, column=0, sticky="nsew")
        for i, (lbl, var) in enumerate([
            ("Number of Pairs",      self.num_pairs),
            ("Avg Pair Rate (Hz)",   self.avg_pair_rate),
            ("Source Fidelity²",     self.source_fidelity),
            ("Distance to Alice (km)", self.distance_a),
            ("Distance to Bob (km)", self.distance_b),
        ]):
            ParamRow(qkd, lbl, var, i)

        # ── Jitter ───────────────────────────────────────────────────────────
        jit = make_section(parent, "Jitter Components (ns)",
                           row=2, column=0, sticky="nsew")
        for i, (lbl, var) in enumerate([
            ("SPAD Jitter",  self.jitter_spad),
            ("Dispersion",   self.jitter_dispersion),
            ("Time Tagger",  self.jitter_timetagger),
        ]):
            ParamRow(jit, lbl, var, i)

        # ── Atmosphere Model ─────────────────────────────────────────────────
        atm = make_section(parent, "Atmosphere Model",
                           row=1, column=1, sticky="nsew")

        cb = ctk.CTkCheckBox(atm, text="Use Loss Data File",
                             variable=self.use_loss_file,
                             text_color=TEXT_PRI, font=ctk.CTkFont(size=12),
                             fg_color=ACCENT, hover_color=ACCENT_HOVER)
        cb.grid(row=0, column=0, columnspan=3, sticky="w", padx=14, pady=(12, 6))

        ctk.CTkEntry(atm, textvariable=self.loss_data_path, width=200,
                     fg_color="#1e2a45", border_color="#2a3f6f",
                     text_color=TEXT_PRI, font=ctk.CTkFont(size=11)
                     ).grid(row=1, column=0, sticky="ew", padx=(14, 6), pady=6)
        atm.columnconfigure(0, weight=1)

        ctk.CTkButton(atm, text="Browse", width=80,
                      fg_color=BG_PANEL, hover_color=ACCENT,
                      text_color=TEXT_PRI, font=ctk.CTkFont(size=12),
                      command=self.browse_loss_file
                      ).grid(row=1, column=1, padx=4, pady=6)

        ctk.CTkButton(atm, text="Visualize", width=88,
                      fg_color=ACCENT, hover_color=ACCENT_HOVER,
                      text_color="white", font=ctk.CTkFont(size=12),
                      command=self.visualize_loss
                      ).grid(row=1, column=2, padx=(4, 14), pady=6)

        # Descriptive blurb to fill space
        ctk.CTkLabel(atm,
                     text="Load pre-recorded transmittance samples that\n"
                          "model the free-space atmospheric channel.\n"
                          "Leave unchecked to use a fixed unit loss.",
                     text_color=TEXT_SEC, font=ctk.CTkFont(size=11),
                     justify="left"
                     ).grid(row=2, column=0, columnspan=3,
                            sticky="w", padx=14, pady=(10, 14))

        # ── Noise Model ──────────────────────────────────────────────────────
        noi = make_section(parent, "Noise Model",
                           row=2, column=1, sticky="nsew")

        # Day / Night toggle (segmented button style)
        seg_frame = ctk.CTkFrame(noi, fg_color="transparent")
        seg_frame.grid(row=0, column=0, columnspan=2, sticky="w", padx=14, pady=(12, 6))
        ctk.CTkLabel(seg_frame, text="Condition Preset:",
                     text_color=TEXT_SEC, font=ctk.CTkFont(size=12)).pack(side="left", padx=(0, 10))
        ctk.CTkSegmentedButton(
            seg_frame, values=["Day", "Night"],
            variable=ctk.StringVar(value="Day"),
            command=lambda v: self.set_day_preset() if v == "Day" else self.set_night_preset(),
            selected_color=ACCENT, selected_hover_color=ACCENT_HOVER,
            unselected_color=BG_PANEL, text_color=TEXT_PRI,
            font=ctk.CTkFont(size=12), width=150
        ).pack(side="left")

        for i, (lbl, var) in enumerate([
            ("Hb (W m⁻² sr⁻¹ nm⁻¹)", self.Hb),
            ("Ω (sr)",                 self.Omega),
            ("Bf (nm)",                self.Bf),
            ("Δt (s)",                 self.delta_t),
            ("Receiver Aperture (m)",  self.receiver_aperture),
            ("Wavelength (nm)",        self.wavelength),
        ], start=1):
            ParamRow(noi, lbl, var, i)

        calc_row = ctk.CTkFrame(noi, fg_color="transparent")
        calc_row.grid(row=7, column=0, columnspan=2, sticky="w", padx=14, pady=(8, 14))
        ctk.CTkButton(calc_row, text="Calculate Noise", width=140,
                      fg_color=ACCENT, hover_color=ACCENT_HOVER,
                      text_color="white", font=ctk.CTkFont(size=12),
                      command=self.calculate_noise).pack(side="left")
        ctk.CTkLabel(calc_row, text="Nbg:", text_color=TEXT_SEC,
                     font=ctk.CTkFont(size=12)).pack(side="left", padx=(20, 6))
        ctk.CTkEntry(calc_row, textvariable=self.calculated_noise, width=140,
                     state="readonly", fg_color="#1e2a45", border_color="#2a3f6f",
                     text_color=SUCCESS, font=ctk.CTkFont(size=12)
                     ).pack(side="left")

        # Footer
        footer = ctk.CTkFrame(parent, fg_color="transparent")
        footer.grid(row=3, column=0, columnspan=2, pady=(4, 14))
        ctk.CTkLabel(footer,
                     text="To learn how different parameters affect performance, "
                          "see our paper: arXiv:2501.12345",
                     text_color=TEXT_SEC, font=ctk.CTkFont(size=11)
                     ).pack()

        self.calculate_noise()

    # ── Run tab ───────────────────────────────────────────────────────────────
    def build_run_tab(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        # Page title
        ctk.CTkLabel(parent, text="Run Simulation",
                     font=ctk.CTkFont(size=20, weight="bold"),
                     text_color=TEXT_PRI).grid(row=0, column=0,
                                               sticky="w", padx=20, pady=(18, 4))

        # ── Control bar ──────────────────────────────────────────────────────
        ctrl = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=12)
        ctrl.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 10))

        self.run_button = ctk.CTkButton(
            ctrl, text="▶  Run Simulation", width=160, height=38,
            fg_color=SUCCESS, hover_color="#27ae60",
            text_color="white", font=ctk.CTkFont(size=13, weight="bold"),
            command=self.start_simulation)
        self.run_button.pack(side="left", padx=12, pady=10)

        self.stop_button = ctk.CTkButton(
            ctrl, text="⏹  Stop", width=100, height=38,
            fg_color="#c0392b", hover_color="#a93226",
            text_color="white", font=ctk.CTkFont(size=13),
            state="disabled", command=self.stop_simulation)
        self.stop_button.pack(side="left", padx=(0, 8), pady=10)

        self.clear_button = ctk.CTkButton(
            ctrl, text="Clear Log", width=100, height=38,
            fg_color=BG_PANEL, hover_color=ACCENT,
            text_color=TEXT_PRI, font=ctk.CTkFont(size=12),
            command=self.clear_events)
        self.clear_button.pack(side="left", padx=(0, 20), pady=10)

        # Status badge
        self.status_label = ctk.CTkLabel(
            ctrl, text="● Ready",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=SUCCESS)
        self.status_label.pack(side="left", padx=10)

        # Progress (right-aligned)
        prog_frame = ctk.CTkFrame(ctrl, fg_color="transparent")
        prog_frame.pack(side="right", padx=16, pady=10)
        self.progress_label = ctk.CTkLabel(
            prog_frame, text="0 / 0 pairs",
            text_color=TEXT_SEC, font=ctk.CTkFont(size=11))
        self.progress_label.pack(anchor="e")
        self.progress_bar = ctk.CTkProgressBar(prog_frame, width=280,
                                               progress_color=ACCENT,
                                               fg_color="#1e2a45")
        self.progress_bar.pack(pady=(4, 0))
        self.progress_bar.set(0)

        # ── Log ──────────────────────────────────────────────────────────────
        log_outer = ctk.CTkFrame(parent, fg_color=BG_PANEL, corner_radius=12)
        log_outer.grid(row=2, column=0, sticky="nsew", padx=20, pady=(0, 16))

        log_header = ctk.CTkFrame(log_outer, fg_color=BG_PANEL, corner_radius=0, height=36)
        log_header.pack(fill="x")
        ctk.CTkLabel(log_header, text="Simulation Events",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=TEXT_PRI).pack(side="left", padx=14, pady=8)

        import tkinter as tk
        log_body = ctk.CTkFrame(log_outer, fg_color=BG_CARD, corner_radius=10)
        log_body.pack(fill="both", expand=True, padx=3, pady=(0, 3))

        self.events_text = tk.Text(
            log_body, wrap="word", bg=BG_CARD, fg=TEXT_PRI,
            insertbackground=TEXT_PRI, selectbackground=ACCENT,
            font=("Consolas", 10), bd=0, padx=14, pady=10,
            relief="flat", highlightthickness=0)
        self.events_text.pack(side="left", fill="both", expand=True)

        sb = ctk.CTkScrollbar(log_body, command=self.events_text.yview,
                              button_color=ACCENT, button_hover_color=ACCENT_HOVER)
        sb.pack(side="right", fill="y")
        self.events_text.configure(yscrollcommand=sb.set)

        self.events_text.tag_config("system",  foreground=TAG_SYSTEM)
        self.events_text.tag_config("source",  foreground="#b57bee")
        self.events_text.tag_config("alice",   foreground=TAG_ALICE)
        self.events_text.tag_config("bob",     foreground=TAG_BOB)
        self.events_text.tag_config("error",   foreground=TAG_ERROR)
        self.events_text.tag_config("success", foreground=TAG_SUCCESS,
                                    font=("Consolas", 10, "bold"))

    # ── All methods below are UNCHANGED from original ─────────────────────────

    def create_param_row(self, parent, label, variable, row, format_sci=False):
        ParamRow(parent, label, variable, row)

    def browse_loss_file(self):
        filename = filedialog.askopenfilename(
            title="Select Loss Data File",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
        if filename:
            if filename.endswith('.pkl'):
                filename = filename[:-4]
            self.loss_data_path.set(filename)
            self.use_loss_file.set(True)

    def visualize_loss(self):
        try:
            from quick import elliptic_beam_model
            model = elliptic_beam_model()
            loss_data = model.load_data(self.loss_data_path.get())
            if loss_data is None:
                messagebox.showerror("Error", "Could not load loss data file")
                return
            viz_window = ctk.CTkToplevel(self.root)
            viz_window.title("Loss Data Distribution")
            viz_window.geometry("800x600")
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            transmittance_array = np.array(loss_data)
            data_range = np.max(transmittance_array) - np.min(transmittance_array)
            iqr = np.percentile(transmittance_array, 75) - np.percentile(transmittance_array, 25)
            fd_bin_width = 2 * iqr * (len(transmittance_array) ** (-1/3))
            n_bins = max(10, min(100, int(data_range / fd_bin_width)))
            fig = Figure(figsize=(10, 6), facecolor=BG_CARD)
            ax = fig.add_subplot(111)
            ax.set_facecolor(BG_DARK)
            count, bins, _ = ax.hist(transmittance_array, bins=n_bins, alpha=0)
            total_items = np.sum(count)
            normalized_counts = count / total_items
            ax.clear()
            ax.set_facecolor(BG_DARK)
            ax.bar(bins[:-1], normalized_counts, width=bins[1]-bins[0],
                   alpha=0.85, color=ACCENT, edgecolor="#2a3f6f", linewidth=0.5)
            ax.set_xlabel('Transmittance', color=TEXT_SEC)
            ax.set_ylabel('Probability', color=TEXT_SEC)
            ax.set_title('Loss Data Distribution', color=TEXT_PRI, fontsize=14)
            ax.tick_params(colors=TEXT_SEC)
            ax.grid(True, alpha=0.15, color=TEXT_SEC)
            for spine in ax.spines.values():
                spine.set_edgecolor("#2a3f6f")
            stats_text = (f'Mean: {np.mean(transmittance_array):.6f}\n'
                          f'Std:  {np.std(transmittance_array):.6f}\n'
                          f'Min:  {np.min(transmittance_array):.6f}\n'
                          f'Max:  {np.max(transmittance_array):.6f}')
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    verticalalignment='top', color=TEXT_PRI,
                    bbox=dict(boxstyle='round', facecolor=BG_PANEL, alpha=0.9,
                              edgecolor="#2a3f6f"))
            canvas = FigureCanvasTkAgg(fig, master=viz_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
        except Exception as e:
            messagebox.showerror("Error", f"Could not visualize loss data:\n{str(e)}")

    def set_day_preset(self):
        self.Hb.set(1.5e-3)
        self.Omega.set(1e-8)
        self.Bf.set(1.0)
        self.calculate_noise()

    def set_night_preset(self):
        self.Hb.set(1.5e-6)
        self.Omega.set(1e-8)
        self.Bf.set(1.0)
        self.calculate_noise()

    def calculate_noise(self):
        A_rx = np.pi * (self.receiver_aperture.get() ** 2)
        wavelength = self.wavelength.get() * 1e-9
        energy = h * c / wavelength
        photon_radiance = self.Hb.get() / energy
        nbg = photon_radiance * self.Omega.get() * A_rx * self.Bf.get() * self.delta_t.get()
        self.calculated_noise.set(f"{nbg:.6e}")

    def add_event(self, source, sim_time, event_type, message):
        event = {
            'source': source, 'sim_time': sim_time,
            'event_type': event_type, 'message': message,
            'timestamp': datetime.now()
        }
        self.event_queue.put(('event', event))

    def update_progress(self, current, total):
        self.event_queue.put(('progress', (current, total)))

    def process_queue(self):
        try:
            while True:
                item = self.event_queue.get_nowait()
                item_type, data = item
                if item_type == 'event':
                    self.display_event(data)
                elif item_type == 'progress':
                    current, total = data
                    self.progress_bar.set(current / total if total > 0 else 0)
                    self.progress_label.configure(
                        text=f"{current} / {total} pairs generated")
                elif item_type == 'status':
                    status, color = data
                    self.status_label.configure(text=f"● {status}",
                                                text_color=color)
                elif item_type == 'complete':
                    self.simulation_complete()
        except queue.Empty:
            pass
        self.root.after(100, self.process_queue)

    def display_event(self, event):
        sim_time_str = (f"{event['sim_time']/1e9:.6f}s"
                        if event['sim_time'] > 0 else "0.000000s")
        timestamp = event['timestamp'].strftime("%H:%M:%S")
        message = (f"[{timestamp}] [{event['source']:8s}] "
                   f"t={sim_time_str}: {event['message']}\n")
        tag = event['source']
        if event['event_type'] == 'error':
            tag = 'error'
        elif event['event_type'] in ['success', 'privacy_amplification_complete']:
            tag = 'success'
        self.events_text.insert("end", message, tag)
        self.events_text.see("end")

    def clear_events(self):
        self.events_text.delete("1.0", "end")
        self.progress_bar.set(0)
        self.progress_label.configure(text="0 / 0 pairs")

    def start_simulation(self):
        if self.is_running:
            return
        self.is_running = True
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.event_queue.put(('status', ('Running…', WARNING)))
        threading.Thread(target=self.run_simulation, daemon=True).start()

    def stop_simulation(self):
        if self.is_running:
            self.is_running = False
            self.add_event('system', 0, 'stop', 'Simulation stopped by user')
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.event_queue.put(('status', ('Stopped', ERROR)))

    def simulation_complete(self):
        self.is_running = False
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.event_queue.put(('status', ('Complete', SUCCESS)))

    def run_simulation(self):
        """Main simulation logic — unchanged."""
        try:
            from key_distribution import run_experiment
            from key_sifting import CoincidenceDetection
            from info_reco import CascadeClient, CascadeServer
            from privacy_amp import PrivacyAmplificationServer, PrivacyAmplificationClient
            from qkd_logger import reset_logger
            from quick import elliptic_beam_model
            import netsquid as ns

            loss_array = [1]
            if self.use_loss_file.get() and self.loss_data_path.get():
                try:
                    model = elliptic_beam_model()
                    loss_array = model.load_data(self.loss_data_path.get())
                    self.add_event('system', 0, 'file_load',
                                   f"Loaded loss data: {len(loss_array)} samples")
                except Exception as e:
                    self.add_event('system', 0, 'error',
                                   f"Failed to load loss data: {str(e)}")
                    loss_array = [1]

            config = {
                'num_pairs':         self.num_pairs.get(),
                'avg_pair_rate_hz':  self.avg_pair_rate.get(),
                'source_fidelity_sq': self.source_fidelity.get(),
                'node_distance_a':   self.distance_a.get(),
                'node_distance_b':   self.distance_b.get(),
                'loss_array':        loss_array,
                'error_prob':        float(self.calculated_noise.get()),
                'jitter_components': [
                    self.jitter_spad.get(),
                    self.jitter_dispersion.get(),
                    self.jitter_timetagger.get()
                ],
                'progress_callback': self.update_progress
            }

            self.add_event('system', 0, 'start',
                           f"Starting simulation with {config['num_pairs']} pairs")

            logger = reset_logger()
            original_log = logger.log_event
            def hooked_log(source, sim_time, event_type, data=None, message=""):
                original_log(source, sim_time, event_type, data, message)
                if self.is_running:
                    self.add_event(source, sim_time, event_type, message)
            logger.log_event = hooked_log

            alice_collector, bob_collector, alice, bob = run_experiment(config)
            if not self.is_running:
                self.add_event('system', ns.sim_time(), 'stop', 'Key distribution stopped')
                return

            with open('qkd_raw_data/metadata.json') as f:
                data = json.load(f)
            is_positive = data['expected_delay_diff_ns'] > 0

            self.add_event('system', ns.sim_time(), 'coincidence_start',
                           'Starting coincidence detection...')
            alice_c = CoincidenceDetection(alice, data_collector=alice_collector,
                                           port=alice.get_conn_port(bob.ID),
                                           name="alice_c", first=is_positive)
            bob_c = CoincidenceDetection(bob, data_collector=bob_collector,
                                         port=bob.get_conn_port(alice.ID),
                                         name="bob_c", first=not is_positive)
            alice_c.start(); bob_c.start(); ns.sim_run()
            alice_c.stop();  bob_c.stop()
            if not self.is_running:
                self.add_event('system', ns.sim_time(), 'stop', 'Coincidence detection stopped')
                return

            with open('qkd_raw_data/metadata.json') as f:
                data = json.load(f)
            qber = data['qber']
            is_secure = data['is_secure']
            if not is_secure:
                self.add_event('system', ns.sim_time(), 'error',
                               'Channel not secure! QBER too high.')
                self.event_queue.put(('complete', None))
                return

            self.add_event('system', ns.sim_time(), 'reconciliation_start',
                           'Starting information reconciliation...')
            alice_r = CascadeServer(alice, port=alice.get_conn_port(bob.ID),
                                    alice_key_file="alice.txt", name="alice_r")
            bob_r = CascadeClient(bob, port=bob.get_conn_port(alice.ID),
                                  bob_key_file="bob.txt",
                                  algorithm_name="original",
                                  estimated_bit_error_rate=qber, name="bob_r")
            alice_r.start(); bob_r.start(); ns.sim_run()
            alice_r.stop();  bob_r.stop()
            if not self.is_running:
                self.add_event('system', ns.sim_time(), 'stop', 'Reconciliation stopped')
                return

            self.add_event('system', ns.sim_time(), 'privacy_amp_start',
                           'Starting privacy amplification...')
            alice_pa = PrivacyAmplificationServer(
                alice, port=alice.get_conn_port(bob.ID),
                alice_key_file="alice_recon.txt",
                output_file="alice_amplified.txt", name="alice_pa")
            bob_pa = PrivacyAmplificationClient(
                bob, port=bob.get_conn_port(alice.ID),
                bob_key_file="bob_recon.txt",
                output_file="bob_amplified.txt", name="bob_pa")
            alice_pa.start(); bob_pa.start(); ns.sim_run()
            alice_pa.stop();  bob_pa.stop()
            if not self.is_running:
                self.add_event('system', ns.sim_time(), 'stop', 'Privacy amplification stopped')
                return

            self.add_event('system', ns.sim_time(), 'success',
                           '✓ Simulation completed successfully!')

        except Exception as e:
            self.add_event('system', 0, 'error', f'Error: {str(e)}')
            import traceback; traceback.print_exc()
        finally:
            self.is_running = False
            self.event_queue.put(('complete', None))


def main():
    root = ctk.CTk()
    app = QKDSimulatorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
