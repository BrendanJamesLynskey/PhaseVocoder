"""
Wavelet Audio Processor — Windows 11 Desktop Application
=========================================================
Time-stretching and pitch-shifting using the Continuous Wavelet Transform
(CWT) with Morlet wavelets instead of the STFT-based phase vocoder.

Features:
  * Record audio from microphone or load WAV files
  * CWT-based time stretching (0.25x - 4.0x)
  * CWT-based pitch shifting (-12 to +12 semitones)
  * Before/after scalogram display (log-frequency)
  * Playback of original and processed audio
  * Export processed audio to WAV

Requirements (install via pip):
  pip install numpy scipy sounddevice soundfile matplotlib

Usage:
  python wavelet_processor.py
"""

import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sys.exit("Missing dependency:  pip install sounddevice")
try:
    import soundfile as sf
except ImportError:
    sys.exit("Missing dependency:  pip install soundfile")
try:
    from scipy.signal import fftconvolve, resample
except ImportError:
    sys.exit("Missing dependency:  pip install scipy")
try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ImportError:
    sys.exit("Missing dependency:  pip install matplotlib")


# ----------------------------------------------
#  Continuous Wavelet Transform Core
# ----------------------------------------------

def morlet_wavelet(N, scale, omega0=6.0):
    """Analytic Morlet wavelet centred in a window of N samples."""
    t = np.arange(N) - (N - 1) / 2.0
    t = t / scale
    norm = (np.pi ** -0.25) / np.sqrt(scale)
    return norm * np.exp(1j * omega0 * t) * np.exp(-0.5 * t ** 2)


def cwt_morlet(x, scales, omega0=6.0):
    """Continuous Wavelet Transform using the Morlet wavelet.
    Returns complex array of shape (len(scales), len(x)).
    """
    n = len(x)
    out = np.zeros((len(scales), n), dtype=np.complex128)
    for i, s in enumerate(scales):
        half = int(4.0 * s)
        wlen = 2 * half + 1
        psi = morlet_wavelet(wlen, s, omega0)
        out[i, :] = fftconvolve(x, psi[::-1].conj(), mode="same")
    return out


def icwt_morlet(W, scales, omega0=6.0):
    """Inverse CWT (approximate reconstruction from Morlet coefficients)."""
    ds = np.gradient(scales)
    real_sum = np.zeros(W.shape[1])
    for i, s in enumerate(scales):
        real_sum += np.real(W[i, :]) * ds[i] / (s ** 1.5)
    C_psi = 0.776  # empirical for omega0=6
    return real_sum / C_psi


def make_scales(sr, n_voices=64, fmin=20.0, fmax=20000.0, omega0=6.0):
    """Logarithmically-spaced wavelet scales from fmin to fmax Hz."""
    fmax = min(fmax, sr / 2 * 0.95)  # stay below Nyquist
    fmin = max(fmin, 10.0)
    freqs = np.geomspace(fmax, fmin, n_voices)
    return sr * omega0 / (2.0 * np.pi * freqs)


def scale_to_freq(scales, sr, omega0=6.0):
    """Convert CWT scales to approximate centre frequencies (Hz)."""
    return sr * omega0 / (2.0 * np.pi * scales)


# ----------------------------------------------
#  Time Stretch & Pitch Shift via CWT
# ----------------------------------------------

def wavelet_time_stretch(x, stretch, sr, n_voices=64, fmin=20.0, fmax=20000.0):
    """Time-stretch a mono signal by *stretch* using the CWT.

    Uses a residual-preservation approach:
    1. Analyse with CWT and reconstruct to find what the CWT captures
    2. Scale CWT reconstruction to best-fit the original (least-squares)
    3. Compute residual = original - scaled_reconstruction
    4. Stretch CWT coefficients via instantaneous-frequency phase synthesis
    5. Resample the residual to the target length
    6. Combine both for full-bandwidth output

    stretch > 1 = slower / longer
    stretch < 1 = faster / shorter
    """
    if abs(stretch - 1.0) < 1e-6:
        return x.copy()

    omega0 = 6.0
    fmax = min(fmax, sr / 2 * 0.95)
    scales = make_scales(sr, n_voices, fmin, fmax, omega0)

    # Forward CWT
    W = cwt_morlet(x, scales, omega0)

    # Reconstruct and find best-fit scaling factor so that
    # scale_factor * icwt(cwt(x)) ~ x in a least-squares sense
    x_cwt = icwt_morlet(W, scales, omega0)
    scale_factor = np.sum(x * x_cwt) / (np.sum(x_cwt ** 2) + 1e-10)
    residual = x - x_cwt * scale_factor

    # --- CWT stretch via instantaneous frequency ---
    n_in = W.shape[1]
    n_out = max(1, int(round(n_in * stretch)))

    t_in = np.arange(n_in, dtype=np.float64)
    t_out = np.linspace(0, n_in - 1, n_out)

    W_stretched = np.zeros((len(scales), n_out), dtype=np.complex128)
    for i in range(len(scales)):
        mag = np.abs(W[i, :])
        phase = np.unwrap(np.angle(W[i, :]))

        # Instantaneous frequency (radians / sample)
        inst_freq = np.gradient(phase)

        # Interpolate magnitude and instantaneous frequency
        mag_out = np.interp(t_out, t_in, mag)
        if_out = np.interp(t_out, t_in, inst_freq)

        # Resynthesize phase by integrating inst. freq
        phase_out = np.cumsum(if_out)
        phase_out = phase_out - phase_out[0] + phase[0]

        W_stretched[i, :] = mag_out * np.exp(1j * phase_out)

    y_cwt = icwt_morlet(W_stretched, scales, omega0) * scale_factor

    # Stretch residual by simple resampling (preserves spectral character
    # of content the CWT doesn't capture, mostly very high/low freqs)
    residual_stretched = resample(residual, n_out)

    return y_cwt + residual_stretched


def wavelet_pitch_shift(x, semitones, sr, n_voices=64, fmin=20.0, fmax=20000.0,
                        target_length=None):
    """Shift pitch by *semitones* using CWT time-stretch + resampling.

    The signal is first time-stretched by the pitch ratio (preserving
    pitch via the wavelet transform), then resampled back to the target
    length to shift all frequencies.
    """
    if abs(semitones) < 1e-6:
        return x.copy()

    factor = 2.0 ** (semitones / 12.0)
    stretched = wavelet_time_stretch(x, factor, sr, n_voices, fmin, fmax)
    tgt = target_length if target_length is not None else len(x)
    return resample(stretched, tgt)


# ----------------------------------------------
#  GUI Application
# ----------------------------------------------

class WaveletProcessorApp:
    SR = 44100
    REC_MAX = 30

    def __init__(self, root):
        self.root = root
        self.root.title("Wavelet Audio Processor")
        self.root.minsize(1100, 780)
        self.root.configure(bg="#1a1b26")

        self.audio_orig = None
        self.audio_proc = None
        self.sr = self.SR
        self.is_recording = False
        self.rec_frames = []
        self._play_stream = None
        self.processing = False

        # Style - Tokyo Night palette
        style = ttk.Style()
        style.theme_use("clam")
        BG = "#1a1b26"; FG = "#c0caf5"; ACCENT = "#7aa2f7"
        SURFACE = "#16161e"; MUTED = "#565f89"
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=FG, font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 13, "bold"),
                         foreground=ACCENT, background=BG)
        style.configure("TButton", font=("Segoe UI", 10), padding=6)
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("TScale", background=BG)
        style.configure("TLabelframe", background=BG, foreground=FG)
        style.configure("TLabelframe.Label", background=BG, foreground=ACCENT,
                         font=("Segoe UI", 10, "bold"))
        style.configure("Status.TLabel", background=SURFACE, foreground=MUTED,
                         font=("Segoe UI", 9))
        self._bg = BG; self._surface = SURFACE
        self._fg = FG; self._muted = MUTED

        self._build_ui()
        self._update_status("Ready - load a file or record audio to begin.")

    def _build_ui(self):
        ctrl = ttk.Frame(self.root)
        ctrl.pack(fill="x", padx=12, pady=(10, 4))

        # --- Source row ---
        src = ttk.LabelFrame(ctrl, text="  Source  ", padding=8)
        src.pack(fill="x", pady=(0, 6))
        self.btn_load = ttk.Button(src, text="\U0001f4c2 Load WAV", command=self._load_file)
        self.btn_load.pack(side="left", padx=(0, 6))
        self.btn_rec = ttk.Button(src, text="\u23fa Record", command=self._toggle_record)
        self.btn_rec.pack(side="left", padx=(0, 6))
        self.btn_play_orig = ttk.Button(src, text="\u25b6 Play Original",
                                         command=self._play_original, state="disabled")
        self.btn_play_orig.pack(side="left", padx=(0, 6))
        self.btn_play_proc = ttk.Button(src, text="\u25b6 Play Processed",
                                         command=self._play_processed, state="disabled")
        self.btn_play_proc.pack(side="left", padx=(0, 6))
        self.btn_stop = ttk.Button(src, text="\u23f9 Stop", command=self._stop_playback)
        self.btn_stop.pack(side="left", padx=(0, 6))
        self.btn_export = ttk.Button(src, text="\U0001f4be Export", command=self._export,
                                      state="disabled")
        self.btn_export.pack(side="right")
        self.lbl_info = ttk.Label(src, text="No audio loaded")
        self.lbl_info.pack(side="right", padx=10)

        # --- Parameters row ---
        param = ttk.LabelFrame(ctrl, text="  Wavelet Parameters  ", padding=8)
        param.pack(fill="x", pady=(0, 6))

        ts = ttk.Frame(param); ts.pack(fill="x", pady=2)
        ttk.Label(ts, text="Time Stretch:", width=14, anchor="e").pack(side="left")
        self.stretch_var = tk.DoubleVar(value=1.0)
        ttk.Scale(ts, from_=0.25, to=4.0, variable=self.stretch_var,
                  orient="horizontal",
                  command=self._on_stretch_change).pack(side="left", fill="x",
                                                        expand=True, padx=6)
        self.stretch_lbl = ttk.Label(ts, text="1.00x", width=7)
        self.stretch_lbl.pack(side="left")
        ttk.Button(ts, text="Reset", width=6,
                    command=self._reset_stretch).pack(side="left", padx=(4,0))

        ps = ttk.Frame(param); ps.pack(fill="x", pady=2)
        ttk.Label(ps, text="Pitch Shift:", width=14, anchor="e").pack(side="left")
        self.pitch_var = tk.DoubleVar(value=0.0)
        ttk.Scale(ps, from_=-12.0, to=12.0, variable=self.pitch_var,
                  orient="horizontal",
                  command=self._on_pitch_change).pack(side="left", fill="x",
                                                      expand=True, padx=6)
        self.pitch_lbl = ttk.Label(ps, text="0 st", width=7)
        self.pitch_lbl.pack(side="left")
        ttk.Button(ps, text="Reset", width=6,
                    command=self._reset_pitch).pack(side="left", padx=(4,0))

        vf = ttk.Frame(param); vf.pack(fill="x", pady=2)
        ttk.Label(vf, text="Voices:", width=14, anchor="e").pack(side="left")
        self.voices_var = tk.StringVar(value="64")
        ttk.Combobox(vf, textvariable=self.voices_var, width=6,
                      values=["24","36","48","64","96"],
                      state="readonly").pack(side="left", padx=6)
        ttk.Label(vf, text="Freq range:", width=10, anchor="e").pack(side="left", padx=(16,0))
        self.fmin_var = tk.StringVar(value="20")
        ttk.Combobox(vf, textvariable=self.fmin_var, width=6,
                      values=["20","30","50","80","100"],
                      state="readonly").pack(side="left", padx=3)
        ttk.Label(vf, text="-").pack(side="left")
        self.fmax_var = tk.StringVar(value="20000")
        ttk.Combobox(vf, textvariable=self.fmax_var, width=6,
                      values=["8000","12000","16000","20000"],
                      state="readonly").pack(side="left", padx=3)
        ttk.Label(vf, text="Hz").pack(side="left")
        self.btn_process = ttk.Button(vf, text="\u2699 Process", style="Accent.TButton",
                                       command=self._process, state="disabled")
        self.btn_process.pack(side="right")

        # --- Scalograms ---
        fig_frame = ttk.Frame(self.root)
        fig_frame.pack(fill="both", expand=True, padx=12, pady=(0, 4))
        self.fig = Figure(figsize=(11, 4.4), dpi=100, facecolor=self._bg)
        self.ax_orig = self.fig.add_subplot(1, 2, 1)
        self.ax_proc = self.fig.add_subplot(1, 2, 2)
        for ax, title in [(self.ax_orig, "Original Scalogram"),
                           (self.ax_proc, "Processed Scalogram")]:
            ax.set_facecolor(self._surface)
            ax.set_title(title, color=self._fg, fontsize=11, fontweight="bold")
            ax.set_xlabel("Time (s)", color=self._muted, fontsize=9)
            ax.set_ylabel("Frequency (Hz)", color=self._muted, fontsize=9)
            ax.tick_params(colors=self._muted, labelsize=8)
        self.fig.tight_layout(pad=2.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()

        # --- Status bar ---
        self.status_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.status_var, style="Status.TLabel",
                  anchor="w", padding=(10, 4)).pack(fill="x", side="bottom")

    # -- Slider callbacks --

    def _on_stretch_change(self, _=None):
        self.stretch_lbl.config(text=f"{self.stretch_var.get():.2f}x")

    def _reset_stretch(self):
        self.stretch_var.set(1.0)
        self.stretch_lbl.config(text="1.00x")

    def _on_pitch_change(self, _=None):
        self.pitch_lbl.config(text=f"{self.pitch_var.get():+.1f} st")

    def _reset_pitch(self):
        self.pitch_var.set(0.0)
        self.pitch_lbl.config(text="0 st")

    def _update_status(self, msg):
        self.status_var.set(msg)
        self.root.update_idletasks()

    # -- Load / Record --

    def _load_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("WAV files", "*.wav"),
                       ("All audio", "*.wav *.flac *.ogg *.mp3")])
        if not path:
            return
        try:
            data, sr = sf.read(path, dtype="float64")
            if data.ndim > 1:
                data = data.mean(axis=1)
            self.audio_orig = data
            self.sr = sr
            self.audio_proc = None
            self._enable_controls()
            self._draw_scalogram(self.ax_orig, data, sr, "Original Scalogram")
            self._clear_axis(self.ax_proc, "Processed Scalogram")
            self.canvas.draw()
            dur = len(data) / sr
            self.lbl_info.config(text=f"{Path(path).name} | {sr} Hz | {dur:.1f}s")
            self._update_status(f"Loaded: {Path(path).name}  ({dur:.1f}s, {sr} Hz)")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load file:\n{e}")

    def _toggle_record(self):
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        self.is_recording = True
        self.rec_frames = []
        self.btn_rec.config(text="\u23f9 Stop Rec")
        self._update_status("Recording... (press Stop Rec to finish)")
        try:
            self.rec_stream = sd.InputStream(
                samplerate=self.SR, channels=1, dtype="float32",
                callback=lambda ind, f, ti, st: self.rec_frames.append(
                    ind[:, 0].copy()),
                blocksize=1024)
            self.rec_stream.start()
        except Exception as e:
            self.is_recording = False
            self.btn_rec.config(text="\u23fa Record")
            messagebox.showerror("Error", f"Could not open microphone:\n{e}")

    def _stop_recording(self):
        self.is_recording = False
        self.btn_rec.config(text="\u23fa Record")
        try:
            self.rec_stream.stop()
            self.rec_stream.close()
        except Exception:
            pass
        if self.rec_frames:
            self.audio_orig = np.concatenate(self.rec_frames).astype(np.float64)
            self.sr = self.SR
            self.audio_proc = None
            self._enable_controls()
            self._draw_scalogram(self.ax_orig, self.audio_orig, self.sr,
                                  "Original Scalogram")
            self._clear_axis(self.ax_proc, "Processed Scalogram")
            self.canvas.draw()
            dur = len(self.audio_orig) / self.sr
            self.lbl_info.config(text=f"Recording | {self.sr} Hz | {dur:.1f}s")
            self._update_status(f"Recorded {dur:.1f}s of audio.")
        else:
            self._update_status("No audio captured.")

    # -- Playback (callback-based for Windows reliability) --

    def _play_audio(self, data, sr):
        self._stop_playback()
        try:
            buf = np.ascontiguousarray(data, dtype=np.float32).ravel()
            np.clip(buf, -1.0, 1.0, out=buf)
            if len(buf) == 0:
                self._update_status("Nothing to play (empty buffer).")
                return

            self._playback_pos = 0
            self._playback_buf = buf

            def _cb(outdata, frames, time_info, status):
                s = self._playback_pos
                e = s + frames
                chunk = self._playback_buf[s:e]
                if len(chunk) < frames:
                    outdata[:len(chunk), 0] = chunk
                    outdata[len(chunk):, 0] = 0.0
                    self._playback_pos = len(self._playback_buf)
                    raise sd.CallbackStop
                else:
                    outdata[:, 0] = chunk
                    self._playback_pos = e

            self._play_stream = sd.OutputStream(
                samplerate=sr, channels=1, dtype="float32",
                callback=_cb, blocksize=1024,
                finished_callback=lambda: (self.root.after(
                    0, lambda: self._update_status("Playback finished.")), None)[-1])
            self._play_stream.start()
            self._update_status(f"Playing... ({len(buf)/sr:.1f}s)")
        except Exception as e:
            messagebox.showerror("Error", f"Playback failed:\n{e}")

    def _play_original(self):
        if self.audio_orig is not None:
            self._play_audio(self.audio_orig, self.sr)

    def _play_processed(self):
        if self.audio_proc is not None:
            self._play_audio(self.audio_proc, self.sr)

    def _stop_playback(self):
        if hasattr(self, "_play_stream") and self._play_stream is not None:
            try:
                self._play_stream.stop()
                self._play_stream.close()
            except Exception:
                pass
            self._play_stream = None
        sd.stop()
        self._update_status("Stopped.")

    # -- Processing --

    def _process(self):
        if self.audio_orig is None or self.processing:
            return
        self.processing = True
        self.btn_process.config(state="disabled", text="\u23f3 Processing...")
        self._update_status("Processing with wavelet transform...")

        stretch = self.stretch_var.get()
        semitones = self.pitch_var.get()
        n_voices = int(self.voices_var.get())
        fmin = float(self.fmin_var.get())
        fmax = float(self.fmax_var.get())

        def worker():
            try:
                data = self.audio_orig.copy()

                if abs(stretch - 1.0) > 1e-6:
                    self.root.after(0, lambda: self._update_status(
                        "Time-stretching via CWT..."))
                    data = wavelet_time_stretch(data, stretch, self.sr,
                                                n_voices, fmin, fmax)

                if abs(semitones) > 1e-6:
                    self.root.after(0, lambda: self._update_status(
                        "Pitch-shifting via CWT..."))
                    cur_len = len(data)
                    data = wavelet_pitch_shift(data, semitones, self.sr,
                                               n_voices, fmin, fmax,
                                               target_length=cur_len)

                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                peak = np.max(np.abs(data))
                if peak > 1e-6:
                    data = data / peak * 0.95

                self.audio_proc = data
                self.root.after(0, self._on_process_done)
            except Exception as e:
                self.root.after(0, lambda: self._on_process_error(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_process_done(self):
        self.processing = False
        self.btn_process.config(state="normal", text="\u2699 Process")
        self.btn_play_proc.config(state="normal")
        self.btn_export.config(state="normal")
        self._draw_scalogram(self.ax_proc, self.audio_proc, self.sr,
                              "Processed Scalogram")
        self.canvas.draw()
        dur = len(self.audio_proc) / self.sr
        self._update_status(
            f"Done - processed audio is {dur:.1f}s.  Ready for playback or export.")

    def _on_process_error(self, msg):
        self.processing = False
        self.btn_process.config(state="normal", text="\u2699 Process")
        messagebox.showerror("Processing Error", msg)
        self._update_status("Processing failed.")

    # -- Export --

    def _export(self):
        if self.audio_proc is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".wav",
                                             filetypes=[("WAV", "*.wav")])
        if path:
            sf.write(path, self.audio_proc.astype(np.float32), self.sr)
            self._update_status(f"Exported to {Path(path).name}")

    # -- Scalogram drawing --

    def _draw_scalogram(self, ax, data, sr, title):
        ax.clear()
        ax.set_facecolor(self._surface)
        ax.set_title(title, color=self._fg, fontsize=11, fontweight="bold")

        n_disp = 48
        fmin_d = 50.0
        fmax_d = min(sr / 2 * 0.95, 16000.0)
        omega0 = 6.0
        scales = make_scales(sr, n_disp, fmin_d, fmax_d, omega0)
        freqs = scale_to_freq(scales, sr, omega0)

        max_display = sr * 10
        if len(data) > max_display:
            ratio = max_display / len(data)
            display_data = resample(data, max_display)
            display_sr = int(sr * ratio)
            scales_d = make_scales(display_sr, n_disp, fmin_d, fmax_d, omega0)
            W = cwt_morlet(display_data, scales_d, omega0)
        else:
            W = cwt_morlet(data, scales, omega0)

        S_db = 20 * np.log10(np.abs(W) + 1e-10)
        vmax = S_db.max()
        vmin = max(vmax - 80, S_db.min())

        extent = [0, len(data) / sr, 0, len(freqs)]
        ax.imshow(S_db, aspect="auto", origin="lower", extent=extent,
                  cmap="inferno", vmin=vmin, vmax=vmax, interpolation="bilinear")

        n_ticks = 6
        tick_pos = np.linspace(0, len(freqs), n_ticks)
        tick_labels = [f"{int(f)}" for f in np.geomspace(fmin_d, fmax_d, n_ticks)]
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_labels)
        ax.set_xlabel("Time (s)", color=self._muted, fontsize=9)
        ax.set_ylabel("Frequency (Hz)", color=self._muted, fontsize=9)
        ax.tick_params(colors=self._muted, labelsize=8)

    def _clear_axis(self, ax, title):
        ax.clear()
        ax.set_facecolor(self._surface)
        ax.set_title(title, color=self._fg, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)", color=self._muted, fontsize=9)
        ax.set_ylabel("Frequency (Hz)", color=self._muted, fontsize=9)
        ax.tick_params(colors=self._muted, labelsize=8)
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                color="#3b4261", fontsize=14, transform=ax.transAxes)

    def _enable_controls(self):
        self.btn_play_orig.config(state="normal")
        self.btn_process.config(state="normal")
        self.btn_play_proc.config(state="disabled")
        self.btn_export.config(state="disabled")


# ----------------------------------------------
#  Entry point
# ----------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = WaveletProcessorApp(root)
    root.mainloop()
