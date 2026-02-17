"""
Phase Vocoder — Windows 11 Desktop Application
================================================
Features:
  • Record audio from microphone or load WAV files
  • Phase-vocoder-based time stretching (0.25x – 4.0x)
  • Phase-vocoder-based pitch shifting (−12 to +12 semitones)
  • Real-time before/after spectrogram display
  • Playback of original and processed audio
  • Export processed audio to WAV

Requirements (install via pip):
  pip install numpy scipy sounddevice soundfile matplotlib
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
    sys.exit("Missing dependency: pip install sounddevice")
try:
    import soundfile as sf
except ImportError:
    sys.exit("Missing dependency: pip install soundfile")
try:
    from scipy.signal import get_window, resample
except ImportError:
    sys.exit("Missing dependency: pip install scipy")
try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ImportError:
    sys.exit("Missing dependency: pip install matplotlib")


# ──────────────────────────────────────────────
#  Phase Vocoder Core
# ──────────────────────────────────────────────

def stft(x: np.ndarray, fft_size: int = 2048, hop: int = 512) -> np.ndarray:
    """Short-Time Fourier Transform."""
    win = get_window("hann", fft_size, fftbins=True)
    # Pad signal so we get complete frames
    n_frames = 1 + (len(x) - fft_size) // hop
    if n_frames < 1:
        x = np.pad(x, (0, fft_size - len(x)))
        n_frames = 1
    out = np.zeros((fft_size // 2 + 1, n_frames), dtype=np.complex128)
    for i in range(n_frames):
        seg = x[i * hop : i * hop + fft_size] * win
        out[:, i] = np.fft.rfft(seg)
    return out


def istft(X: np.ndarray, fft_size: int = 2048, hop: int = 512) -> np.ndarray:
    """Inverse STFT with overlap-add."""
    win = get_window("hann", fft_size, fftbins=True)
    n_frames = X.shape[1]
    length = fft_size + (n_frames - 1) * hop
    out = np.zeros(length)
    win_sum = np.zeros(length)
    for i in range(n_frames):
        frame = np.fft.irfft(X[:, i], n=fft_size) * win
        start = i * hop
        out[start : start + fft_size] += frame
        win_sum[start : start + fft_size] += win ** 2
    # Normalise by window sum (avoid divide-by-zero)
    nz = win_sum > 1e-8
    out[nz] /= win_sum[nz]
    # Trim leading/trailing regions where overlap is too sparse for good
    # reconstruction — these produce near-silence or edge pops.
    good = np.where(win_sum > 0.1)[0]
    if len(good) > 0:
        out = out[good[0] : good[-1] + 1]
    return out


def phase_vocoder_stretch(x: np.ndarray, stretch_factor: float,
                          fft_size: int = 2048, hop: int = 512) -> np.ndarray:
    """
    Time-stretch a mono signal by *stretch_factor* using phase vocoder.
    stretch_factor > 1 → slower / longer
    stretch_factor < 1 → faster / shorter
    """
    if abs(stretch_factor - 1.0) < 1e-6:
        return x.copy()

    X = stft(x, fft_size, hop)
    n_bins, n_frames = X.shape
    hop_out = int(round(hop * stretch_factor))

    # Expected phase advance per bin per hop
    omega = 2.0 * np.pi * np.arange(n_bins) * hop / fft_size

    # Interpolation positions in the original spectrogram
    time_steps = np.arange(0, n_frames - 1, 1.0 / stretch_factor)
    n_out = len(time_steps)

    Y = np.zeros((n_bins, n_out), dtype=np.complex128)
    phase_acc = np.angle(X[:, 0])
    Y[:, 0] = np.abs(X[:, 0]) * np.exp(1j * phase_acc)

    for t in range(1, n_out):
        # Surrounding frame indices
        idx = time_steps[t]
        i0 = int(np.floor(idx))
        i1 = min(i0 + 1, n_frames - 1)
        frac = idx - i0

        # Interpolate magnitude
        mag = (1 - frac) * np.abs(X[:, i0]) + frac * np.abs(X[:, i1])

        # Phase advance
        dp = np.angle(X[:, i1]) - np.angle(X[:, i0]) - omega
        dp = dp - 2.0 * np.pi * np.round(dp / (2.0 * np.pi))  # wrap
        inst_freq = omega + dp

        phase_acc += inst_freq * stretch_factor
        Y[:, t] = mag * np.exp(1j * phase_acc)

    return istft(Y, fft_size, hop_out)


def pitch_shift(x: np.ndarray, semitones: float, sr: int,
                fft_size: int = 2048, hop: int = 512,
                target_length: int | None = None) -> np.ndarray:
    """Shift pitch by *semitones* (positive = up) via phase vocoder + resampling.

    If *target_length* is given the output is resampled to that many samples;
    otherwise it is resampled to ``len(x)`` (preserving original duration).
    """
    if abs(semitones) < 1e-6:
        return x.copy()
    factor = 2.0 ** (semitones / 12.0)
    # Stretch then resample to change pitch without changing duration
    stretched = phase_vocoder_stretch(x, factor, fft_size, hop)
    tgt = target_length if target_length is not None else len(x)
    shifted = resample(stretched, tgt)
    return shifted


# ──────────────────────────────────────────────
#  GUI Application
# ──────────────────────────────────────────────

class PhaseVocoderApp:
    SR = 44100  # sample rate
    REC_MAX = 30  # max recording seconds

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Phase Vocoder")
        self.root.minsize(1100, 750)
        self.root.configure(bg="#1e1e2e")

        # State
        self.audio_orig: np.ndarray | None = None
        self.audio_proc: np.ndarray | None = None
        self.sr = self.SR
        self.is_recording = False
        self.rec_frames: list[np.ndarray] = []
        self.playback_stream = None
        self.processing = False

        # Style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#1e1e2e")
        style.configure("TLabel", background="#1e1e2e", foreground="#cdd6f4",
                         font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 13, "bold"),
                         foreground="#89b4fa", background="#1e1e2e")
        style.configure("TButton", font=("Segoe UI", 10), padding=6)
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("TScale", background="#1e1e2e")
        style.configure("Status.TLabel", background="#181825", foreground="#a6adc8",
                         font=("Segoe UI", 9))

        self._build_ui()
        self._update_status("Ready — load a file or record audio to begin.")

    # ── UI construction ──────────────────────

    def _build_ui(self):
        # Top controls
        ctrl = ttk.Frame(self.root)
        ctrl.pack(fill="x", padx=12, pady=(10, 4))

        # --- Row 1: Source ---
        src_frame = ttk.LabelFrame(ctrl, text="  Source  ", padding=8)
        src_frame.pack(fill="x", pady=(0, 6))

        self.btn_load = ttk.Button(src_frame, text="📂 Load WAV", command=self._load_file)
        self.btn_load.pack(side="left", padx=(0, 6))

        self.btn_rec = ttk.Button(src_frame, text="⏺ Record", command=self._toggle_record)
        self.btn_rec.pack(side="left", padx=(0, 6))

        self.btn_play_orig = ttk.Button(src_frame, text="▶ Play Original",
                                         command=self._play_original, state="disabled")
        self.btn_play_orig.pack(side="left", padx=(0, 6))

        self.btn_play_proc = ttk.Button(src_frame, text="▶ Play Processed",
                                         command=self._play_processed, state="disabled")
        self.btn_play_proc.pack(side="left", padx=(0, 6))

        self.btn_stop = ttk.Button(src_frame, text="⏹ Stop", command=self._stop_playback)
        self.btn_stop.pack(side="left", padx=(0, 6))

        self.btn_export = ttk.Button(src_frame, text="💾 Export", command=self._export,
                                      state="disabled")
        self.btn_export.pack(side="right")

        self.lbl_info = ttk.Label(src_frame, text="No audio loaded", style="TLabel")
        self.lbl_info.pack(side="right", padx=10)

        # --- Row 2: Parameters ---
        param_frame = ttk.LabelFrame(ctrl, text="  Phase Vocoder Parameters  ", padding=8)
        param_frame.pack(fill="x", pady=(0, 6))

        # Time stretch
        ts_frame = ttk.Frame(param_frame)
        ts_frame.pack(fill="x", pady=2)
        ttk.Label(ts_frame, text="Time Stretch:", width=14, anchor="e").pack(side="left")
        self.stretch_var = tk.DoubleVar(value=1.0)
        self.stretch_scale = ttk.Scale(ts_frame, from_=0.25, to=4.0,
                                        variable=self.stretch_var, orient="horizontal",
                                        command=self._on_stretch_change)
        self.stretch_scale.pack(side="left", fill="x", expand=True, padx=6)
        self.stretch_lbl = ttk.Label(ts_frame, text="1.00x", width=7)
        self.stretch_lbl.pack(side="left")
        ttk.Button(ts_frame, text="Reset", width=6,
                    command=self._reset_stretch).pack(side="left", padx=(4, 0))

        # Pitch shift
        ps_frame = ttk.Frame(param_frame)
        ps_frame.pack(fill="x", pady=2)
        ttk.Label(ps_frame, text="Pitch Shift:", width=14, anchor="e").pack(side="left")
        self.pitch_var = tk.DoubleVar(value=0.0)
        self.pitch_scale = ttk.Scale(ps_frame, from_=-12.0, to=12.0,
                                      variable=self.pitch_var, orient="horizontal",
                                      command=self._on_pitch_change)
        self.pitch_scale.pack(side="left", fill="x", expand=True, padx=6)
        self.pitch_lbl = ttk.Label(ps_frame, text="0 st", width=7)
        self.pitch_lbl.pack(side="left")
        ttk.Button(ps_frame, text="Reset", width=6,
                    command=self._reset_pitch).pack(side="left", padx=(4, 0))

        # FFT size
        fft_frame = ttk.Frame(param_frame)
        fft_frame.pack(fill="x", pady=2)
        ttk.Label(fft_frame, text="FFT Size:", width=14, anchor="e").pack(side="left")
        self.fft_var = tk.StringVar(value="2048")
        fft_combo = ttk.Combobox(fft_frame, textvariable=self.fft_var, width=8,
                                  values=["512", "1024", "2048", "4096", "8192"],
                                  state="readonly")
        fft_combo.pack(side="left", padx=6)

        ttk.Label(fft_frame, text="Hop Size:", width=10, anchor="e").pack(side="left", padx=(16, 0))
        self.hop_var = tk.StringVar(value="512")
        hop_combo = ttk.Combobox(fft_frame, textvariable=self.hop_var, width=8,
                                  values=["128", "256", "512", "1024"],
                                  state="readonly")
        hop_combo.pack(side="left", padx=6)

        self.btn_process = ttk.Button(fft_frame, text="⚙ Process", style="Accent.TButton",
                                       command=self._process, state="disabled")
        self.btn_process.pack(side="right")

        # --- Spectrograms ---
        fig_frame = ttk.Frame(self.root)
        fig_frame.pack(fill="both", expand=True, padx=12, pady=(0, 4))

        self.fig = Figure(figsize=(11, 4.2), dpi=100, facecolor="#1e1e2e")
        self.ax_orig = self.fig.add_subplot(1, 2, 1)
        self.ax_proc = self.fig.add_subplot(1, 2, 2)
        for ax, title in [(self.ax_orig, "Original"), (self.ax_proc, "Processed")]:
            ax.set_facecolor("#181825")
            ax.set_title(title, color="#cdd6f4", fontsize=11, fontweight="bold")
            ax.set_xlabel("Time (s)", color="#a6adc8", fontsize=9)
            ax.set_ylabel("Frequency (Hz)", color="#a6adc8", fontsize=9)
            ax.tick_params(colors="#6c7086", labelsize=8)
        self.fig.tight_layout(pad=2.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()

        # --- Status bar ---
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self.root, textvariable=self.status_var, style="Status.TLabel",
                                anchor="w", padding=(10, 4))
        status_bar.pack(fill="x", side="bottom")

    # ── Slider callbacks ─────────────────────

    def _on_stretch_change(self, _=None):
        v = self.stretch_var.get()
        self.stretch_lbl.config(text=f"{v:.2f}x")

    def _reset_stretch(self):
        self.stretch_var.set(1.0)
        self.stretch_lbl.config(text="1.00x")

    def _on_pitch_change(self, _=None):
        v = self.pitch_var.get()
        self.pitch_lbl.config(text=f"{v:+.1f} st")

    def _reset_pitch(self):
        self.pitch_var.set(0.0)
        self.pitch_lbl.config(text="0 st")

    # ── Status ───────────────────────────────

    def _update_status(self, msg: str):
        self.status_var.set(msg)
        self.root.update_idletasks()

    # ── Load / Record ────────────────────────

    def _load_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("WAV files", "*.wav"), ("All audio", "*.wav *.flac *.ogg *.mp3")])
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
            self._draw_spectrogram(self.ax_orig, data, sr, "Original")
            self._clear_axis(self.ax_proc, "Processed")
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
        self.btn_rec.config(text="⏹ Stop Rec")
        self._update_status("Recording… (press Stop Rec to finish)")

        def callback(indata, frames, time_info, status):
            self.rec_frames.append(indata[:, 0].copy())

        try:
            self.rec_stream = sd.InputStream(samplerate=self.SR, channels=1,
                                              dtype="float32", callback=callback,
                                              blocksize=1024)
            self.rec_stream.start()
        except Exception as e:
            self.is_recording = False
            self.btn_rec.config(text="⏺ Record")
            messagebox.showerror("Error", f"Could not open microphone:\n{e}")

    def _stop_recording(self):
        self.is_recording = False
        self.btn_rec.config(text="⏺ Record")
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
            self._draw_spectrogram(self.ax_orig, self.audio_orig, self.sr, "Original")
            self._clear_axis(self.ax_proc, "Processed")
            self.canvas.draw()
            dur = len(self.audio_orig) / self.sr
            self.lbl_info.config(text=f"Recording | {self.sr} Hz | {dur:.1f}s")
            self._update_status(f"Recorded {dur:.1f}s of audio.")
        else:
            self._update_status("No audio captured.")

    # ── Playback ─────────────────────────────

    def _play_audio(self, data: np.ndarray, sr: int):
        self._stop_playback()
        try:
            # Ensure contiguous float32 1-D array, clipped to valid range
            buf = np.ascontiguousarray(data, dtype=np.float32).ravel()
            np.clip(buf, -1.0, 1.0, out=buf)

            if len(buf) == 0:
                self._update_status("Nothing to play (empty buffer).")
                return

            # Use an OutputStream so we control the lifecycle and avoid
            # the silent-fail that sd.play() can exhibit on some Windows
            # WASAPI / DirectSound backends when the default device sample
            # rate doesn't match.
            self._playback_pos = 0
            self._playback_buf = buf

            def _callback(outdata, frames, time_info, status):
                start = self._playback_pos
                end = start + frames
                chunk = self._playback_buf[start:end]
                if len(chunk) < frames:
                    outdata[:len(chunk), 0] = chunk
                    outdata[len(chunk):, 0] = 0.0
                    self._playback_pos = len(self._playback_buf)
                    raise sd.CallbackStop
                else:
                    outdata[:, 0] = chunk
                    self._playback_pos = end

            self._play_stream = sd.OutputStream(
                samplerate=sr,
                channels=1,
                dtype="float32",
                callback=_callback,
                blocksize=1024,
                finished_callback=lambda: (self.root.after(
                    0, lambda: self._update_status("Playback finished.")), None)[-1],
            )
            self._play_stream.start()
            dur = len(buf) / sr
            self._update_status(f"Playing… ({dur:.1f}s)")
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
        sd.stop()  # belt-and-suspenders
        self._update_status("Stopped.")

    # ── Processing ───────────────────────────

    def _process(self):
        if self.audio_orig is None or self.processing:
            return
        self.processing = True
        self.btn_process.config(state="disabled", text="⏳ Processing…")
        self._update_status("Processing with phase vocoder…")

        stretch = self.stretch_var.get()
        semitones = self.pitch_var.get()
        fft_size = int(self.fft_var.get())
        hop = int(self.hop_var.get())

        def worker():
            try:
                data = self.audio_orig.copy()
                orig_len = len(data)

                # Apply time stretching
                if abs(stretch - 1.0) > 1e-6:
                    data = phase_vocoder_stretch(data, stretch, fft_size, hop)

                # Apply pitch shifting (resample back to current length so
                # duration set by the stretch slider is preserved)
                if abs(semitones) > 1e-6:
                    cur_len = len(data)
                    data = pitch_shift(data, semitones, self.sr, fft_size, hop,
                                       target_length=cur_len)

                # Ensure finite values only
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

                # Normalize to avoid clipping
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
        self.btn_process.config(state="normal", text="⚙ Process")
        self.btn_play_proc.config(state="normal")
        self.btn_export.config(state="normal")
        self._draw_spectrogram(self.ax_proc, self.audio_proc, self.sr, "Processed")
        self.canvas.draw()
        dur = len(self.audio_proc) / self.sr
        self._update_status(f"Done — processed audio is {dur:.1f}s. Ready for playback or export.")

    def _on_process_error(self, msg: str):
        self.processing = False
        self.btn_process.config(state="normal", text="⚙ Process")
        messagebox.showerror("Processing Error", msg)
        self._update_status("Processing failed.")

    # ── Export ───────────────────────────────

    def _export(self):
        if self.audio_proc is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".wav",
                                             filetypes=[("WAV", "*.wav")])
        if path:
            sf.write(path, self.audio_proc.astype(np.float32), self.sr)
            self._update_status(f"Exported to {Path(path).name}")

    # ── Spectrogram drawing ──────────────────

    def _draw_spectrogram(self, ax, data: np.ndarray, sr: int, title: str):
        ax.clear()
        ax.set_facecolor("#181825")
        ax.set_title(title, color="#cdd6f4", fontsize=11, fontweight="bold")

        fft_n = min(2048, len(data))
        hop_n = fft_n // 4

        S = stft(data, fft_n, hop_n)
        S_db = 20 * np.log10(np.abs(S) + 1e-10)
        vmax = S_db.max()
        vmin = max(vmax - 80, S_db.min())

        extent = [0, len(data) / sr, 0, sr / 2]
        ax.imshow(S_db, aspect="auto", origin="lower", extent=extent,
                  cmap="magma", vmin=vmin, vmax=vmax, interpolation="bilinear")
        ax.set_xlabel("Time (s)", color="#a6adc8", fontsize=9)
        ax.set_ylabel("Frequency (Hz)", color="#a6adc8", fontsize=9)
        ax.tick_params(colors="#6c7086", labelsize=8)
        # Limit y-axis to useful range
        ax.set_ylim(0, min(sr / 2, 8000))

    def _clear_axis(self, ax, title: str):
        ax.clear()
        ax.set_facecolor("#181825")
        ax.set_title(title, color="#cdd6f4", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)", color="#a6adc8", fontsize=9)
        ax.set_ylabel("Frequency (Hz)", color="#a6adc8", fontsize=9)
        ax.tick_params(colors="#6c7086", labelsize=8)
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                color="#45475a", fontsize=14, transform=ax.transAxes)

    # ── Helpers ──────────────────────────────

    def _enable_controls(self):
        self.btn_play_orig.config(state="normal")
        self.btn_process.config(state="normal")
        self.btn_play_proc.config(state="disabled")
        self.btn_export.config(state="disabled")


# ──────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app = PhaseVocoderApp(root)
    root.mainloop()
