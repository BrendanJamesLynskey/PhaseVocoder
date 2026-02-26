"""
Constant-Q Transform Audio Processor — Windows 11 Desktop Application
======================================================================
Time-stretching and pitch-shifting using the Constant-Q Transform (CQT),
which analyses the signal on a logarithmic frequency grid with
geometrically-varying window lengths — short windows at high frequencies,
long windows at low frequencies — giving musically-meaningful frequency
resolution (constant ratio between adjacent bins, e.g. 24 bins/octave).

The CQT sits between the STFT (uniform bins, fixed window) and the CWT
(continuous wavelet at arbitrary scales): it uses discrete, log-spaced
frequency bins computed efficiently via FFT-based convolution with
per-bin windowed complex exponentials.

Features:
  * Record audio from microphone or load WAV files
  * CQT-based time stretching (0.25x – 4.0x)
  * CQT-based pitch shifting (−12 to +12 semitones)
  * Before/after CQ-spectrogram display (log-frequency axis)
  * Playback of original and processed audio
  * Export processed audio to WAV

Requirements (install via pip):
  pip install numpy scipy sounddevice soundfile matplotlib

Usage:
  python cqt_processor.py
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
    from scipy.signal import fftconvolve, resample, get_window
except ImportError:
    sys.exit("Missing dependency:  pip install scipy")
try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ImportError:
    sys.exit("Missing dependency:  pip install matplotlib")


# ──────────────────────────────────────────────
#  Constant-Q Transform Core
# ──────────────────────────────────────────────

def design_cqt_kernels(sr: int, fmin: float = 32.7, n_bins: int = 84,
                       bins_per_octave: int = 12, q_factor: float = 1.0):
    """
    Pre-compute the CQT analysis kernels (windowed complex exponentials).

    Each bin k has centre frequency  f_k = fmin * 2^(k / bins_per_octave)
    and window length  N_k = ceil(Q * sr / f_k)  where  Q = q_factor /
    (2^(1/bins_per_octave) − 1).

    Returns
    -------
    kernels : list[np.ndarray]
        Complex analysis kernels, one per frequency bin.
    freqs : np.ndarray
        Centre frequencies in Hz, shape (n_bins,).
    Q : float
        The constant quality factor used.
    """
    Q = q_factor / (2.0 ** (1.0 / bins_per_octave) - 1.0)
    freqs = fmin * 2.0 ** (np.arange(n_bins) / bins_per_octave)

    # Clamp to Nyquist
    nyq = sr / 2.0 * 0.95
    freqs = freqs[freqs <= nyq]
    n_bins = len(freqs)

    kernels = []
    for k in range(n_bins):
        N_k = int(np.ceil(Q * sr / freqs[k]))
        # Ensure odd length for symmetric window
        if N_k % 2 == 0:
            N_k += 1
        win = get_window("hann", N_k, fftbins=False)
        t = np.arange(N_k)
        kernel = (win / N_k) * np.exp(2j * np.pi * freqs[k] * t / sr)
        kernels.append(kernel)

    return kernels, freqs, Q


def cqt(x: np.ndarray, sr: int, hop: int = 512, fmin: float = 32.7,
        n_bins: int = 84, bins_per_octave: int = 12,
        q_factor: float = 1.0):
    """
    Compute the Constant-Q Transform of a mono signal.

    Uses overlap-save convolution with pre-computed per-bin kernels.
    Returns a complex CQ spectrogram of shape (n_bins, n_frames).

    Parameters
    ----------
    x : 1-D float array
    sr : sample rate
    hop : hop size in samples (analysis time step)
    fmin : lowest frequency bin centre (Hz)
    n_bins : number of CQ frequency bins
    bins_per_octave : bins per octave (12 = semitone, 24 = quarter-tone, …)
    q_factor : multiplier on the base Q (1.0 = standard)

    Returns
    -------
    CQ : complex array, shape (n_bins_actual, n_frames)
    freqs : centre frequencies (Hz)
    Q_val : the quality factor used
    """
    kernels, freqs, Q_val = design_cqt_kernels(
        sr, fmin, n_bins, bins_per_octave, q_factor
    )
    actual_bins = len(freqs)
    n_frames = max(1, 1 + (len(x) - 1) // hop)

    CQ = np.zeros((actual_bins, n_frames), dtype=np.complex128)

    for k in range(actual_bins):
        # Full convolution via FFT, then downsample at hop positions
        conv = fftconvolve(x, kernels[k][::-1].conj(), mode="same")
        # Sample at hop positions
        positions = np.arange(n_frames) * hop
        positions = np.clip(positions, 0, len(conv) - 1)
        CQ[k, :] = conv[positions]

    return CQ, freqs, Q_val


def icqt(CQ: np.ndarray, sr: int, hop: int, freqs: np.ndarray,
         Q_val: float, length: int, bins_per_octave: int = 12,
         q_factor: float = 1.0):
    """
    Inverse CQT via overlap-add of windowed complex exponentials.

    For each bin and each frame, a windowed sinusoid at the bin's centre
    frequency is placed at the frame's time position, scaled by the CQ
    coefficient magnitude and phase.  A normalisation pass accounts for
    the overlap density.

    Parameters
    ----------
    CQ : complex array (n_bins, n_frames)
    sr : sample rate
    hop : hop size
    freqs : centre frequencies for each bin
    Q_val : quality factor
    length : desired output signal length in samples
    bins_per_octave, q_factor : same as in cqt()

    Returns
    -------
    y : reconstructed signal, shape (length,)
    """
    n_bins, n_frames = CQ.shape
    y = np.zeros(length)
    win_sum = np.zeros(length)

    for k in range(n_bins):
        N_k = int(np.ceil(Q_val * sr / freqs[k]))
        if N_k % 2 == 0:
            N_k += 1
        half = N_k // 2
        win = get_window("hann", N_k, fftbins=False)
        t = np.arange(N_k)

        for n in range(n_frames):
            centre = n * hop
            start = centre - half
            end = start + N_k

            # Boundaries
            if start < 0 or end > length:
                s_lo = max(0, -start)
                s_hi = min(N_k, length - start)
                if s_lo >= s_hi:
                    continue
                o_start = max(0, start)
                o_end = o_start + (s_hi - s_lo)
            else:
                s_lo = 0
                s_hi = N_k
                o_start = start
                o_end = end

            coeff = CQ[k, n]
            # Synthesis atom: coefficient * conjugate-kernel (windowed sinusoid)
            atom = win[s_lo:s_hi] * np.exp(
                2j * np.pi * freqs[k] * t[s_lo:s_hi] / sr
            )
            y[o_start:o_end] += np.real(coeff * atom)
            win_sum[o_start:o_end] += win[s_lo:s_hi] ** 2

    # Normalise
    nz = win_sum > 1e-8
    y[nz] /= win_sum[nz]
    return y


# ──────────────────────────────────────────────
#  Time Stretch & Pitch Shift via CQT
# ──────────────────────────────────────────────

def _stft_stretch(x: np.ndarray, stretch: float,
                  fft_size: int = 2048, hop_a: int = 512) -> np.ndarray:
    """
    Lightweight STFT phase-vocoder time-stretch.

    Used to stretch the residual signal (energy the CQT does not
    capture).  Unlike ``scipy.signal.resample`` — which changes pitch —
    the phase vocoder preserves pitch while changing duration, preventing
    audible phantom tones from appearing in the output.
    """
    if abs(stretch - 1.0) < 1e-6:
        return x.copy()

    N = fft_size
    Ha = hop_a
    win = get_window("hann", N, fftbins=False)

    n_frames = 1 + (len(x) - N) // Ha
    if n_frames < 2:
        target = max(1, int(round(len(x) * stretch)))
        return resample(x, target)

    xp = np.pad(x, (0, max(0, (n_frames - 1) * Ha + N - len(x))))

    # Forward STFT
    stft = np.zeros((N // 2 + 1, n_frames), dtype=np.complex128)
    for m in range(n_frames):
        stft[:, m] = np.fft.rfft(xp[m * Ha : m * Ha + N] * win)

    # Phase-vocoder synthesis
    Hs = Ha
    n_out_frames = max(1, int(round(n_frames * stretch)))
    omega = 2.0 * np.pi * np.arange(N // 2 + 1) * Ha / N

    out_len = (n_out_frames - 1) * Hs + N
    y = np.zeros(out_len)
    win_sum = np.zeros(out_len)
    phase_acc = np.angle(stft[:, 0])

    for m_out in range(n_out_frames):
        pos = m_out / stretch
        m0 = min(int(pos), n_frames - 1)
        m1 = min(m0 + 1, n_frames - 1)
        frac = pos - int(pos)

        mag = (1.0 - frac) * np.abs(stft[:, m0]) + frac * np.abs(stft[:, m1])

        if m_out > 0:
            dphi = np.angle(stft[:, m1]) - np.angle(stft[:, m0])
            dev = dphi - omega
            dev -= 2.0 * np.pi * np.round(dev / (2.0 * np.pi))
            phase_acc += omega + dev

        frame = np.fft.irfft(mag * np.exp(1j * phase_acc), N) * win
        start = m_out * Hs
        y[start : start + N] += frame
        win_sum[start : start + N] += win ** 2

    nz = win_sum > 1e-8
    y[nz] /= win_sum[nz]

    target_len = max(1, int(round(len(x) * stretch)))
    if len(y) >= target_len:
        return y[:target_len]
    return np.pad(y, (0, target_len - len(y)))


def _compute_inst_freq(CQ: np.ndarray, freqs: np.ndarray,
                       hop: int, sr: int) -> np.ndarray:
    """
    Compute true instantaneous frequency for each CQT bin using
    heterodyne (carrier-relative) phase unwrapping.

    Standard ``np.unwrap`` fails on CQT coefficients because the phase
    advances by ``2π · f_k · hop / sr`` radians per frame — often many
    full cycles — so the wrapped frame-to-frame difference looks like a
    small step and ``unwrap`` leaves it alone.

    Fix: subtract the expected carrier advance, wrap the residual to
    [−π, π], then add the carrier back.  This recovers the true
    instantaneous frequency even when the per-frame advance exceeds 2π.

    Returns
    -------
    inst_freq : array, shape (n_bins, n_frames)
        Instantaneous frequency in **radians per sample**.
    """
    n_bins, n_frames = CQ.shape
    inst_freq = np.zeros((n_bins, n_frames))

    for k in range(n_bins):
        raw_phase = np.angle(CQ[k, :])
        carrier_advance = 2.0 * np.pi * freqs[k] * hop / sr

        dphi = np.diff(raw_phase)
        residual = dphi - carrier_advance
        residual_wrapped = (residual + np.pi) % (2.0 * np.pi) - np.pi

        true_advance = carrier_advance + residual_wrapped   # rad / frame
        true_if = true_advance / hop                        # rad / sample

        inst_freq[k, 0] = (true_if[0] if len(true_if) > 0
                           else 2.0 * np.pi * freqs[k] / sr)
        inst_freq[k, 1:] = true_if

    return inst_freq


def _phase_lock_inst_freq(inst_freq: np.ndarray,
                          mag: np.ndarray) -> np.ndarray:
    """
    Phase-lock adjacent CQT bins to the nearest spectral peak.

    The CQT's finite frequency resolution spreads a pure tone across
    several bins.  Without correction, each bin synthesises a sinusoid
    at its *own* slightly different frequency, producing audible beating
    / tone-splitting artefacts.

    For every frame, local magnitude peaks are found across bins, and
    every non-peak bin adopts the instantaneous frequency of the nearest
    peak.  This forces leakage sidebands to cohere with the true partial.
    """
    n_bins, n_frames = inst_freq.shape
    locked = inst_freq.copy()

    for n in range(n_frames):
        col = mag[:, n]
        peaks = []
        for k in range(1, n_bins - 1):
            if col[k] > col[k - 1] and col[k] > col[k + 1]:
                peaks.append(k)
        if n_bins > 1:
            if col[0] > col[1]:
                peaks.append(0)
            if col[-1] > col[-2]:
                peaks.append(n_bins - 1)
        if not peaks:
            continue
        peaks_arr = np.array(sorted(peaks))
        for k in range(n_bins):
            nearest = peaks_arr[np.argmin(np.abs(peaks_arr - k))]
            locked[k, n] = inst_freq[nearest, n]

    return locked


def _synth_from_cqt(CQ, inst_freq, frame_times, target_times, n_out):
    """
    Additive synthesis: for each bin, interpolate magnitude and
    instantaneous frequency to the output sample grid and integrate
    phase sample-by-sample.
    """
    n_bins = CQ.shape[0]
    mag = np.abs(CQ)
    y = np.zeros(n_out)
    for k in range(n_bins):
        mag_out = np.interp(target_times, frame_times, mag[k, :])
        if_out  = np.interp(target_times, frame_times, inst_freq[k, :])
        phase   = np.cumsum(if_out)
        phase   = phase - phase[0] + np.angle(CQ[k, 0])
        y += mag_out * np.cos(phase)
    return y


def cqt_time_stretch(x: np.ndarray, stretch: float, sr: int,
                     hop: int = 512, fmin: float = 32.7,
                     n_bins: int = 84, bins_per_octave: int = 12,
                     q_factor: float = 1.0) -> np.ndarray:
    """
    Time-stretch a mono signal by *stretch* using the CQT.

    Algorithm
    ---------
    1. Compute the forward CQT.
    2. Derive true instantaneous frequency per bin via heterodyne
       (carrier-relative) phase unwrapping — essential because the
       raw inter-frame phase advance can exceed many full cycles.
    3. Phase-lock adjacent bins: spectral-leakage sidebands adopt the
       frequency of the nearest magnitude peak so they reinforce rather
       than beat against the true partial.
    4. For each output sample, map back to the corresponding input time,
       interpolate magnitude and instantaneous frequency, and integrate
       to obtain phase.  This sample-level additive resynthesis avoids
       the artefacts of the frame-based iCQT overlap-add.
    5. A least-squares scale factor matches the output level.  The
       residual (energy not captured by the CQT) is always resampled
       and blended back to maintain full-bandwidth fidelity.
    6. The output is RMS-matched to the input to ensure consistent level.

    stretch > 1 → slower / longer
    stretch < 1 → faster / shorter
    """
    if abs(stretch - 1.0) < 1e-6:
        return x.copy()

    # ---- analysis ----
    CQ, freqs, Q_val = cqt(x, sr, hop, fmin, n_bins, bins_per_octave,
                            q_factor)
    n_cq_bins, n_frames = CQ.shape

    inst_freq = _compute_inst_freq(CQ, freqs, hop, sr)
    inst_freq = _phase_lock_inst_freq(inst_freq, np.abs(CQ))

    # ---- time grids ----
    frame_times = np.arange(n_frames, dtype=np.float64) * hop
    n_in  = len(x)
    n_out = max(1, int(round(n_in * stretch)))

    # Output samples mapped back to input time for interpolation
    in_times = np.arange(n_out, dtype=np.float64) / stretch

    # ---- synthesis (stretched) ----
    y = _synth_from_cqt(CQ, inst_freq, frame_times, in_times, n_out)

    # ---- scale factor via unstretched resynthesis ----
    orig_times = np.arange(n_in, dtype=np.float64)
    y_orig = _synth_from_cqt(CQ, inst_freq, frame_times, orig_times, n_in)

    scale_fac = np.dot(x, y_orig) / (np.dot(y_orig, y_orig) + 1e-10)
    y      *= scale_fac
    y_orig *= scale_fac

    # ---- residual blending ----
    # The residual often contains tonal energy that the CQT missed.
    # Naive resampling would shift the residual's pitch by 1/stretch,
    # creating audible phantom tones.  Instead, time-stretch the residual
    # with a lightweight STFT phase vocoder to preserve its pitch.
    residual = x - y_orig
    y += _stft_stretch(residual, stretch)

    # ---- RMS-match output to input ----
    rms_in  = np.sqrt(np.mean(x ** 2))
    rms_out = np.sqrt(np.mean(y ** 2))
    if rms_out > 1e-10:
        y *= rms_in / rms_out

    return y


def cqt_pitch_shift(x: np.ndarray, semitones: float, sr: int,
                    hop: int = 512, fmin: float = 32.7,
                    n_bins: int = 84, bins_per_octave: int = 12,
                    q_factor: float = 1.0,
                    target_length: int | None = None) -> np.ndarray:
    """
    Shift pitch by *semitones* using CQT time-stretch + resampling.

    The signal is first time-stretched by the pitch ratio (preserving
    pitch via the CQT), then resampled back to the target length to
    shift all frequencies.
    """
    if abs(semitones) < 1e-6:
        return x.copy()

    factor = 2.0 ** (semitones / 12.0)
    stretched = cqt_time_stretch(x, factor, sr, hop, fmin, n_bins,
                                 bins_per_octave, q_factor)
    tgt = target_length if target_length is not None else len(x)
    return resample(stretched, tgt)


# ──────────────────────────────────────────────
#  GUI Application
# ──────────────────────────────────────────────

class CQTProcessorApp:
    SR = 44100  # sample rate
    REC_MAX = 30  # max recording seconds

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Constant-Q Transform Audio Processor")
        self.root.minsize(1100, 780)
        self.root.configure(bg="#1a1b26")

        # State
        self.audio_orig: np.ndarray | None = None
        self.audio_proc: np.ndarray | None = None
        self.sr = self.SR
        self.is_recording = False
        self.rec_frames: list[np.ndarray] = []
        self.processing = False

        # Style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame",  background="#1a1b26")
        style.configure("TLabel",  background="#1a1b26", foreground="#c0caf5",
                         font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 13, "bold"),
                         foreground="#7aa2f7", background="#1a1b26")
        style.configure("TButton", font=("Segoe UI", 10), padding=6)
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("TScale",  background="#1a1b26")
        style.configure("Status.TLabel", background="#16161e", foreground="#9aa5ce",
                         font=("Segoe UI", 9))

        self._build_ui()
        self._update_status("Ready — load a file or record audio to begin.")

    # ── UI construction ──────────────────────

    def _build_ui(self):
        ctrl = ttk.Frame(self.root)
        ctrl.pack(fill="x", padx=12, pady=(10, 4))

        # --- Row 1: Source ---
        src_frame = ttk.LabelFrame(ctrl, text="  Source  ", padding=8)
        src_frame.pack(fill="x", pady=(0, 6))

        self.btn_load = ttk.Button(src_frame, text="📂 Load WAV",
                                    command=self._load_file)
        self.btn_load.pack(side="left", padx=(0, 6))

        self.btn_rec = ttk.Button(src_frame, text="⏺ Record",
                                   command=self._toggle_record)
        self.btn_rec.pack(side="left", padx=(0, 6))

        self.btn_play_orig = ttk.Button(src_frame, text="▶ Play Original",
                                         command=self._play_original,
                                         state="disabled")
        self.btn_play_orig.pack(side="left", padx=(0, 6))

        self.btn_play_proc = ttk.Button(src_frame, text="▶ Play Processed",
                                         command=self._play_processed,
                                         state="disabled")
        self.btn_play_proc.pack(side="left", padx=(0, 6))

        self.btn_stop = ttk.Button(src_frame, text="⏹ Stop",
                                    command=self._stop_playback)
        self.btn_stop.pack(side="left", padx=(0, 6))

        self.btn_export = ttk.Button(src_frame, text="💾 Export",
                                      command=self._export, state="disabled")
        self.btn_export.pack(side="right")

        self.lbl_info = ttk.Label(src_frame, text="No audio loaded",
                                   style="TLabel")
        self.lbl_info.pack(side="right", padx=10)

        # --- Row 2: Parameters ---
        param_frame = ttk.LabelFrame(ctrl,
                                      text="  Constant-Q Transform Parameters  ",
                                      padding=8)
        param_frame.pack(fill="x", pady=(0, 6))

        # Time stretch
        ts_frame = ttk.Frame(param_frame)
        ts_frame.pack(fill="x", pady=2)
        ttk.Label(ts_frame, text="Time Stretch:", width=16,
                  anchor="e").pack(side="left")
        self.stretch_var = tk.DoubleVar(value=1.0)
        self.stretch_scale = ttk.Scale(ts_frame, from_=0.25, to=4.0,
                                        variable=self.stretch_var,
                                        orient="horizontal",
                                        command=self._on_stretch_change)
        self.stretch_scale.pack(side="left", fill="x", expand=True, padx=6)
        self.stretch_lbl = ttk.Label(ts_frame, text="1.00x", width=7)
        self.stretch_lbl.pack(side="left")
        ttk.Button(ts_frame, text="Reset", width=6,
                    command=self._reset_stretch).pack(side="left", padx=(4, 0))

        # Pitch shift
        ps_frame = ttk.Frame(param_frame)
        ps_frame.pack(fill="x", pady=2)
        ttk.Label(ps_frame, text="Pitch Shift:", width=16,
                  anchor="e").pack(side="left")
        self.pitch_var = tk.DoubleVar(value=0.0)
        self.pitch_scale = ttk.Scale(ps_frame, from_=-12.0, to=12.0,
                                      variable=self.pitch_var,
                                      orient="horizontal",
                                      command=self._on_pitch_change)
        self.pitch_scale.pack(side="left", fill="x", expand=True, padx=6)
        self.pitch_lbl = ttk.Label(ps_frame, text="0 st", width=7)
        self.pitch_lbl.pack(side="left")
        ttk.Button(ps_frame, text="Reset", width=6,
                    command=self._reset_pitch).pack(side="left", padx=(4, 0))

        # CQT parameters row
        cqt_frame = ttk.Frame(param_frame)
        cqt_frame.pack(fill="x", pady=2)

        ttk.Label(cqt_frame, text="Bins/Octave:", width=16,
                  anchor="e").pack(side="left")
        self.bpo_var = tk.StringVar(value="24")
        bpo_combo = ttk.Combobox(cqt_frame, textvariable=self.bpo_var, width=6,
                                  values=["12", "24", "36", "48"],
                                  state="readonly")
        bpo_combo.pack(side="left", padx=6)

        ttk.Label(cqt_frame, text="Hop Size:", width=10,
                  anchor="e").pack(side="left", padx=(10, 0))
        self.hop_var = tk.StringVar(value="512")
        hop_combo = ttk.Combobox(cqt_frame, textvariable=self.hop_var, width=6,
                                  values=["128", "256", "512", "1024"],
                                  state="readonly")
        hop_combo.pack(side="left", padx=6)

        ttk.Label(cqt_frame, text="f_min (Hz):", width=10,
                  anchor="e").pack(side="left", padx=(10, 0))
        self.fmin_var = tk.StringVar(value="32.7")
        fmin_combo = ttk.Combobox(cqt_frame, textvariable=self.fmin_var, width=7,
                                   values=["16.35", "32.7", "55.0", "65.4",
                                           "130.8"],
                                   state="readonly")
        fmin_combo.pack(side="left", padx=6)

        self.btn_process = ttk.Button(cqt_frame, text="⚙ Process",
                                       style="Accent.TButton",
                                       command=self._process, state="disabled")
        self.btn_process.pack(side="right")

        # --- CQ spectrograms ---
        fig_frame = ttk.Frame(self.root)
        fig_frame.pack(fill="both", expand=True, padx=12, pady=(0, 4))

        self.fig = Figure(figsize=(11, 4.2), dpi=100, facecolor="#1a1b26")
        self.ax_orig = self.fig.add_subplot(1, 2, 1)
        self.ax_proc = self.fig.add_subplot(1, 2, 2)
        for ax, title in [(self.ax_orig, "Original"),
                          (self.ax_proc, "Processed")]:
            ax.set_facecolor("#16161e")
            ax.set_title(title, color="#c0caf5", fontsize=11, fontweight="bold")
            ax.set_xlabel("Time (s)", color="#9aa5ce", fontsize=9)
            ax.set_ylabel("Frequency (Hz)", color="#9aa5ce", fontsize=9)
            ax.tick_params(colors="#565f89", labelsize=8)
        self.fig.tight_layout(pad=2.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()

        # --- Status bar ---
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                                style="Status.TLabel", anchor="w",
                                padding=(10, 4))
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
            self._draw_cq_spectrogram(self.ax_orig, data, sr, "Original")
            self._clear_axis(self.ax_proc, "Processed")
            self.canvas.draw()
            dur = len(data) / sr
            self.lbl_info.config(
                text=f"{Path(path).name} | {sr} Hz | {dur:.1f}s")
            self._update_status(
                f"Loaded: {Path(path).name}  ({dur:.1f}s, {sr} Hz)")
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
            self.rec_stream = sd.InputStream(
                samplerate=self.SR, channels=1, dtype="float32",
                callback=callback, blocksize=1024)
            self.rec_stream.start()
        except Exception as e:
            self.is_recording = False
            self.btn_rec.config(text="⏺ Record")
            messagebox.showerror("Error",
                                 f"Could not open microphone:\n{e}")

    def _stop_recording(self):
        self.is_recording = False
        self.btn_rec.config(text="⏺ Record")
        try:
            self.rec_stream.stop()
            self.rec_stream.close()
        except Exception:
            pass
        if self.rec_frames:
            self.audio_orig = np.concatenate(
                self.rec_frames).astype(np.float64)
            self.sr = self.SR
            self.audio_proc = None
            self._enable_controls()
            self._draw_cq_spectrogram(self.ax_orig, self.audio_orig,
                                       self.sr, "Original")
            self._clear_axis(self.ax_proc, "Processed")
            self.canvas.draw()
            dur = len(self.audio_orig) / self.sr
            self.lbl_info.config(
                text=f"Recording | {self.sr} Hz | {dur:.1f}s")
            self._update_status(f"Recorded {dur:.1f}s of audio.")
        else:
            self._update_status("No audio captured.")

    # ── Playback ─────────────────────────────

    def _play_audio(self, data: np.ndarray, sr: int):
        self._stop_playback()
        try:
            buf = np.ascontiguousarray(data, dtype=np.float32).ravel()
            np.clip(buf, -1.0, 1.0, out=buf)

            if len(buf) == 0:
                self._update_status("Nothing to play (empty buffer).")
                return

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
                samplerate=sr, channels=1, dtype="float32",
                callback=_callback, blocksize=1024,
                finished_callback=lambda: (
                    self.root.after(0, lambda: self._update_status(
                        "Playback finished.")), None)[-1],
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
        sd.stop()
        self._update_status("Stopped.")

    # ── Processing ───────────────────────────

    def _process(self):
        if self.audio_orig is None or self.processing:
            return
        self.processing = True
        self.btn_process.config(state="disabled", text="⏳ Processing…")
        self._update_status("Processing with Constant-Q Transform…")

        stretch = self.stretch_var.get()
        semitones = self.pitch_var.get()
        hop = int(self.hop_var.get())
        bpo = int(self.bpo_var.get())
        fmin = float(self.fmin_var.get())

        # n_bins: cover from fmin up to ~20 kHz
        n_octaves = np.log2(min(self.sr / 2 * 0.95, 20000.0) / fmin)
        n_bins = int(np.ceil(n_octaves * bpo))

        def worker():
            try:
                data = self.audio_orig.copy()

                # Apply time stretching
                if abs(stretch - 1.0) > 1e-6:
                    data = cqt_time_stretch(data, stretch, self.sr, hop,
                                            fmin, n_bins, bpo)

                # Apply pitch shifting (keep current duration)
                if abs(semitones) > 1e-6:
                    cur_len = len(data)
                    data = cqt_pitch_shift(data, semitones, self.sr, hop,
                                           fmin, n_bins, bpo,
                                           target_length=cur_len)

                # Sanitise
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

                # Normalise
                peak = np.max(np.abs(data))
                if peak > 1e-6:
                    data = data / peak * 0.95

                self.audio_proc = data
                self.root.after(0, self._on_process_done)
            except Exception as e:
                self.root.after(0,
                                lambda: self._on_process_error(str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_process_done(self):
        self.processing = False
        self.btn_process.config(state="normal", text="⚙ Process")
        self.btn_play_proc.config(state="normal")
        self.btn_export.config(state="normal")
        self._draw_cq_spectrogram(self.ax_proc, self.audio_proc, self.sr,
                                   "Processed")
        self.canvas.draw()
        dur = len(self.audio_proc) / self.sr
        self._update_status(
            f"Done — processed audio is {dur:.1f}s. "
            f"Ready for playback or export.")

    def _on_process_error(self, msg: str):
        self.processing = False
        self.btn_process.config(state="normal", text="⚙ Process")
        messagebox.showerror("Processing Error", msg)
        self._update_status("Processing failed.")

    # ── Export ───────────────────────────────

    def _export(self):
        if self.audio_proc is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV", "*.wav")])
        if path:
            sf.write(path, self.audio_proc.astype(np.float32), self.sr)
            self._update_status(f"Exported to {Path(path).name}")

    # ── CQ Spectrogram drawing ───────────────

    def _draw_cq_spectrogram(self, ax, data: np.ndarray, sr: int,
                              title: str):
        """Draw a CQ spectrogram (log-frequency axis) for visualisation."""
        ax.clear()
        ax.set_facecolor("#16161e")
        ax.set_title(title, color="#c0caf5", fontsize=11, fontweight="bold")

        # Use a lightweight CQT for visualisation (fewer bins, larger hop)
        vis_bpo = 24
        vis_fmin = 32.7
        n_oct = np.log2(min(sr / 2 * 0.95, 16000.0) / vis_fmin)
        vis_bins = int(np.ceil(n_oct * vis_bpo))
        vis_hop = max(512, len(data) // 500)  # keep frame count manageable

        CQ_vis, freqs_vis, _ = cqt(data, sr, vis_hop, vis_fmin,
                                    vis_bins, vis_bpo)
        S_db = 20 * np.log10(np.abs(CQ_vis) + 1e-10)
        vmax = S_db.max()
        vmin = max(vmax - 80, S_db.min())

        t_axis = np.arange(CQ_vis.shape[1]) * vis_hop / sr

        ax.pcolormesh(t_axis, freqs_vis, S_db, cmap="inferno",
                      vmin=vmin, vmax=vmax, shading="gouraud")
        ax.set_yscale("log")
        ax.set_ylim(freqs_vis.min(), freqs_vis.max())
        ax.set_xlabel("Time (s)", color="#9aa5ce", fontsize=9)
        ax.set_ylabel("Frequency (Hz)", color="#9aa5ce", fontsize=9)
        ax.tick_params(colors="#565f89", labelsize=8)

    def _clear_axis(self, ax, title: str):
        ax.clear()
        ax.set_facecolor("#16161e")
        ax.set_title(title, color="#c0caf5", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)", color="#9aa5ce", fontsize=9)
        ax.set_ylabel("Frequency (Hz)", color="#9aa5ce", fontsize=9)
        ax.tick_params(colors="#565f89", labelsize=8)
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                color="#414868", fontsize=14, transform=ax.transAxes)

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
    app = CQTProcessorApp(root)
    root.mainloop()
