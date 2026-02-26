"""
Test Suite — Phase Vocoder, Wavelet Processor, and CQT Processor
=================================================================
Verifies pitch preservation, duration accuracy, pitch-shift accuracy,
level consistency, and absence of phantom tones for all three
time-stretch / pitch-shift implementations.

Run:
    python tests/test_all.py            (standalone — prints results)
    pytest tests/test_all.py -v         (with pytest)
"""

import sys
import os
import numpy as np

# ── Locate the source modules regardless of working directory ──
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, os.pardir, "src"))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scipy.signal import resample

# ── Imports from each processor ──
from phase_vocoder import phase_vocoder_stretch, pitch_shift as stft_pitch_shift
from wavelet_processor import wavelet_time_stretch, wavelet_pitch_shift
from cqt_processor import cqt_time_stretch, cqt_pitch_shift

# ────────────────────────────────────────────────
#  Shared helpers
# ────────────────────────────────────────────────

SR = 44100

def make_sine(freq, duration=2.0, sr=SR):
    t = np.arange(int(sr * duration)) / sr
    return np.sin(2 * np.pi * freq * t) * 0.9

def make_harmonics(f0, duration=2.0, sr=SR):
    t = np.arange(int(sr * duration)) / sr
    x = (np.sin(2 * np.pi * f0 * t)
         + 0.5 * np.sin(2 * np.pi * 2 * f0 * t)
         + 0.25 * np.sin(2 * np.pi * 3 * f0 * t))
    return x / np.max(np.abs(x)) * 0.9

def make_whistle(freq=1500.0, duration=2.0, sr=SR):
    """Whistle-like tone with slight vibrato and a touch of noise."""
    np.random.seed(42)
    t = np.arange(int(sr * duration)) / sr
    vibrato = 5.0 * np.sin(2 * np.pi * 5.0 * t)
    phase = 2 * np.pi * np.cumsum(freq + vibrato) / sr
    x = 0.8 * np.sin(phase) + 0.05 * np.random.randn(len(t))
    return x / np.max(np.abs(x)) * 0.9

def make_noise(duration=2.0, sr=SR):
    np.random.seed(0)
    return np.random.randn(int(sr * duration)) * 0.3

def peak_freq(sig, sr=SR):
    """Dominant spectral peak (Hz)."""
    Y = np.fft.rfft(sig * np.hanning(len(sig)))
    f = np.fft.rfftfreq(len(sig), 1.0 / sr)
    mag = np.abs(Y)
    mag[:10] = 0
    return f[np.argmax(mag)]

def top_peaks(sig, sr=SR, n=5):
    """Return the *n* strongest spectral peaks as (freq, magnitude) tuples."""
    Y = np.fft.rfft(sig * np.hanning(len(sig)))
    f = np.fft.rfftfreq(len(sig), 1.0 / sr)
    mag = np.abs(Y)
    mag[:10] = 0
    peaks = []
    m = mag.copy()
    for _ in range(n):
        idx = np.argmax(m)
        if m[idx] < np.max(mag) * 0.005:
            break
        peaks.append((f[idx], m[idx]))
        lo, hi = max(0, idx - 80), min(len(m), idx + 80)
        m[lo:hi] = 0
    return peaks

def rms(sig):
    return np.sqrt(np.mean(sig ** 2))

def phantom_level(sig, expected_freq, stretch, sr=SR):
    """
    Return the level (dB relative to main peak) of any tone near
    expected_freq / stretch — the frequency a naive-resampled residual
    would produce.  -999 means nothing detected.
    """
    phantom_freq = expected_freq / stretch
    peaks = top_peaks(sig, sr, 8)
    if not peaks:
        return -999.0
    main_mag = peaks[0][1]
    for pf, pm in peaks:
        if abs(pf - phantom_freq) < 30:
            return 20 * np.log10(pm / main_mag + 1e-15)
    return -999.0


# ────────────────────────────────────────────────
#  Wrapper for each implementation
# ────────────────────────────────────────────────

def stretch_stft(x, factor):
    return phase_vocoder_stretch(x, factor)

def stretch_cwt(x, factor):
    return wavelet_time_stretch(x, factor, SR, n_voices=48, fmin=50.0, fmax=18000.0)

def stretch_cqt(x, factor):
    return cqt_time_stretch(x, factor, SR)

def pshift_stft(x, semi):
    return stft_pitch_shift(x, semi, SR, target_length=len(x))

def pshift_cwt(x, semi):
    return wavelet_pitch_shift(x, semi, SR, n_voices=48, fmin=50.0, fmax=18000.0,
                               target_length=len(x))

def pshift_cqt(x, semi):
    return cqt_pitch_shift(x, semi, SR, target_length=len(x))


STRETCH_FNS = [
    ("STFT",    stretch_stft),
    ("CWT",     stretch_cwt),
    ("CQT",     stretch_cqt),
]
PSHIFT_FNS = [
    ("STFT",    pshift_stft),
    ("CWT",     pshift_cwt),
    ("CQT",     pshift_cqt),
]


# ────────────────────────────────────────────────
#  Tests
# ────────────────────────────────────────────────

class Results:
    """Collects pass/fail for standalone runner."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
    def ok(self, msg):
        self.passed += 1
        print(f"  ✅ {msg}")
    def fail(self, msg):
        self.failed += 1
        print(f"  ❌ {msg}")
    def check(self, cond, msg):
        (self.ok if cond else self.fail)(msg)

R = Results()

# ── 1. Duration accuracy ────────────────────────

def test_duration():
    """Output length should equal input length × stretch factor."""
    print("\n── Duration accuracy ──")
    x = make_sine(440, duration=1.5)
    for name, fn in STRETCH_FNS:
        for s in [0.5, 1.5, 2.0]:
            y = fn(x, s)
            expected = len(x) * s
            ratio = len(y) / expected
            # STFT trims overlap-add edges so allow wider tolerance
            tol = 0.10 if name == "STFT" else 0.05
            R.check(
                (1 - tol) < ratio < (1 + tol),
                f"{name} stretch={s:.1f}x  duration ratio={ratio:.3f}",
            )

# ── 2. Pitch preservation during time-stretch ───

def test_pitch_preservation():
    """Dominant frequency should not change after time-stretching."""
    print("\n── Pitch preservation (time-stretch) ──")
    for freq, label in [(440, "440 Hz sine"), (1200, "1200 Hz sine")]:
        x = make_sine(freq, duration=2.0)
        for name, fn in STRETCH_FNS:
            for s in [0.5, 1.5, 2.0]:
                y = fn(x, s)
                pf = peak_freq(y)
                err = abs(pf - freq)
                R.check(
                    err < 15,
                    f"{name} {label} s={s:.1f}x  peak={pf:.1f}Hz (err={err:.1f}Hz)",
                )

# ── 3. Pitch-shift accuracy ─────────────────────

def test_pitch_shift_accuracy():
    """After pitch-shifting, the dominant frequency should match the target."""
    print("\n── Pitch-shift accuracy ──")
    x = make_sine(440, duration=2.0)
    for name, fn in PSHIFT_FNS:
        for semi in [-12, -5, 5, 12]:
            y = fn(x, semi)
            pf = peak_freq(y)
            expected = 440.0 * 2 ** (semi / 12.0)
            err = abs(pf - expected)
            # STFT pitch_shift uses stretch+resample; allow wider tolerance
            tol = 25 if name == "STFT" else 15
            R.check(
                err < tol,
                f"{name} semi={semi:+3d}  peak={pf:.1f}Hz (exp {expected:.1f}, err={err:.1f})",
            )

# ── 4. Pitch-shift duration preservation ────────

def test_pitch_shift_preserves_duration():
    """Pitch-shifting with target_length should preserve the signal length."""
    print("\n── Pitch-shift duration preservation ──")
    x = make_sine(440, duration=2.0)
    for name, fn in PSHIFT_FNS:
        for semi in [-7, 7]:
            y = fn(x, semi)
            ratio = len(y) / len(x)
            R.check(
                0.98 < ratio < 1.02,
                f"{name} semi={semi:+d}  length ratio={ratio:.4f}",
            )

# ── 5. Harmonic structure preservation ───────────

def test_harmonic_structure():
    """Time-stretching should preserve harmonic ratios."""
    print("\n── Harmonic structure preservation ──")
    x = make_harmonics(440)
    for name, fn in STRETCH_FNS:
        y = fn(x, 1.5)
        peaks = top_peaks(y, n=3)
        freqs_found = sorted([p[0] for p in peaks])
        # Should find roughly 440, 880, 1320
        has_f0 = any(abs(f - 440) < 20 for f in freqs_found)
        has_h2 = any(abs(f - 880) < 20 for f in freqs_found)
        has_h3 = any(abs(f - 1320) < 30 for f in freqs_found)
        R.check(
            has_f0 and has_h2,
            f"{name} s=1.5x  found=[{', '.join(f'{f:.0f}' for f in freqs_found)}]  "
            f"(f0={'✓' if has_f0 else '✗'} h2={'✓' if has_h2 else '✗'} h3={'✓' if has_h3 else '✗'})",
        )

# ── 6. No phantom tones from residual ───────────

def test_no_phantom_tones():
    """
    Stretched tonal signals should not contain a strong tone at
    freq / stretch (which naive residual resampling would produce).
    """
    print("\n── Absence of phantom tones ──")
    whistle = make_whistle(1500.0)
    for name, fn in STRETCH_FNS:
        for s in [1.5, 2.0]:
            y = fn(whistle, s)
            plvl = phantom_level(y, 1500.0, s)
            R.check(
                plvl < -20,
                f"{name} whistle@1500Hz s={s:.1f}x  phantom@{1500/s:.0f}Hz = {plvl:+.1f}dB",
            )

# ── 7. Level consistency ────────────────────────

def test_level_consistency():
    """Output RMS should be within ±6 dB of input RMS."""
    print("\n── Level consistency ──")
    for sig_name, x in [("sine 440Hz", make_sine(440)),
                         ("noise", make_noise())]:
        rms_in = rms(x)
        for name, fn in STRETCH_FNS:
            for s in [0.5, 1.5, 2.0]:
                y = fn(x, s)
                ratio_db = 20 * np.log10(rms(y) / rms_in + 1e-15)
                R.check(
                    abs(ratio_db) < 6.0,
                    f"{name} {sig_name} s={s:.1f}x  level={ratio_db:+.1f}dB",
                )

# ── 8. Identity stretch ─────────────────────────

def test_identity_stretch():
    """Stretch factor of 1.0 should return the input unchanged."""
    print("\n── Identity stretch (factor=1.0) ──")
    x = make_sine(440, duration=1.0)
    for name, fn in STRETCH_FNS:
        y = fn(x, 1.0)
        R.check(
            np.allclose(x, y, atol=1e-10),
            f"{name} stretch=1.0  output == input",
        )

# ── 9. Identity pitch-shift ─────────────────────

def test_identity_pitch_shift():
    """Pitch shift of 0 semitones should return the input unchanged."""
    print("\n── Identity pitch-shift (0 semitones) ──")
    x = make_sine(440, duration=1.0)
    for name, fn in PSHIFT_FNS:
        y = fn(x, 0.0)
        R.check(
            np.allclose(x, y, atol=1e-10),
            f"{name} semi=0  output == input",
        )

# ── 10. Extreme stretch factors ─────────────────

def test_extreme_stretch():
    """The processors should handle boundary stretch values without crashing."""
    print("\n── Extreme stretch factors ──")
    x = make_sine(440, duration=1.0)
    for name, fn in STRETCH_FNS:
        for s in [0.25, 4.0]:
            try:
                y = fn(x, s)
                ok = len(y) > 0 and np.all(np.isfinite(y))
                R.check(ok, f"{name} stretch={s:.2f}x  len={len(y)}, finite={ok}")
            except Exception as e:
                R.fail(f"{name} stretch={s:.2f}x  raised {type(e).__name__}: {e}")

# ── 11. Octave pitch-shift round-trip ────────────

def test_octave_round_trip():
    """Shifting up 12 semitones then down 12 should recover the original frequency."""
    print("\n── Octave round-trip (+12 then −12 semitones) ──")
    x = make_sine(440, duration=2.0)
    for name, fn in PSHIFT_FNS:
        y_up = fn(x, 12)
        y_rt = fn(y_up, -12)
        pf = peak_freq(y_rt)
        err = abs(pf - 440)
        R.check(err < 15, f"{name}  round-trip peak={pf:.1f}Hz (err={err:.1f}Hz)")

# ── 12. Short signal handling ────────────────────

def test_short_signal():
    """Processors should handle very short signals (< 1 hop) without crashing."""
    print("\n── Short signal handling ──")
    x_short = make_sine(440, duration=0.05)  # ~2205 samples — enough for at least 1 STFT frame
    for name, fn in STRETCH_FNS:
        try:
            y = fn(x_short, 1.5)
            R.check(len(y) > 0, f"{name}  short signal ({len(x_short)} samples) → {len(y)} samples")
        except Exception as e:
            R.fail(f"{name}  short signal raised {type(e).__name__}: {e}")


# ────────────────────────────────────────────────
#  Runner
# ────────────────────────────────────────────────

ALL_TESTS = [
    test_duration,
    test_pitch_preservation,
    test_pitch_shift_accuracy,
    test_pitch_shift_preserves_duration,
    test_harmonic_structure,
    test_no_phantom_tones,
    test_level_consistency,
    test_identity_stretch,
    test_identity_pitch_shift,
    test_extreme_stretch,
    test_octave_round_trip,
    test_short_signal,
]


def main():
    print("=" * 64)
    print("  Audio Processor Test Suite")
    print("  STFT · CWT · CQT")
    print("=" * 64)

    for test_fn in ALL_TESTS:
        test_fn()

    print("\n" + "=" * 64)
    total = R.passed + R.failed
    print(f"  Results: {R.passed}/{total} passed, {R.failed} failed")
    print("=" * 64)
    return 0 if R.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
