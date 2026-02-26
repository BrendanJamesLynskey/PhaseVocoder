# Constant-Q Transform Audio Processor — Technical Report

*Time-Stretching and Pitch-Shifting Using the Constant-Q Transform*

## 1. Overview

The Constant-Q Transform Audio Processor is a desktop application for Windows that performs time-stretching and pitch-shifting of audio signals using the Constant-Q Transform (CQT). The CQT analyses a signal on a logarithmic frequency grid where each bin has a constant ratio between its centre frequency and its bandwidth, giving musically meaningful resolution: low frequencies receive fine spectral detail while high frequencies receive fine temporal detail. This places the CQT between the STFT (uniform linear bins, fixed window) and the CWT (continuous scales at arbitrary resolution).

The application is written in Python with a Tkinter graphical interface and uses NumPy and SciPy for all signal processing. It supports loading WAV files, recording from a microphone, adjusting time-stretch and pitch-shift parameters, visualising results as CQ spectrograms (log-frequency axis), and exporting processed audio.

## 2. The Constant-Q Transform

### 2.1 Analysis Kernels

The CQT decomposes a signal into frequency bins whose centre frequencies are geometrically spaced. For a given lowest frequency fmin and bins_per_octave parameter B, the centre frequency of bin k is:

> *f_k = fmin × 2\^(k / B)*

The quality factor Q --- the ratio of centre frequency to bandwidth --- is constant across all bins:

> *Q = q_factor / (2\^(1/B) − 1)*

For B = 12 (semitone spacing) and q_factor = 1.0, Q ≈ 17.3, meaning each bin's bandwidth is approximately 1/17 of its centre frequency. Higher values of B (e.g. 24 for quarter-tone resolution) increase Q and narrow the bandwidth.

Each analysis kernel is a windowed complex exponential of length N_k = ⌈Q × sr / f_k⌉, where sr is the sample rate. Low-frequency bins have long kernels (good frequency resolution) and high-frequency bins have short kernels (good time resolution). Each kernel is multiplied by a Hann window and normalised by 1/N_k:

> *kernel_k(t) = (1/N_k) × w(t) × exp(2πj × f_k × t / sr), t = 0, 1, ..., N_k−1*

Kernels that would place a centre frequency above the Nyquist rate (95% of sr/2) are discarded.

### 2.2 Forward Transform

The CQT of a mono signal x(n) is computed by convolving the signal with the conjugate-reversed analysis kernel at each bin, then downsampling the result at hop-size intervals:

> *CQ(k, m) = (x \* kernel_k̅)\[m × hop\]*

where m is the frame index and hop is the analysis time step (default 512 samples). The convolution is performed efficiently via FFT (SciPy's fftconvolve). The result is a complex CQ spectrogram of shape (n_bins, n_frames), where n_bins depends on the frequency range and bins_per_octave, and n_frames depends on the signal length and hop size.

With B = 12 and fmin = 32.7 Hz (C1), the default configuration produces 84 bins spanning 7 octaves up to approximately 4186 Hz (C8). Higher bins_per_octave values give finer frequency resolution at the cost of more computation and longer kernel lengths at low frequencies.

### 2.3 Inverse Transform

The inverse CQT reconstructs the time-domain signal by overlap-add of windowed sinusoidal atoms. For each frequency bin k and each frame m, a synthesis atom is placed at the frame's time position:

> *atom(t) = w(t) × exp(2πj × f_k × t / sr)*

The atom is scaled by the complex CQ coefficient CQ(k, m), and its real part is accumulated into the output buffer. A running sum of squared window values tracks the overlap density, and the output is normalised by dividing by this sum. This is the standard overlap-add approach used by many CQT implementations.

Note: the inverse CQT is used only for spectrogram visualisation in this application. Time-stretching uses a different resynthesis method (see Section 3) because the overlap-add inverse is sensitive to phase coherence between the coefficient and the carrier sinusoid inside each atom, producing audible artefacts when the coefficients have been modified.

## 3. Time Stretching

### 3.1 Challenges Specific to the CQT

Time-stretching with the CQT presents two challenges not encountered in the CWT-based approach:

**Phase unwrapping failure.** The CQT coefficients are sampled at hop-size intervals. For a bin at frequency f_k with hop H, the phase advances by 2π × f_k × H / sr radians per frame. At typical settings (f_k = 1200 Hz, H = 512, sr = 44100), this is approximately 87.5 radians --- roughly 14 full cycles. After wrapping to \[−π, π\], the per-frame difference appears to be a small step of about −0.4 radians. Standard phase unwrapping (which only corrects jumps exceeding π) interprets this as a smooth, nearly-stationary phase and fails to recover the true instantaneous frequency.

**Spectral leakage and tone splitting.** A pure tone falling between CQ bin centres spreads energy across several adjacent bins. If each bin synthesises a sinusoid at its own slightly different instantaneous frequency, the overlapping sinusoids beat against each other, splitting a single tone into two or more audible sidebands.

### 3.2 Heterodyne Phase Unwrapping

The phase unwrapping problem is solved by a heterodyne (carrier-relative) approach. Instead of applying standard unwrapping to the raw coefficient phase, the algorithm subtracts the expected carrier phase advance, wraps the small residual to \[−π, π\], and adds the carrier back:

**1.** Compute the raw phase difference between consecutive frames: Δφ = ∠CQ(k, m+1) − ∠CQ(k, m).

**2.** Subtract the expected carrier advance: residual = Δφ − 2π f_k H / sr.

**3.** Wrap the residual to \[−π, π\]: this captures the deviation between the signal's true frequency and the bin's centre frequency.

**4.** Recover the true phase advance: Δφ_true = 2π f_k H / sr + wrapped residual.

The true instantaneous frequency in radians per sample is then Δφ_true / H. This technique is analogous to how a radio receiver demodulates a signal by mixing it down to baseband before measuring its frequency content.

### 3.3 Phase Locking

The tone-splitting problem is addressed by phase locking. For each frame, the algorithm identifies local magnitude peaks across the frequency bins (bins whose magnitude exceeds both their neighbours). Every non-peak bin is then assigned the instantaneous frequency of its nearest peak bin. This forces spectral-leakage sidebands to oscillate at the same frequency as the true partial they belong to, so their energy reinforces the peak rather than creating phantom tones.

The phase-locking step is critical for perceptual quality on tonal material. Without it, a pure 1200 Hz whistle stretched to 1.5× produces audible sidebands at approximately 1114 Hz and 1286 Hz (the adjacent CQ bin centres). With phase locking, spectral purity reaches 100% --- only the original frequency is present in the output.

### 3.4 Sample-Level Additive Resynthesis

Rather than using the inverse CQT (Section 2.3), the time-stretching algorithm resynthesises the output by additive synthesis at sample-level resolution. For each CQ bin k:

**1.** The magnitude and instantaneous frequency envelopes (computed at frame rate) are interpolated to every output sample position, using the mapping output_time / stretch to look up the corresponding input time.

**2.** The output phase is obtained by cumulative summation (numerical integration) of the interpolated instantaneous frequency, aligned to the initial phase of the first CQ coefficient.

**3.** A cosine oscillator at the integrated phase, modulated by the interpolated magnitude, contributes to the output signal.

All bins are summed to produce the final stretched waveform. This approach avoids the overlap-add artefacts of the inverse CQT: because each bin's sinusoid is generated continuously from an integrated phase trajectory, there are no frame-boundary discontinuities and no dependence on window-overlap normalisation.

A least-squares scale factor is computed by performing the same additive resynthesis on the original (unstretched) signal and comparing it to the input: α = ⟨x, x̂⟩ / ⟨x̂, x̂⟩. This factor is applied to the stretched output to match the original signal level.

### 3.5 Residual Preservation

The CQT with a finite number of bins cannot perfectly represent the entire signal. Content above the highest bin (e.g. above 4 kHz with the default 84-bin configuration), noise, and transient energy that falls between the sparse analysis frames all contribute to a residual:

> *residual = x − α × x̂*

where x̂ is the additive resynthesis of the original signal and α is the least-squares scale factor.

The residual is always included in the output to preserve full-bandwidth fidelity and prevent silence on broadband material. However, naive resampling of the residual to the target length would shift its pitch by 1/stretch — creating audible phantom tones. Testing confirmed that a 1500 Hz whistle stretched to 1.5× produced a spurious 1000 Hz tone at only −10 dB below the true pitch when the residual was naively resampled.

To avoid this, the residual is time-stretched using a lightweight STFT phase vocoder (2048-point FFT, 512-sample hop) which preserves pitch while changing duration. This eliminates the phantom tones completely: the same 1500 Hz whistle stretched to 1.5× shows only the original frequency in the output, with no detectable pitch-shifted artefacts.

After blending, the output is RMS-matched to the input to ensure consistent loudness regardless of stretch factor.

## 4. Pitch Shifting

Pitch shifting combines two operations:

**1. Time-stretch** by the pitch ratio. For a shift of n semitones, the stretch factor is 2\^(n/12). This changes the signal's duration without changing its pitch.

**2. Resample** back to the target length using SciPy's polyphase resampling. This compresses or expands the waveform in time, shifting all frequencies by the desired ratio while restoring the original duration.

For example, to shift pitch up by 7 semitones (a perfect fifth): the signal is first stretched to 2\^(7/12) ≈ 1.498 times its original length, then resampled back to the original length, raising all frequencies by the same ratio.

## 5. Comparison with STFT and CWT Approaches

The following table summarises the key differences between the three approaches:

**Property**             **Phase Vocoder (STFT)**                               **Wavelet Processor (CWT)**                       **CQT Processor**
**Property**             **Phase Vocoder (STFT)**                               **Wavelet Processor (CWT)**                       **CQT Processor**

  Frequency resolution     Linear (uniform FFT bins)                              Logarithmic (continuous scales)                   Logarithmic (discrete bins)

  Time resolution          Fixed (determined by FFT size)                         Adaptive (finer at high frequencies)              Adaptive (shorter kernels at high frequencies)

  Frequency grid           Fixed by FFT size (e.g. 1025 bins from 0 to Nyquist)   User-defined continuous scales (e.g. 64 voices)   User-defined discrete bins (e.g. 12 per octave)

  Resynthesis method       Overlap-add (inverse FFT)                              Weighted sum over scales (inverse CWT)            Sample-level additive synthesis

  Reconstruction quality   Near-perfect (COLA-compliant overlap-add)              Approximate (least-squares compensated)           Approximate (least-squares + RMS matching)

  Phase unwrapping         Standard (bin-relative, small hop advance)             Standard (sample-level, no hop aliasing)          Heterodyne (carrier-relative, handles large hop advances)

  Phase coherence          Explicit accumulation across frames                    Natural per-scale tracking                        Phase locking across bins + continuous integration

  Transient handling       Smeared across fixed analysis window                   Better preserved (short high-freq wavelets)       Intermediate (adaptive kernels, but hop-limited)

  Computational cost       Lowest (single FFT size)                               Highest (convolution per scale per sample)        Medium (convolution per bin, hop subsampling)

  Typical artefacts        Metallic / phasey at extreme stretches                 Smooth but less precise; residual colouration     Clean on tonal; STFT residual stretch preserves pitch

  Best for                 Tonal material, moderate stretches                     Percussive material, speech                       Musical material (log-freq matches musical intervals)

## 6. Application Parameters

The user interface exposes the following controls:

**Parameter**   **Range**       **Description**
**Parameter**   **Range**       **Description**

  Time Stretch    0.25× -- 4.0×   Stretch factor. Values \> 1 slow the audio down; values \< 1 speed it up.

  Pitch Shift     −12 -- +12 st   Pitch shift in semitones. Positive values raise pitch; negative values lower it.

  Hop Size        128 -- 1024     Analysis time step in samples. Smaller values give finer time resolution but increase computation.

  Bins/Octave     12 -- 48        Number of CQ bins per octave. 12 = semitone resolution; 24 = quarter-tone; 48 = eighth-tone. Higher values improve spectral purity.

  Min Freq        20 -- 200 Hz    Lowest CQ bin centre frequency. Lower values extend the analysis range but increase kernel length and computation time.

## 7. Audio Playback on Windows

Audio playback uses the sounddevice library with an explicit callback-based OutputStream rather than the simpler sd.play() function. This design choice addresses a reliability issue on Windows where sd.play() can silently fail on some WASAPI and DirectSound backends. The callback feeds audio data frame-by-frame from a pre-allocated buffer, and the stream lifecycle is explicitly managed to prevent garbage collection from interrupting playback.

A related issue was discovered with the finished_callback parameter: sounddevice's underlying C callback requires a void return, but Tkinter's root.after() returns a timer ID. This caused TypeError exceptions. The fix wraps the callback in a tuple expression that discards the return value.

## 8. Signal Flow Summary

The complete signal processing chain for a combined time-stretch and pitch-shift operation:

**1.** Input signal is loaded (mono, float64).

**2.** If time-stretch ≠ 1.0: CQT analysis → heterodyne instantaneous frequency extraction → phase locking → sample-level additive resynthesis + residual blending → RMS matching.

**3.** If pitch shift ≠ 0: the (possibly already stretched) signal is CQT-stretched by the pitch ratio, then polyphase-resampled to the current length.

**4.** NaN/Inf values are replaced with zeros.

**5.** Peak normalisation to 0.95 prevents clipping.

**6.** Output is stored for playback, CQ spectrogram visualisation, and WAV export.

## 9. Dependencies

The application requires Python 3.10 or newer with the following packages:

**numpy** --- array operations and FFT-based convolution

**scipy** --- signal processing (fftconvolve, Hann window, polyphase resampling)

**sounddevice** --- audio I/O via PortAudio

**soundfile** --- WAV file reading and writing via libsndfile

**matplotlib** --- CQ spectrogram visualisation (TkAgg backend)

All dependencies are installable via pip install -r requirements.txt.

*--- End of Report ---*
