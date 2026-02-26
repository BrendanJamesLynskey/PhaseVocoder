# Phase Vocoder — Technical Report

*Time-Stretching and Pitch-Shifting Using the Short-Time Fourier Transform*

## 1. Overview

The Phase Vocoder is a desktop application for Windows that performs time-stretching and pitch-shifting of audio signals using the Short-Time Fourier Transform (STFT). The STFT decomposes a signal into overlapping windowed frames, transforms each to the frequency domain via the FFT, and provides a uniform-resolution time--frequency representation. By manipulating the magnitude and phase of these spectral frames and resynthesising via the inverse STFT, the phase vocoder can change the duration of a signal independently of its pitch.

The application is written in Python with a Tkinter graphical interface and uses NumPy and SciPy for all signal processing. It supports loading WAV files, recording from a microphone, adjusting time-stretch and pitch-shift parameters, visualising results as spectrograms, and exporting processed audio.

## 2. The Short-Time Fourier Transform

### 2.1 Analysis (Forward STFT)

The forward STFT divides the input signal *x(n)* into overlapping frames of length *N* (the FFT size), each offset by *H* samples (the hop size). Each frame is multiplied by a Hann window *w(n)* before the FFT is applied:

$$X(m,k) = \sum_{n = 0}^{N - 1}{x(n + mH) \cdot w(n) \cdot e^{- j2\pi kn/N}}$$

where *m* is the frame index, *k* is the frequency bin index, *N* is the FFT size, and *H* is the hop size. The Hann window is defined as:

$$w(n) = 0.5\left( 1 - cos\left( \frac{2\pi n}{N} \right) \right)$$

The application uses *N* = 2048 and *H* = 512 by default, giving 75% overlap between successive frames and yielding *N*/2 + 1 = 1025 frequency bins from 0 Hz to the Nyquist frequency.

### 2.2 Synthesis (Inverse STFT)

The inverse STFT reconstructs the time-domain signal from modified spectral frames using overlap-add synthesis. Each frequency-domain frame is transformed back to the time domain via the inverse FFT, windowed again by the Hann window, and accumulated into an output buffer:

$$y(n) = \sum_{m}^{\ }{IFFT\left( Y(m,k) \right) \cdot w\left( n - mH_{out} \right)}$$

The output is normalised by the squared-window sum to compensate for the overlap-add gain. Regions at the start and end of the signal where fewer than two frames overlap are trimmed, since the window sum is too small there for accurate reconstruction. With a Hann window at 75% overlap, the squared-window sum is approximately constant (COLA condition), ensuring near-perfect reconstruction when *H* = *N*/4.

## 3. Time Stretching

### 3.1 Phase Accumulation Algorithm

Time-stretching changes the duration of a signal without altering its pitch. The phase vocoder achieves this by reading through the STFT spectrogram at a different rate than it was produced, while carefully managing phase coherence between adjacent synthesis frames.

The algorithm proceeds as follows:

1.  **Analysis:** Compute the STFT of the input signal with analysis hop size H.

2.  **Frame interpolation:** Generate output frame positions by stepping through the original spectrogram at intervals of 1/α, where α is the stretch factor. At each output position, the magnitude is linearly interpolated between the two nearest analysis frames.

3.  **Phase propagation:** For each frequency bin, compute the instantaneous frequency from the phase difference between adjacent analysis frames, then accumulate phase at the synthesis hop rate.

4.  **Synthesis:** Reconstruct the time-domain signal via inverse STFT with synthesis hop size Hₒᵤₜ = α·H.

The critical step is the phase propagation in Stage 3. The expected phase advance for bin k across one analysis hop is:

$$\omega_{k} = \frac{2\pi kH}{N}$$

The actual phase difference between consecutive frames may deviate from this expected value due to frequency content that does not fall exactly on a bin centre. The deviation is computed and wrapped to \[−π, π\]:

$$\Delta\varphi_{k} = \angle X(m + 1,k) - \angle X(m,k) - \omega_{k}$$

The true instantaneous frequency for bin k at frame m is then:

$$\omega_{inst,k} = \omega_{k} + \Delta\varphi_{k}$$

The output phase accumulator advances by the instantaneous frequency scaled by the stretch factor:

$$\varphi_{k}(m + 1) = \varphi_{k}(m) + \omega_{inst,k} \cdot \alpha$$

Each output frame is constructed from the interpolated magnitudes and accumulated phases, then the inverse STFT with synthesis hop Hₒᵤₜ reconstructs the stretched signal.

### 3.2 The Overlap-Add Constraint

For correct reconstruction, the synthesis window and hop must satisfy the Constant Overlap-Add (COLA) condition. With a Hann window and 75% overlap (*H* = *N*/4), the sum of squared windows is approximately unity at all points, meaning no amplitude modulation is introduced by the overlap-add process. When the synthesis hop *H*ₒᵤₜ differs from the analysis hop, the squared-window sum changes; the application compensates for this by dividing the output by the actual squared-window sum at each sample.

## 4. Pitch Shifting

Pitch shifting combines two operations:

5.  **Time-stretch** by the pitch ratio. For a shift of n semitones, the stretch factor is 2ⁿ˲¹². This changes the signal's duration without changing its pitch.

6.  **Resample** back to the target length using SciPy's polyphase resampling. This compresses or expands the waveform in time, shifting all frequencies by the desired ratio while restoring the original duration.

For example, to shift pitch up by 5 semitones: the signal is first stretched to 2⁵˲¹² ≈ 1.335 times its original length (making it longer and slower at the same pitch), then resampled back to its original length (which speeds it up, raising all frequencies by the same 1.335 ratio).

## 5. Comparison with Wavelet-Based Processing

The following table summarises the key differences between the STFT-based phase vocoder and a CWT-based wavelet approach:

**Property**           **Phase Vocoder (STFT)**               **Wavelet Processor (CWT)**
**Property**           **Phase Vocoder (STFT)**               **Wavelet Processor (CWT)**

  Frequency resolution   Linear (uniform bins)                  Logarithmic (matches hearing)

  Time resolution        Fixed (determined by FFT size)         Adaptive (better at high frequencies)

  Transient handling     Smeared across analysis window         Better preserved due to short high-freq wavelets

  Computational cost     Lower (single FFT size)                Higher (convolution at each scale)

  Reconstruction         Near-perfect (overlap-add)             Approximate (requires residual compensation)

  Phase coherence        Requires explicit phase accumulation   Natural per-scale phase tracking

The phase vocoder is the more established method, with lower computational cost and near-perfect reconstruction through overlap-add. Its main weakness is fixed time-frequency resolution: the analysis window cannot simultaneously resolve fast transients and closely-spaced spectral lines. Extreme stretch factors (above 3×) or pitch shifts (beyond ±8 semitones) tend to produce metallic or "phasey" artefacts, particularly on percussive material. Enhancements such as phase-locking and transient detection can mitigate these issues.

## 6. Application Parameters

The user interface exposes the following controls:

**Parameter**   **Range**       **Description**
**Parameter**   **Range**       **Description**

  Time Stretch    0.25× -- 4.0×   Stretch factor. Values \> 1 slow the audio down; values \< 1 speed it up.

  Pitch Shift     −12 -- +12 st   Pitch shift in semitones. Positive values raise pitch; negative lower it.

  FFT Size        512 -- 8192     Analysis window length. Larger values give better frequency resolution but poorer time resolution.

  Hop Size        128 -- 1024     Frame step size. Smaller values give more overlap and smoother results, at higher computational cost.

## 7. Audio Playback on Windows

Audio playback uses the sounddevice library with an explicit callback-based OutputStream rather than the simpler sd.play() function. This design choice addresses a reliability issue on Windows where sd.play() can silently fail on some WASAPI and DirectSound backends. The callback feeds audio data frame-by-frame from a pre-allocated buffer, and the stream lifecycle is explicitly managed to prevent garbage collection from interrupting playback.

A related issue was discovered with the finished_callback parameter: sounddevice's underlying C callback requires a void return, but Tkinter's root.after() returns a timer ID. This caused TypeError exceptions on the console. The fix wraps the callback in a tuple expression that discards the return value.

## 8. Signal Flow Summary

The complete signal processing chain for a combined time-stretch and pitch-shift operation:

7.  Input signal is loaded (mono, float64).

8.  If time-stretch ≠ 1.0: STFT analysis → magnitude interpolation → phase accumulation → inverse STFT with modified hop size.

9.  If pitch shift ≠ 0: the (possibly already stretched) signal is phase-vocoder-stretched by the pitch ratio, then polyphase-resampled to the current length.

10. NaN/Inf values are replaced with zeros.

11. Peak normalisation to 0.95 prevents clipping.

12. Output is stored for playback, spectrogram visualisation, and WAV export.

## 9. Dependencies

The application requires Python 3.10 or newer with the following packages:

-   numpy -- array operations and FFT (rfft, irfft)

-   scipy -- signal processing (Hann window, polyphase resampling)

-   sounddevice -- audio I/O via PortAudio

-   soundfile -- WAV file reading and writing via libsndfile

-   matplotlib -- spectrogram visualisation (TkAgg backend)

All dependencies are installable via pip install -r requirements.txt.

*--- End of Report ---*
