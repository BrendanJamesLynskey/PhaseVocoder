# Wavelet Audio Processor — Technical Report

*Time-Stretching and Pitch-Shifting Using the Continuous Wavelet Transform*

## 1. Overview

The Wavelet Audio Processor is a desktop application for Windows that performs time-stretching and pitch-shifting of audio signals. Unlike the conventional Short-Time Fourier Transform (STFT) phase vocoder, which analyses audio with fixed-size windows, this application uses the Continuous Wavelet Transform (CWT) with Morlet wavelets. The CWT provides logarithmic frequency resolution that naturally matches the human auditory system, offering better time resolution at high frequencies and better frequency resolution at low frequencies.

The application is written in Python with a Tkinter graphical interface and uses NumPy and SciPy for all signal processing. It supports loading WAV files, recording from a microphone, adjusting time-stretch and pitch-shift parameters, visualising results as scalograms, and exporting processed audio.

## 2. The Continuous Wavelet Transform

### 2.1 The Morlet Wavelet

The application uses the analytic Morlet wavelet as its mother wavelet. The Morlet wavelet is a complex sinusoid modulated by a Gaussian envelope. For a given scale *s* and centre frequency parameter ω₀ = 6.0, the wavelet at sample position *t* is defined as:

$$\psi(t) = \pi^{- 1/4} \cdot s^{- 1/2} \cdot exp\left( \frac{i\omega_{0}t}{s} \right) \cdot exp\left( - \frac{t^{2}}{2s^{2}} \right)$$

The normalisation factor ensures that wavelet energy is independent of scale, allowing fair comparison of coefficients across the frequency spectrum. The parameter ω₀ = 6.0 provides a good balance between time and frequency localisation.

### 2.2 Forward Transform

The CWT decomposes the input signal *x(t)* into a two-dimensional representation *W(s, t)* where *s* is the wavelet scale and *t* is time. Each coefficient is computed as the convolution of the signal with the complex conjugate of the wavelet at scale *s*:

$$W(s,t) = x(t) \ast \overline{\psi}\left( \frac{t}{s} \right)$$

In practice, this convolution is computed efficiently using the FFT via SciPy's fftconvolve function. The wavelet at each scale is truncated to ±4 standard deviations of the Gaussian envelope, keeping computation manageable while retaining over 99.99% of the wavelet's energy.

### 2.3 Scale-to-Frequency Mapping

Wavelet scales are logarithmically spaced between a minimum frequency (default 20 Hz) and a maximum frequency (default 20,000 Hz, clamped to 95% of Nyquist). For the Morlet wavelet, the relationship between scale *s* and centre frequency *f* is:

$$f = \frac{\omega_{0} \cdot f_{s}}{2\pi \cdot s}$$

where *fₛ* is the sample rate. The default configuration uses 64 voices (scales) distributed geometrically across the full audible range, giving approximately 10 voices per octave.

### 2.4 Inverse Transform

The inverse CWT reconstructs the time-domain signal by integrating the real parts of the wavelet coefficients across all scales, weighted by the scale spacing and an inverse power of the scale:

$$\widehat{x}(t) = \frac{1}{C_{\psi}}\sum_{s}^{\ }{Re\left\lbrack W(s,t) \right\rbrack \cdot \frac{\Delta s}{s^{3/2}}}$$

The admissibility constant *Cψ* = 0.776 is determined empirically for ω₀ = 6.0. Because the CWT is a redundant (overcomplete) transform, perfect reconstruction is not guaranteed; the approximate inverse recovers approximately 27% of the original amplitude uniformly across frequencies. A least-squares scaling factor is computed and applied to compensate for this (see Section 3.2).

## 3. Time Stretching

### 3.1 Instantaneous Frequency Phase Synthesis

The time-stretching algorithm operates in three stages:

1.  **Analysis:** The input signal is decomposed via the CWT into complex coefficients at each scale.

2.  **Coefficient stretching:** At each scale, the magnitude envelope and instantaneous frequency are interpolated to the target length, and phase is resynthesised by integration.

3.  **Reconstruction:** The inverse CWT reconstructs the stretched signal, and the residual is blended back in (see Section 3.2).

The critical step is the phase synthesis in Stage 2. A naive approach of linearly interpolating the unwrapped phase between original time positions produces incorrect results: the phase advances at the wrong rate, unintentionally shifting the pitch of the output. This was confirmed experimentally, with a 440 Hz sine wave dropping to 370 Hz after a 2× stretch.

The correct approach extracts the **instantaneous frequency** at each scale, defined as the numerical derivative of the unwrapped phase:

$$\omega_{inst}(t) = \frac{d\varphi(t)}{dt}$$

This instantaneous frequency is interpolated to the output timeline, and the output phase is synthesised by cumulative summation (numerical integration):

$$\varphi_{out}(n) = \sum_{k = 0}^{n}{\omega_{inst,interp}(k)}$$

This ensures that the oscillation at each scale continues at its natural frequency regardless of how the magnitude envelope is stretched, preserving pitch while changing duration.

### 3.2 Residual Preservation

The CWT with a finite set of scales cannot perfectly represent the entire signal. A residual-preservation strategy ensures that spectral content outside the analysis band (and reconstruction error within it) is not lost:

4.  After the forward CWT, an immediate inverse CWT reconstructs what the transform captures.

5.  A least-squares scaling factor is computed: α = ⟨x, x̂⟩ / ⟨x̂, x̂⟩, so that α·x̂ best approximates x.

6.  The residual r = x − α·x̂ is computed. This contains energy the CWT cannot represent.

7.  After the CWT coefficients are stretched and reconstructed (scaled by α), the residual is time-stretched using a stochastic synthesis method and added back.

The residual is predominantly noise-like (transients, breath, consonants, unpitched texture). A naive approach of interpolating the residual waveform sample-by-sample fails because it merely thins out the noise rather than properly stretching it — the noise events land at approximately correct positions but the texture between them is sparse and low-pass filtered. Similarly, frequency-domain resampling (e.g. `scipy.signal.resample`) compresses the spectrum, shifting the residual's pitch down by the inverse of the stretch factor.

The stochastic residual method, inspired by the Spectral Modeling Synthesis (SMS) framework, avoids both problems:

1. The residual is analysed with a short STFT to capture per-frame magnitude spectra, and a per-frame RMS amplitude envelope is extracted.

2. Both the magnitude spectra and the envelope are interpolated to the target stretched length.

3. Fresh noise is synthesised at each output frame by applying random phases to the interpolated magnitude spectra, then overlap-added.

4. The synthesised noise is modulated by the interpolated envelope so that transient positions move correctly in time.

This regenerates the noise at full density at every output sample with the correct spectral colour and amplitude contour. Without this strategy, the spectral centroid of the output drops dramatically. Testing with broadband noise showed the centroid falling from 11,033 Hz to 4,269 Hz (a ratio of 0.39). With residual preservation and the full 20--20,000 Hz analysis range, the centroid ratio improves to 0.96, which is perceptually transparent.

## 4. Pitch Shifting

Pitch shifting combines two operations:

8.  **Time-stretch** by the pitch ratio. For a shift of *n* semitones, the stretch factor is 2ⁿ˲¹². This changes the signal's duration without changing its pitch.

9.  **Resample** back to the target length using SciPy's polyphase resampling. This compresses or expands the waveform in time, shifting all frequencies by the desired ratio.

An alternative approach---directly reassigning energy between wavelet scales---was investigated but found to be ineffective. The inverse CWT synthesis is tied to fixed scales: placing 440 Hz content into the slot for scale index corresponding to 523 Hz does not produce 523 Hz output, because the synthesis wavelets oscillate at their own fixed frequencies. The stretch-plus-resample method avoids this fundamental limitation.

## 5. Comparison with the Phase Vocoder

The following table summarises the key differences between the STFT-based phase vocoder and the wavelet-based approach:

**Property**           **Phase Vocoder (STFT)**         **Wavelet Processor (CWT)**
**Property**           **Phase Vocoder (STFT)**         **Wavelet Processor (CWT)**

  Frequency resolution   Linear (uniform bins)            Logarithmic (matches hearing)

  Time resolution        Fixed (determined by FFT size)   Adaptive (better at high frequencies)

  Transient handling     Smeared across analysis window   Better preserved due to short high-freq wavelets

  Computational cost     Lower (single FFT size)          Higher (convolution at each scale)

  Reconstruction         Near-perfect (overlap-add)       Approximate (requires residual compensation)

  Phase coherence        Requires explicit locking        Natural per-scale phase tracking

In practice, the wavelet approach tends to produce fewer metallic artefacts on percussive and speech material, at the cost of higher computation time and a more complex reconstruction pipeline.

## 6. Application Parameters

The user interface exposes the following controls:

**Parameter**    **Range**        **Description**
**Parameter**    **Range**        **Description**

  Time Stretch     0.25× -- 4.0×    Stretch factor. Values \> 1 slow the audio down; values \< 1 speed it up.

  Pitch Shift      −12 -- +12 st    Pitch shift in semitones. Positive values raise pitch; negative values lower it.

  Voices           24 -- 96         Number of wavelet scales. More voices give finer frequency resolution but increase processing time.

  Freq range       20 -- 20000 Hz   Analysis band. Should span the full audible range for best results. Content outside is handled by the residual.

## 7. Audio Playback on Windows

Audio playback uses the sounddevice library with an explicit callback-based OutputStream rather than the simpler sd.play() function. This design choice addresses a reliability issue on Windows where sd.play() can silently fail on some WASAPI and DirectSound backends. The callback feeds audio data frame-by-frame from a pre-allocated buffer, and the stream lifecycle is explicitly managed to prevent garbage collection from interrupting playback.

A related issue was discovered with the finished_callback parameter: sounddevice's underlying C callback requires a void return, but Tkinter's root.after() returns a timer ID. This caused TypeError exceptions. The fix wraps the callback in a tuple expression that discards the return value.

## 8. Signal Flow Summary

The complete signal processing chain for a combined time-stretch and pitch-shift operation:

10. Input signal is loaded (mono, float64).

11. If time-stretch ≠ 1.0: CWT analysis → instantaneous frequency extraction → magnitude/phase interpolation → iCWT reconstruction + residual blending.

12. If pitch shift ≠ 0: the (possibly already stretched) signal is CWT-stretched by the pitch ratio, then resampled to the current length.

13. NaN/Inf values are replaced with zeros.

14. Peak normalisation to 0.95 prevents clipping.

15. Output is stored for playback, visualisation, and export.

## 9. Dependencies

The application requires Python 3.10 or newer with the following packages:

-   numpy -- array operations and FFT

-   scipy -- signal processing (fftconvolve, resample), interpolation

-   sounddevice -- audio I/O via PortAudio

-   soundfile -- WAV file reading and writing via libsndfile

-   matplotlib -- scalogram visualisation (TkAgg backend)

All dependencies are installable via pip install -r requirements.txt.

*--- End of Report ---*
