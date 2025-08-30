# Zoo API: Digital Signal Processing (DSP)

The `xinfer::zoo::dsp` module provides a suite of high-performance, GPU-accelerated primitives for common Digital Signal Processing tasks.

While many `zoo` modules are built around neural networks, the `dsp` module provides fundamental building blocks for processing raw signals like audio waveforms. These classes are designed to be integrated into larger pipelines, such as the pre-processing stage for an audio model or a real-time signal analysis application.

Every class in this module is a C++ wrapper around a custom, high-performance CUDA/cuFFT implementation, designed to keep the entire signal processing chain on the GPU.

---

## `Spectrogram`

Calculates the spectrogram of a raw audio waveform. This is the most fundamental feature representation for most audio-based AI tasks.

**Header:** `#include <xinfer/zoo/dsp/spectrogram.h>`

### Use Case: Feature Extraction for Speech Recognition

Before an audio clip can be fed into a speech recognition model, it must be converted into a spectrogram, which represents how the signal's frequency content changes over time. The `Spectrogram` class performs this entire conversion on the GPU.

```cpp
#include <xinfer/zoo/dsp/spectrogram.h>
#include <opencv2/opencv.hpp> // Used here for easy visualization/saving
#include <vector>

int main() {
    // 1. Configure the spectrogram parameters.
    xinfer::zoo::dsp::SpectrogramConfig config;
    config.sample_rate = 16000;
    config.n_fft = 400;       // FFT window size
    config.hop_length = 160;  // Step size between frames

    // 2. Initialize the spectrogram generator.
    xinfer::zoo::dsp::Spectrogram spectrogram_generator(config);

    // 3. Load a raw audio waveform.
    std::vector<float> waveform; // Assume this is loaded from a .wav file
    // ... load waveform data ...

    // 4. Process the waveform to get the spectrogram.
    //    The result is a cv::Mat where rows are frequency bins and columns are time frames.
    cv::Mat spectrogram = spectrogram_generator.process(waveform);

    // 5. Save the spectrogram as an image for visualization.
    cv::Mat db_spectrogram;
    cv::normalize(spectrogram, db_spectrogram, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imwrite("spectrogram.png", db_spectrogram);
}
```
**Config Struct:** `SpectrogramConfig`
**Input:** `std::vector<float>` containing the raw audio waveform.
**Output:** `cv::Mat` (type `CV_32F`) containing the log-power spectrogram.
**"F1 Car" Technology:** This class orchestrates a chain of CUDA kernels and **NVIDIA's `cuFFT` library**. It performs framing, windowing, FFT, and power-to-dB conversion entirely on the GPU, avoiding any slow CPU round-trips.

---

## `SignalFilter`

Applies a Finite Impulse Response (FIR) filter to a raw signal. This is a fundamental operation for noise reduction and signal separation.

**Header:** `#include <xinfer/zoo/dsp/signal_filter.h>`

### Use Case: Filtering a Noisy Signal

An engineer needs to apply a low-pass filter to an audio signal to remove high-frequency noise before further processing.

```cpp
#include <xinfer/zoo/dsp/signal_filter.h>
#include <vector>

int main() {
    // 1. Configure a low-pass filter.
    xinfer::zoo::dsp::SignalFilterConfig config;
    config.type = xinfer::zoo::dsp::FilterType::LOW_PASS;
    config.sample_rate = 44100;
    config.cutoff_freq1 = 3000.0f; // Cut off frequencies above 3kHz
    config.filter_length = 129;    // The length of the filter kernel

    // 2. Initialize the filter. This pre-computes the filter kernel on the GPU.
    xinfer::zoo::dsp::SignalFilter low_pass_filter(config);

    // 3. Load a noisy audio signal.
    std::vector<float> noisy_waveform;
    // ... load waveform data ...

    // 4. Process the signal.
    //    The filtering is done via high-speed convolution in the frequency domain.
    std::vector<float> filtered_waveform = low_pass_filter.process(noisy_waveform);
    
    // `filtered_waveform` now contains the clean signal.
}
```
**Config Struct:** `SignalFilterConfig` (can specify `LOW_PASS`, `HIGH_PASS`, or `BAND_PASS`).
**Input:** `std::vector<float>` of the input signal.
**Output:** `std::vector<float>` of the filtered signal.
**"F1 Car" Technology:** This class implements the **fast convolution** algorithm. It uses `cuFFT` to transform both the signal and the filter kernel into the frequency domain, performs a single, fast element-wise multiplication, and then uses an inverse FFT to get the final, filtered signal back in the time domain. This is orders of magnitude faster than a direct, time-domain convolution on the CPU for large signals.
