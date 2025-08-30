# API Reference: Pre-processing Kernels

The `xinfer::preproc` module is dedicated to solving one of the most common bottlenecks in real-world AI applications: **data pre-processing**.

**The Philosophy:** Before a model can be run, input data (like an image from a camera or an audio clip) must be transformed into a correctly formatted tensor. Performing these steps on the CPU—resizing, normalizing, converting data types, and changing memory layouts—and then transferring the result to the GPU is incredibly inefficient.

The `preproc` module provides high-performance, fused CUDA kernels that perform this entire pipeline **directly on the GPU**. This minimizes CPU-GPU data transfers and leverages the GPU's massive parallelism for a significant speedup.

---

### **Image Processing: `preproc::ImageProcessor`**

**Header:** `#include <xinfer/preproc/image_processor.h>`

This is the universal tool for all image-based tasks in `xInfer`. It is designed to take a standard `cv::Mat` from the CPU and efficiently produce a model-ready tensor on the GPU.

#### **Core Feature: The Fused Pipeline**

A single call to `process()` executes a monolithic CUDA kernel that performs the following steps in one operation:
1.  **Upload:** Copies the `cv::Mat` data from the CPU to the GPU.
2.  **Resize:** Resizes the image to the model's required input dimensions using bilinear interpolation.
3.  **(Optional) Letterbox/Pad:** Adds padding to maintain the aspect ratio, a common requirement for models like YOLO.
4.  **Layout Conversion:** Converts the image from OpenCV's interleaved `HWC` (Height, Width, Channel) layout to the `CHW` (Channel, Height, Width) layout required by deep learning models.
5.  **Normalization:** Converts the 8-bit integer pixel values to 32-bit floats and applies the specified normalization formula: `(pixel / 255.0 - mean) / std`.
6.  **Output:** Writes the final, model-ready tensor directly to the destination GPU buffer.

This fused approach is **5x-10x faster** than a traditional pipeline using a chain of OpenCV calls on the CPU.

#### **Example: Preparing an Image for a Classifier**

```cpp
#include <xinfer/preproc/image_processor.h>
#include <xinfer/core/tensor.h>
#include <opencv2/opencv.hpp>
#include <vector>

int main() {
    // 1. Configure the pre-processor for a standard ImageNet model.
    int input_width = 224;
    int input_height = 224;
    std::vector<float> mean = {0.485, 0.456, 0.406};
    std::vector<float> std = {0.229, 0.224, 0.225};

    xinfer::preproc::ImageProcessor preprocessor(input_width, input_height, mean, std);

    // 2. Load an image from disk.
    cv::Mat image = cv::imread("my_image.jpg");

    // 3. Create a destination tensor on the GPU.
    xinfer::core::Tensor input_tensor({1, 3, input_height, input_width}, xinfer::core::DataType::kFLOAT);

    // 4. Run the entire pre-processing pipeline.
    preprocessor.process(image, input_tensor);

    // `input_tensor` is now ready to be passed to an InferenceEngine.
    
    return 0;
}
```

#### **API Overview**

- `ImageProcessor(int width, int height, const std::vector<float>& mean, const std::vector<float>& std)`
  Constructor for a standard resizing and normalization pipeline.

- `ImageProcessor(int width, int height, bool letterbox = false)`
  A convenience constructor for models like YOLO that require letterbox padding and simple `[0, 1]` scaling instead of mean/std normalization.

- `void process(const cv::Mat& cpu_image, core::Tensor& output_tensor)`
  Executes the full pipeline on a CPU-based `cv::Mat`.

- `void process(const core::Tensor& gpu_image, core::Tensor& output_tensor)`
  An advanced overload that processes an image that is *already* on the GPU, avoiding the CPU-GPU upload step entirely.

---

### **Audio Processing: `preproc::AudioProcessor`**

**Header:** `#include <xinfer/preproc/audio_processor.h>`

This class is the universal entry point for all audio-based tasks, such as speech recognition and audio classification. It efficiently converts a raw audio waveform into a mel spectrogram.

#### **Core Feature: The Fused Spectrogram Pipeline**

A single call to `process()` uses a chain of custom CUDA kernels and the NVIDIA `cuFFT` library to perform the entire spectrogram generation pipeline on the GPU:

1.  **Upload:** Copies the raw audio waveform from the CPU to the GPU.
2.  **Framing & Windowing:** A CUDA kernel splits the audio into overlapping frames and applies a windowing function (e.g., Hann).
3.  **FFT:** The NVIDIA `cuFFT` library, the fastest available FFT implementation, is used to compute the Short-Time Fourier Transform (STFT).
4.  **Power & Mel Scaling:** A final fused kernel calculates the power spectrum, applies a Mel filterbank to the frequencies, and converts the result to a log scale.

#### **Example: Creating a Mel Spectrogram**

```cpp
#include <xinfer/preproc/audio_processor.h>
#include <xinfer/core/tensor.h>
#include <vector>

int main() {
    // 1. Configure the audio processor.
    xinfer::preproc::AudioProcessorConfig config;
    config.sample_rate = 16000;
    config.n_fft = 400;
    config.hop_length = 160;
    config.n_mels = 80;

    xinfer::preproc::AudioProcessor preprocessor(config);

    // 2. Load a raw audio waveform (e.g., from a .wav file).
    std::vector<float> waveform; // Assume this is loaded with audio data

    // 3. Create a destination tensor on the GPU.
    // The output shape depends on the waveform length and config.
    int n_frames = (waveform.size() - config.n_fft) / config.hop_length + 1;
    xinfer::core::Tensor mel_spectrogram({1, config.n_mels, n_frames}, xinfer::core::DataType::kFLOAT);

    // 4. Run the entire spectrogram generation pipeline.
    preprocessor.process(waveform, mel_spectrogram);

    // `mel_spectrogram` is now ready to be passed to a model like Whisper.

    return 0;
}
```
