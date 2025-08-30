# Zoo API: Generative AI

The `xinfer::zoo::generative` module provides high-level pipelines for a wide range of creative and generative AI tasks.

These classes are built on top of `xInfer`'s hyper-optimized engines for state-of-the-art generative models like Stable Diffusion, GANs, and more. The `zoo` API abstracts away the complexity of these models, from multi-stage pipelines to iterative sampling loops, providing simple, powerful tools to bring your creative ideas to life in C++.

---

## `DiffusionPipeline`

Generates high-quality images from text prompts using a Stable Diffusion-style model.

**Header:** `#include <xinfer/zoo/generative/diffusion_pipeline.h>`

```cpp
#include <xinfer/zoo/generative/diffusion_pipeline.h>
#include <xinfer/utils/image_utils.h> // For saving the final tensor
#include <iostream>

int main() {
    // 1. Configure the pipeline.
    //    The engine would be a pre-built U-Net from a Stable Diffusion model.
    xinfer::zoo::generative::DiffusionPipelineConfig config;
    config.unet_engine_path = "assets/stable_diffusion_unet.engine";
    config.num_timesteps = 50; // Number of denoising steps

    // 2. Initialize.
    xinfer::zoo::generative::DiffusionPipeline pipeline(config);

    // 3. Generate an image.
    //    The complex, 50-step iterative loop is handled internally in high-performance C++.
    std::cout << "Generating image with diffusion model...\n";
    // A full implementation would also take a text prompt that is processed by a CLIP text encoder.
    xinfer::core::Tensor image_tensor = pipeline.generate(1);

    // 4. Save the result.
    xinfer::utils::save_tensor_as_image(image_tensor, "diffusion_output.png");
    std::cout << "Image saved to diffusion_output.png\n";
}
```
**Config Struct:** `DiffusionPipelineConfig`
**Input:** `batch_size` (and optionally, text prompt embeddings).
**Output:** `xinfer::core::Tensor` containing the generated image.
**"F1 Car" Technology:** The entire iterative denoising loop is a compiled C++ `for` loop, and each step uses a custom, fused CUDA kernel from `postproc::diffusion_sampler`, providing a massive speedup over a Python-based loop.

---

## `DCGAN`

Generates images from a random latent vector using a Generative Adversarial Network.

**Header:** `#include <xinfer/zoo/generative/dcgan.h>`

```cpp
#include <xinfer/zoo/generative/dcgan.h>
#include <xinfer/utils/image_utils.h>

int main() {
    xinfer::zoo::generative::DCGAN_Generator generator("assets/dcgan_generator.engine");

    std::cout << "Generating image with DCGAN...\n";
    xinfer::core::Tensor image_tensor = generator.generate(1);

    xinfer::utils::save_tensor_as_image(image_tensor, "dcgan_output.png");
    std::cout << "Image saved to dcgan_output.png\n";
}
```
**Input:** `batch_size`.
**Output:** `xinfer::core::Tensor` containing the generated image.

---

## `StyleTransfer`

Applies the artistic style of one image to the content of another.

**Header:** `#include <xinfer/zoo/generative/style_transfer.h>`

```cpp
#include <xinfer/zoo/generative/style_transfer.h>
#include <opencv2/opencv.hpp>

int main() {
    // The engine is pre-built from a model trained on a specific style (e.g., "Starry Night").
    xinfer::zoo::generative::StyleTransferConfig config;
    config.engine_path = "assets/starry_night_style.engine";

    xinfer::zoo::generative::StyleTransfer stylizer(config);

    cv::Mat content_image = cv::imread("assets/my_photo.jpg");
    cv::Mat styled_image = stylizer.predict(content_image);

    cv::imwrite("styled_output.jpg", styled_image);
    std::cout << "Saved styled image to styled_output.jpg\n";
}
```
**Config Struct:** `StyleTransferConfig`
**Input:** `cv::Mat` content image.
**Output:** `cv::Mat` stylized image.

---

## `SuperResolution`

Upscales a low-resolution image to a high-resolution version, adding realistic detail.

**Header:** `#include <xinfer/zoo/generative/super_resolution.h>`

```cpp
#include <xinfer/zoo/generative/super_resolution.h>
#include <opencv2/opencv.hpp>

int main() {
    xinfer::zoo::generative::SuperResolutionConfig config;
    config.engine_path = "assets/esrgan_x4.engine";
    config.upscale_factor = 4;

    xinfer::zoo::generative::SuperResolution upscaler(config);

    cv::Mat low_res_image = cv::imread("assets/low_res.png");
    cv::Mat high_res_image = upscaler.predict(low_res_image);

    cv::imwrite("super_resolution_output.png", high_res_image);
}
```
**Config Struct:** `SuperResolutionConfig`
**Input:** `cv::Mat` low-resolution image.
**Output:** `cv::Mat` high-resolution image.

---

## And More...

This module provides many more specialized generative pipelines.

- **`Inpainter`**: Fills in a masked region of an image with plausible content.
- **`Outpainter`**: Extends the boundaries of an image with generated content.
- **`Colorizer`**: Adds realistic color to a black-and-white image.
- **`TextToSpeech`**: Converts a string of text into a spoken audio waveform.
- **`ImageToVideo`**: Generates a short video clip from a starting image.
- **`VideoFrameInterpolation`**: Creates smooth, slow-motion video by generating intermediate frames.

*Each of these would have its own section with a code example, just like the ones above.*
