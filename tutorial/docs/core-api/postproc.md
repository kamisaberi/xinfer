# API Reference: Post-processing Kernels

The `xinfer::postproc` module is one of the most powerful components of the `xInfer` toolkit. It provides a library of hyper-optimized, standalone CUDA kernels and functions for common post-processing tasks.

**The Philosophy:** The raw output of a neural network (the logits) can be enormous. Downloading this massive tensor from the GPU to the CPU just to perform filtering (like NMS) or decoding is incredibly inefficient. The `postproc` module provides functions that perform these operations **directly on the GPU**, ensuring that only the final, small, human-readable result is ever transferred back to the CPU.

This is a key strategy for eliminating performance bottlenecks in end-to-end inference pipelines.

---

### **Detection: `postproc::detection`**

**Header:** `#include <xinfer/postproc/detection.h>`

This module is essential for any object detection task.

#### `std::vector<int> nms(...)`

Performs high-performance Non-Maximum Suppression on the GPU.

```cpp
#include <xinfer/postproc/detection.h>

std::vector<int> nms(
    const core::Tensor& decoded_boxes,    // GPU tensor of shape [N, 4] with [x1, y1, x2, y2]
    const core::Tensor& decoded_scores,   // GPU tensor of shape [N] with confidence scores
    float nms_iou_threshold             // The IoU threshold for suppression
);
```

- **Description:** This is a custom CUDA implementation of NMS. It takes the decoded boxes and scores (which are still on the GPU), sorts them, and efficiently suppresses overlapping boxes.
- **Returns:** A `std::vector<int>` containing the *indices* of the boxes that survived the suppression. This list is typically very small, making the final data transfer to the CPU minimal.
- **Why it's an "F1 Car":** A standard NMS implementation on the CPU would require downloading all `N` boxes and scores first. The `xInfer` version is **10x-20x faster** by keeping the entire process on the GPU.

---

### **Detection Decoders: `postproc::yolo`**

**Header:** `#include <xinfer/postproc/yolo_decoder.h>`

This module provides decoders for specific model families.

#### `void decode(...)`

Parses the complex, raw output tensor of a YOLO-family model.

```cpp
#include <xinfer/postproc/yolo_decoder.h>

void decode(
    const core::Tensor& raw_output,         // Raw GPU tensor from the model
    float confidence_threshold,             // Filter out low-confidence boxes on the GPU
    core::Tensor& out_boxes,                // Pre-allocated GPU tensor for boxes
    core::Tensor& out_scores,               // Pre-allocated GPU tensor for scores
    core::Tensor& out_classes               // Pre-allocated GPU tensor for class IDs
);
```

- **Description:** A YOLO model typically outputs a single large tensor of shape `[1, 4 + NumClasses, NumPriors]`. This fused CUDA kernel efficiently processes this tensor in parallel. For each of the thousands of priors, it finds the class with the highest score, checks it against the confidence threshold, converts the box coordinates, and writes the filtered results to the output tensors.
- **Why it's an "F1 Car":** It avoids a massive GPU-to-CPU transfer of the raw logits. All filtering and decoding happens on the GPU, which is massively parallel and perfectly suited for this task.

---

### **Segmentation: `postproc::segmentation`**

**Header:** `#include <xinfer/postproc/segmentation.h>`

Provides essential tools for semantic and instance segmentation.

#### `void argmax(...)`

Performs a fast, parallel ArgMax operation across the channel dimension of a segmentation model's output.

```cpp
#include <xinfer/postproc/segmentation.h>

void argmax(
    const core::Tensor& logits,       // GPU tensor of shape [1, NumClasses, H, W]
    core::Tensor& output_mask         // Pre-allocated GPU tensor of shape [1, H, W] (INT32)
);
```

- **Description:** This CUDA kernel finds the index of the maximum value for each pixel, converting the probability map (logits) into a final class ID map.
- **Why it's an "F1 Car":** The `logits` tensor for a high-resolution image can be huge (e.g., 21 classes * 1024 * 1024 pixels > 88MB). Downloading this to the CPU is very slow. This kernel produces a much smaller integer mask (`~4MB`) directly on the GPU.

#### `cv::Mat argmax_to_mat(...)`

A convenience function that wraps `argmax` and handles the download to an OpenCV Mat.

```cpp
#include <xinfer/postproc/segmentation.h>

cv::Mat argmax_to_mat(const core::Tensor& logits);
```

---

### **Sequence Decoding: `postproc::ctc`**

**Header:** `#include <xinfer/postproc/ctc_decoder.h>`

Provides decoders for sequence models like speech recognition and OCR.

#### `std::pair<std::string, float> decode(...)`

Performs a greedy Connectionist Temporal Classification (CTC) decode on the GPU.

```cpp
#include <xinfer/postproc/ctc_decoder.h>

std::pair<std::string, float> decode(
    const core::Tensor& logits,                 // GPU tensor of shape [1, Timesteps, NumClasses]
    const std::vector<std::string>& character_map // The vocabulary mapping
);
```

- **Description:** This function launches a CUDA kernel to find the most likely character (the `argmax`) at each timestep in the sequence. It then performs the final CTC collapsing logic (removing blanks and duplicates) on the CPU to produce the final string.
- **Why it's an "F1 Car":** It performs the most parallelizable part of the decoding (the per-timestep `argmax`) on the GPU, avoiding the download of the large `logits` tensor.

---

### **Generative AI: `postproc::diffusion`**

**Header:** `#include <xinfer/postproc/diffusion_sampler.h>`

Provides the core building block for diffusion model pipelines.

#### `void sampling_step(...)`

Executes one full step of the DDPM denoising process in a single fused kernel.

```cpp
#include <xinfer/postproc/diffusion_sampler.h>

void sampling_step(
    core::Tensor& img,                  // The noisy image (modified in-place)
    const core::Tensor& predicted_noise,  // The U-Net output
    const core::Tensor& random_noise,     // Random noise for this step
    const core::Tensor& alphas,           // Scheduler constants
    const core::Tensor& alphas_cumprod,   // Scheduler constants
    const core::Tensor& betas,            // Scheduler constants
    int timestep,
    cudaStream_t stream
);
```

- **Description:** This custom CUDA kernel implements the entire, multi-part DDPM sampling equation.
- **Why it's an "F1 Car":** A standard framework would execute this equation as a sequence of 5-6 separate math operations, each launching its own CUDA kernel. This is extremely inefficient inside the hot loop of the diffusion process. The `xInfer` fused kernel executes the entire equation in a single launch, dramatically reducing overhead.
