# How-To Guide: INT8 Quantization for Maximum Performance

This guide is for advanced users who want to squeeze every last drop of performance out of their NVIDIA GPU. **INT8 quantization** is a powerful technique that can provide a **2x or greater speedup** on top of an already-optimized FP16 model, resulting in the absolute fastest inference speeds possible.

`xInfer` provides a streamlined API for performing INT8 quantization, but it's important to understand the concepts first.

**What You'll Learn:**
1.  What INT8 quantization is and why it's so fast.
2.  The concept of "calibration" and why it's necessary.
3.  How to use the `xinfer::builders` API to create a high-performance INT8 engine.

---

## What is INT8 Quantization?

By default, neural networks perform their calculations using 32-bit floating-point numbers (`FP32`). INT8 quantization is a process that converts the model's weights and activations to use 8-bit integers (`INT8`) instead.

**Why is this so much faster?**

1.  **Reduced Memory Bandwidth:** An INT8 value takes up **4 times less memory** than an FP32 value. This means the GPU can move data from its slow VRAM to its fast compute cores much more quickly. For many models, memory bandwidth is the primary bottleneck, so this provides a huge speedup.
2.  **Specialized Hardware (Tensor Cores):** Modern NVIDIA GPUs (Turing architecture and newer) have dedicated hardware cores called **Tensor Cores**. These cores are specifically designed to perform integer matrix math at an incredible rate. Running a model in INT8 mode allows TensorRT to take full advantage of this specialized hardware, dramatically increasing the number of operations the GPU can perform per second (TOPS).

**The Catch: The Loss of Precision**
Converting from a 32-bit float to an 8-bit integer is a "lossy" conversion. The challenge is to do this conversion in a smart way that minimizes the impact on the model's final accuracy. This is where **calibration** comes in.

---

## The Calibration Process

TensorRT uses a process called **Post-Training Quantization (PTQ)**. To figure out the best way to convert the floating-point numbers to integers, it needs to look at the typical range of values that flow through the network.

This is what the **calibrator** does. You provide a small, representative sample of your validation dataset (usually 100-500 images). TensorRT then runs the FP32 version of the model on this sample data and observes the distribution of activation values at each layer.

Based on this observation, it calculates the optimal "scaling factor" for each layer to map the floating-point range to the `[-128, 127]` integer range with the minimum possible loss of information.

---

## Using `xInfer` for INT8 Quantization

`xInfer` makes this complex process simple. You need to provide an implementation of the `INT8Calibrator` interface. For convenience, `xInfer` provides a ready-made `DataLoaderCalibrator` that works directly with an `xTorch` data loader.

### Example: Building an INT8 Classifier Engine

This example shows how to take a trained `xTorch` ResNet-18 and build a hyper-performant INT8 engine.

**File: `build_int8_engine.cpp`**
```cpp
#include <xinfer/builders/engine_builder.h>
#include <xinfer/builders/calibrator.h> // The Calibrator interface
#include <xtorch/xtorch.h> // We need xTorch for the data loader
#include <iostream>
#include <memory>

int main() {
    try {
        std::string onnx_path = "resnet18.onnx"; // Assume this was exported from xTorch
        std::string engine_path = "resnet18_int8.engine";

        // --- Step 1: Prepare the Calibration Dataloader ---
        // We need a small, representative sample of our validation data.
        auto calibration_dataset = xt::datasets::ImageFolder(
            "/path/to/your/calibration_images/", // A folder with ~500 images
            xt::transforms::Compose({
                std::make_shared<xt::transforms::image::Resize>({224, 224}),
                std::make_shared<xt::transforms::general::Normalize>(
                    std::vector<float>{0.485, 0.456, 0.406},
                    std::vector<float>{0.229, 0.224, 0.225}
                )
            })
        );
        xt::dataloaders::ExtendedDataLoader calib_loader(calibration_dataset, 32, false);

        // --- Step 2: Create the Calibrator Object ---
        // We wrap our xTorch dataloader with the xInfer calibrator.
        auto calibrator = std::make_shared<xinfer::builders::DataLoaderCalibrator>(calib_loader);

        // --- Step 3: Configure and Run the Engine Builder ---
        std::cout << "Building INT8 engine... This will take several minutes as it runs calibration.\n";
        xinfer::builders::EngineBuilder builder;

        builder.from_onnx(onnx_path)
               .with_int8(calibrator) // Pass the calibrator to enable INT8 mode
               .with_max_batch_size(32);

        builder.build_and_save(engine_path);
        
        std::cout << "INT8 engine built successfully: " << engine_path << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error building INT8 engine: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
```

### **When to Use INT8**

- **Best For:** CNNs and Transformers with standard layers like `Conv`, `Linear`, and `ReLU`.
- **Latency-Critical Applications:** If you need the absolute lowest possible latency (e.g., in robotics or real-time video analysis), INT8 is the best choice.
- **High-Throughput Services:** If you are running a model in a data center and want to maximize the number of inferences per second per dollar, INT8 is the most cost-effective solution.

### **When to Be Cautious**

- **Accuracy:** Always validate the accuracy of your INT8 model against your FP32 baseline. For some models, especially those with very wide and unusual activation ranges, there can be a small drop in accuracy.
- **Exotic Layers:** Models with many non-standard or custom layers may not quantize as effectively.

By following this guide, you can leverage the `xInfer` builders to unlock the full potential of your NVIDIA hardware, creating inference engines that are not just fast, but state-of-the-art in their performance and efficiency.
