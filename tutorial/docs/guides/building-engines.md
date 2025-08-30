# How-To Guide: Building TensorRT Engines

The core of the `xInfer` workflow is the **TensorRT engine**. An engine is a hyper-optimized version of your neural network, compiled and tuned for a specific GPU architecture and precision. This ahead-of-time compilation is what gives `xInfer` its incredible speed.

This guide will walk you through the two primary methods for building an engine using the `xInfer` toolkit:

1.  **The Easy Way:** Using the `xinfer-cli` command-line tool.
2.  **The Powerful Way:** Using the `xinfer::builders::EngineBuilder` C++ API.

**The Goal:** To convert a standard `.onnx` model file into a high-performance `.engine` file that can be loaded by the `xInfer` runtime.

---

## Prerequisites

Before you begin, you need a trained model saved in the **ONNX (Open Neural Network Exchange)** format. ONNX is a universal format that `xInfer`'s builder uses as its starting point.

### Exporting from PyTorch/xTorch

If you have a model trained in PyTorch or `xTorch`, you can export it to ONNX easily.

**Python (`PyTorch`):**
```python
import torch

# Load your trained model
model = YourModelClass()
model.load_state_dict(torch.load("my_model.pth"))
model.eval()

# Create a dummy input with the correct shape and batch size
dummy_input = torch.randn(1, 3, 224, 224)

# Export the model
torch.onnx.export(
    model,
    dummy_input,
    "my_model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
```

**C++ (`xTorch`):**
You can use the `xinfer::builders::export_to_onnx` utility for this. See the **[Core API: Builders](../core-api/builders.md)** reference for an example.

---

## Method 1: The `xinfer-cli` (Recommended)

The command-line interface is the simplest and most direct way to build your engine. It's a powerful tool for quick experiments and for integrating into build scripts.

Assume you have `my_model.onnx` ready.

### Basic FP32 Engine

This creates a standard, full-precision engine. It's a good starting point for verifying correctness.

```bash
# General Syntax:
# xinfer-cli --build --onnx <input.onnx> --save_engine <output.engine>

xinfer-cli --build --onnx my_model.onnx --save_engine my_model_fp32.engine
```

### **FP16 Engine (High Performance)**

This is the most common and highly recommended optimization. It enables FP16 precision, which can provide a **~2x speedup** on modern GPUs (Turing architecture and newer) by leveraging their Tensor Cores.

```bash
xinfer-cli --build \
    --onnx my_model.onnx \
    --save_engine my_model_fp16.engine \
    --fp16
```

### INT8 Engine (Maximum Performance)

This provides the highest possible performance (**~4x+ speedup**) but requires a "calibration" step. You must provide a small, representative dataset of images for TensorRT to analyze the model's activation distributions.

For more details, see the **[INT8 Quantization](./int8-quantization.md)** guide.

```bash
# This is a conceptual example. The CLI would need a way to specify a calibrator.
xinfer-cli --build \
    --onnx my_model.onnx \
    --save_engine my_model_int8.engine \
    --int8 \
    --calibration_data /path/to/calibration/images
```

### Specifying Batch Size

You can also specify the maximum batch size the engine should be optimized for.

```bash
xinfer-cli --build \
    --onnx my_model.onnx \
    --save_engine my_model_fp16_b16.engine \
    --fp16 \
    --batch 16
```

---

## Method 2: The C++ `EngineBuilder` API

For more advanced use cases, such as building engines programmatically as part of a larger application, you can use the `EngineBuilder` C++ class directly. This gives you the full power and flexibility of the toolkit.

### Example C++ Build Script

This C++ program does the exact same thing as the `xinfer-cli` FP16 example above.

**File: `build_my_engine.cpp`**
```cpp
#include <xinfer/builders/engine_builder.h>
#include <iostream>
#include <stdexcept>

int main() {
    std::string onnx_path = "my_model.onnx";
    std::string engine_path = "my_model_fp16_cpp.engine";

    std::cout << "Building FP16 engine from: " << onnx_path << std::endl;
    std::cout << "This may take a few minutes...\n";

    try {
        // 1. Create the builder
        xinfer::builders::EngineBuilder builder;

        // 2. Configure the build process using the fluent API
        builder.from_onnx(onnx_path)
               .with_fp16()
               .with_max_batch_size(16);

        // 3. Execute the build and save the final engine
        builder.build_and_save(engine_path);
        
        std::cout << "Engine built successfully and saved to: " << engine_path << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error during engine build: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
```

To compile and run this, you would link it against the `xinfer` library, just like the `xinfer_example` target in the main `CMakeLists.txt`.

---

## Next Steps

Once you have successfully built your `.engine` file, you are ready for the fun part: running high-speed inference!

- **See the [🚀 Quickstart](./quickstart.md)** for a guide on how to load your new engine with the `zoo` API.
- **See the [Core API: Engine](./core-api/engine.md)** documentation to learn how to load and run it manually for custom pipelines.
