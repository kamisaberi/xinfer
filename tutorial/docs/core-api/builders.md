# API Reference: Engine Builders

The `xinfer::builders` module is the heart of the `xInfer` optimization pipeline. It provides a high-level, fluent C++ API that wraps the immense complexity of the NVIDIA TensorRT build process.

This is the toolkit you use to perform the crucial "Build Step" of the `xInfer` workflow, converting a standard model format like ONNX into a hyper-optimized, hardware-specific TensorRT engine.

While the `xinfer-cli` tool provides a command-line interface to this module, the C++ API gives you the full power and flexibility to integrate engine building directly into your applications or build automation scripts.

## Key Classes
- **`EngineBuilder`**: The main class for configuring and running the optimization process.
- **`ONNXExporter`**: A utility for converting trained `xTorch` models into the ONNX format.
- **`INT8Calibrator`**: An interface for providing data for INT8 quantization.

---

### `EngineBuilder`

This is the primary class you will interact with. It uses a "fluent" design pattern, allowing you to chain configuration calls together in a clean, readable way.

**Header:** `#include <xinfer/builders/engine_builder.h>`

#### **Core Workflow**

The `EngineBuilder` follows a simple, three-step process:
1.  Specify the input model (`.from_onnx()`).
2.  Configure the desired optimizations (`.with_fp16()`, `.with_int8()`, etc.).
3.  Execute the build process (`.build_and_save()`).

#### **Example: Building an FP16 Engine**

This is the most common use case. It takes an ONNX file and creates a TensorRT engine optimized for FP16 precision, which typically provides a **2x speedup** on modern NVIDIA GPUs with Tensor Cores.

```cpp
#include <xinfer/builders/engine_builder.h>
#include <iostream>

int main() {
    try {
        std::string onnx_path = "path/to/your/model.onnx";
        std::string engine_path = "my_model_fp16.engine";

        // 1. Create the builder
        xinfer::builders::EngineBuilder builder;

        // 2. Configure the build process by chaining calls
        builder.from_onnx(onnx_path)
               .with_fp16()
               .with_max_batch_size(16); // Specify the max batch size you'll use

        // 3. Execute the build and save the engine
        std::cout << "Building FP16 engine... this may take a few minutes.\n";
        builder.build_and_save(engine_path);
        std::cout << "Engine built successfully: " << engine_path << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error building engine: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
```

#### **API Overview**

- `EngineBuilder& from_onnx(const std::string& onnx_path)`
  Specifies the path to the input `.onnx` file.

- `EngineBuilder& with_fp16()`
  Enables FP16 precision mode. TensorRT will convert the model's layers to use half-precision floats where possible. This is highly recommended for all GPUs from the Turing architecture (sm_75) onwards.

- `EngineBuilder& with_int8(std::shared_ptr<INT8Calibrator> calibrator)`
  Enables INT8 precision mode for maximum performance (~4x+ speedup). This requires providing a calibrator object. See the [INT8 Quantization Guide](../guides/int8-quantization.md) for details.

- `EngineBuilder& with_max_batch_size(int batch_size)`
  Specifies the maximum batch size that the optimized engine will support. This allows TensorRT to tune its kernels for a specific batch size, which can improve performance.

- `void build_and_save(const std::string& output_engine_path)`
  Triggers the final build process and saves the resulting compiled engine to the specified path.

---

### `ONNXExporter`

This is a convenience utility for developers using the `xTorch` training library. It provides a seamless bridge between a trained `xTorch` model and the ONNX format needed by the `EngineBuilder`.

**Header:** `#include <xinfer/builders/onnx_exporter.h>`

#### **Example: Exporting an `xTorch` Model**

```cpp
#include <xinfer/builders/onnx_exporter.h>
#include <xtorch/models/resnet.h> // Assuming you have an xTorch model
#include <xtorch/util/serialization.h>

int main() {
    // 1. Instantiate and load your trained xTorch model
    auto model = std::make_shared<xt::models::ResNet18>();
    // xt::load(model, "my_trained_resnet.weights");
    model->eval();

    // 2. Define the input specification for your model
    xinfer::builders::InputSpec input_spec;
    input_spec.name = "input";
    input_spec.shape = {1, 3, 224, 224}; // Batch, Channels, Height, Width

    // 3. Call the export function
    std::string onnx_path = "resnet18.onnx";
    bool success = xinfer::builders::export_to_onnx(*model, {input_spec}, onnx_path);

    if (success) {
        std::cout << "xTorch model exported successfully to " << onnx_path << std::endl;
    }

    return 0;
}
```

---

### `build_engine_from_url` (Convenience Function)

This high-level function combines downloading and building into a single step, perfect for use in the `xinfer-cli` or for quickly grabbing a model from an online repository.

**Header:** `#include <xinfer/builders/engine_builder.h>`

#### **Example: Downloading and Building from the Web**

```cpp
#include <xinfer/builders/engine_builder.h>
#include <iostream>

int main() {
    // 1. Define the configuration for the download-and-build process
    xinfer::builders::BuildFromUrlConfig config;
    config.onnx_url = "https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx";
    config.output_engine_path = "resnet50_web.engine";
    config.use_fp16 = true;
    config.max_batch_size = 1;

    // 2. Call the function
    std::cout << "Downloading and building ResNet-50...\n";
    bool success = xinfer::builders::build_engine_from_url(config);

    if (success) {
        std::cout << "Engine created successfully at " << config.output_engine_path << std::endl;
    }

    return 0;
}
```
