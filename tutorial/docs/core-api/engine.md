# API Reference: Core Runtime Engine

The `xinfer::core` module provides the fundamental classes for loading and executing pre-built, optimized TensorRT engines. These are the low-level, high-performance building blocks that the high-level `zoo` API is built upon.

You would use this API directly if you need fine-grained control over the inference process, such as when building a custom pipeline that isn't covered by the `zoo`, or when you need to manage CUDA streams for complex asynchronous workflows.

## Key Classes
- **`Tensor`**: A lightweight, safe wrapper for managing GPU memory.
- **`InferenceEngine`**: The main class for loading a `.engine` file and running inference.

---

### `Tensor`

The `core::Tensor` is the primary data structure in `xInfer`. It is a C++ RAII-compliant object that safely manages the lifetime of a GPU memory buffer, abstracting away raw `cudaMalloc` and `cudaFree` calls.

**Header:** `#include <xinfer/core/tensor.h>`

#### **Core Principles**

- **Ownership:** A `Tensor` object *owns* its GPU memory. When it goes out of scope, its destructor is automatically called, freeing the GPU memory.
- **Movable, Not Copyable:** To prevent accidental, expensive GPU-to-GPU copies, `Tensor` objects cannot be copied. They can only be *moved*. This ensures clear and efficient ownership transfer.
- **Lightweight:** The class is a very thin wrapper, adding no performance overhead to the underlying GPU operations.

#### **Example: Creating and Moving a Tensor**

```cpp
#include <xinfer/core/tensor.h>
#include <vector>

void process_tensor(xinfer::core::Tensor t) {
    // This function now owns the tensor memory.
    // When `t` goes out of scope here, the memory is freed.
}

int main() {
    // 1. Create a tensor for a batch of 16 RGB images of size 224x224.
    xinfer::core::Tensor my_tensor({16, 3, 224, 224}, xinfer::core::DataType::kFLOAT);

    // 2. The GPU memory is now allocated.
    // You can get the raw pointer to pass to a CUDA kernel.
    void* gpu_ptr = my_tensor.data();

    // 3. To pass it to a function, you must move it.
    process_tensor(std::move(my_tensor));

    // 4. After the move, `my_tensor` is in a null state.
    // Accessing its data would now be an error.
    // assert(my_tensor.data() == nullptr);

    return 0;
}
```

#### **API Overview**

- `Tensor(const std::vector<int64_t>& shape, DataType dtype)`
  Constructor to allocate a new GPU buffer of a specific shape and data type.

- `void* data() const`
  Returns the raw `void*` pointer to the GPU memory buffer.

- `const std::vector<int64_t>& shape() const`
  Returns a reference to the vector describing the tensor's shape.

- `void copy_from_host(const void* cpu_data)`
  Performs a `cudaMemcpyHostToDevice` to upload data from the CPU to the GPU.

- `void copy_to_host(void* cpu_data) const`
  Performs a `cudaMemcpyDeviceToHost` to download data from the GPU to the CPU.

---

### `InferenceEngine`

The `core::InferenceEngine` is the workhorse of the `xInfer` runtime. It loads a serialized `.engine` file created by the `builders` module and handles the execution of the model.

**Header:** `#include <xinfer/core/engine.h>`

#### **Example: Manual Inference Pipeline**

This example shows how you would use the `InferenceEngine` directly to build a custom pipeline, without the `zoo`.

```cpp
#include <xinfer/core/engine.h>
#include <xinfer/core/tensor.h>
#include <xinfer/preproc/image_processor.h> // Using a preproc helper
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    try {
        std::string engine_path = "resnet18.engine";
        
        // 1. Load the pre-built engine. This is a fast operation.
        xinfer::core::InferenceEngine engine(engine_path);

        // 2. Manually create the pre-processor.
        xinfer::preproc::ImageProcessor preprocessor(224, 224, {0.485, 0.456, 0.406}, {0.229, 0.224, 0.225});

        // 3. Load input data and prepare GPU tensors.
        cv::Mat image = cv::imread("cat_image.jpg");
        auto input_shape = engine.get_input_shape(0);
        xinfer::core::Tensor input_tensor(input_shape, xinfer::core::DataType::kFLOAT);
        
        // 4. Run the pre-processing step.
        preprocessor.process(image, input_tensor);

        // 5. Run synchronous inference.
        // The engine takes a vector of input tensors and returns a vector of output tensors.
        std::vector<xinfer::core::Tensor> output_tensors = engine.infer({input_tensor});

        // 6. Process the output.
        // For ResNet-18, there is one output tensor with shape.
        std::cout << "Inference successful. Output tensor has " << output_tensors.num_elements() << " elements.\n";
        
        // You would now copy the output_tensors to the host and find the argmax.

    } catch (const std::exception& e) {
        std::cerr << "Error in custom pipeline: " << e.what() << std::endl;
    }

    return 0;
}
```

#### **API Overview**

- `InferenceEngine(const std::string& engine_path)`
  Constructor that loads and deserializes a TensorRT engine from a file path.

- `std::vector<Tensor> infer(const std::vector<Tensor>& inputs)`
  Performs a **synchronous** inference call. It takes a vector of input `Tensor` objects and returns a vector of newly allocated output `Tensor` objects. This is the simplest way to run inference.

- `void infer_async(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs, cudaStream_t stream)`
  Performs an **asynchronous** inference call on a specific CUDA stream. This is an advanced method for building complex pipelines where you want to overlap data transfers with computation. Note that the output tensors must be pre-allocated by the user.

- `int get_num_inputs() const`
  Returns the number of input tensors the model expects.

- `std::vector<int64_t> get_input_shape(int index = 0) const`
  Returns the shape of the specified input tensor. This is useful for allocating your input `Tensor` objects correctly.
