# Core Toolkit API Reference

Welcome to the API reference for the `xInfer` Core Toolkit.

While the **[Model Zoo API](../zoo-api/index.md)** provides high-level, pre-packaged solutions for common tasks, the Core Toolkit is for developers who need maximum control and flexibility. These are the powerful, low-level building blocks that the `zoo` itself is built upon.

You should use the Core Toolkit when you are:
- Building a hyper-optimized pipeline for a custom model architecture not found in the `zoo`.
- Integrating `xInfer` into a complex, existing C++ application with custom data structures.
- Implementing advanced asynchronous workflows with multiple CUDA streams.
- Creating your own high-level `zoo`-like abstractions for a specific domain.

---

## The Core Modules

The toolkit is divided into four logical modules, each responsible for a specific part of the high-performance inference pipeline.

### **1. `xinfer::core` - The Inference Runtime**

This is the heart of the `xInfer` runtime. These classes are responsible for loading and executing your pre-built, optimized TensorRT engines.

- **`Tensor`**: A lightweight, safe C++ wrapper for managing GPU memory. It's the primary data structure for all I/O in `xInfer`.
- **`InferenceEngine`**: The workhorse class that loads a `.engine` file and provides simple, powerful methods for running synchronous and asynchronous inference.

➡️ **[Full API Reference for `core`](./engine.md)**

### **2. `xinfer::builders` - The Optimization Toolkit**

This module provides the "factory" tools for performing the crucial, offline "Build Step." You use these classes to convert a standard model format like ONNX into a hyper-optimized TensorRT engine.

- **`EngineBuilder`**: A fluent API that automates the entire TensorRT build process, including enabling optimizations like FP16 and INT8.
- **`ONNXExporter`**: A convenience utility to bridge the gap from a trained `xTorch` model to the ONNX format.
- **`INT8Calibrator`**: The interface for providing calibration data for INT8 quantization.

➡️ **[Full API Reference for `builders`](./builders.md)**

### **3. `xinfer::preproc` - GPU-Accelerated Pre-processing**

This module contains a library of unique, high-performance CUDA kernels designed to eliminate CPU bottlenecks during data preparation.

- **`ImageProcessor`**: A powerful class that can perform an entire image pre-processing pipeline (`Resize -> Pad -> Normalize -> HWC to CHW`) in a single, fused CUDA kernel.
- **`AudioProcessor`**: A fused pipeline for converting raw audio waveforms into mel spectrograms, using `cuFFT` for maximum performance.

➡️ **[Full API Reference for `preproc`](./preproc.md)**

### **4. `xinfer::postproc` - GPU-Accelerated Post-processing**

This module provides custom CUDA kernels to accelerate the most common post-processing tasks, avoiding slow GPU-to-CPU data transfers of large, raw model outputs.

- **`detection::nms`**: A hyper-performant, GPU-based implementation of Non-Maximum Suppression for object detection.
- **`yolo_decoder::decode`**: A fused kernel for parsing the complex output of YOLO-family models.
- **`segmentation::argmax`**: A GPU-based kernel for converting raw segmentation logits into a final class mask.
- **`ctc::decode`**: A GPU-based kernel for decoding the output of speech recognition and OCR models.

➡️ **[Full API Reference for `postproc`](./postproc.md)**

---

### **Next Steps**

To see how these core components are used in practice, check out the **[Building Custom Pipelines](../guides/custom-pipelines.md)** guide.