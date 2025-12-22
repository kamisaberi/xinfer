# NVIDIA TensorRT Backend for xInfer

This backend provides industry-standard performance for NVIDIA GPUs. It supports **dynamic shapes**, **asynchronous execution**, and **Jetson DLA** offloading.

## 🛠️ Prerequisites

1.  **NVIDIA Drivers:** Latest GPU drivers installed.
2.  **CUDA Toolkit:** Version 11.8 or 12.x.
3.  **TensorRT:** Version 8.6 or 10.x.

## ⚙️ Model Compilation

TensorRT engines are specific to the exact GPU they were compiled on. You cannot share an `.engine` file between an RTX 3090 and an RTX 4090.

**Using xInfer CLI:**
```bash
xinfer-cli compile --target nv-trt \
                   --onnx model.onnx \
                   --output model.engine \
                   --precision fp16
```

## 💻 C++ Usage

```cpp
xinfer::zoo::vision::DetectorConfig config;
config.target = xinfer::Target::NVIDIA_TRT;
config.model_path = "models/yolo_v8.engine";

// Optional: Run on DLA Core 0 (Jetson Orin Only)
config.vendor_params = { "DLA=0" };
```

## ⚠️ Performance Note

*   **Warmup:** The first inference call may trigger CUDA lazy loading. It is recommended to run one dummy inference before starting your real-time loop.
*   **Memory:** For maximum performance, allocate your input tensors using `cudaMalloc` (or `xinfer::core::Tensor::alloc_device`). Passing CPU memory to `predict()` forces a slow PCIe copy.
