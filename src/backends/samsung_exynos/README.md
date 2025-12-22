# Samsung Exynos Backend for xInfer

This backend executes models on **Samsung Exynos** SoCs (e.g., Exynos 2200, 2400) using the **ENN (Exynos Neural Network)** SDK. It is designed for high-performance mobile and automotive applications.

## ⚠️ Proprietary SDK Warning

The ENN SDK is **NOT** open source. It is typically available:
1.  Inside the **Android Source Tree** (AOSP) for specific devices (Pixel / Galaxy).
2.  Via the **Samsung Partner Portal** for Automotive (Exynos Auto).
3.  Sometimes exposed via **Eden** libraries on Tizen.

## 🛠️ Prerequisites

1.  **Environment Variable:**
    Set `export ENN_SDK_ROOT=/path/to/enn_sdk`.
    The SDK must contain `include/EnnApi.h` and `lib/libenn_public_api_cpp.so`.

## ⚙️ Model Conversion

ENN requires models (TFLite/ONNX) to be converted to the **`.nnc`** (Neural Network Container) or **`.sg`** (Samsung Graph) format using the **ENN Compiler**.

**Using xInfer CLI:**
```bash
xinfer-cli compile --target samsung-exynos \
                   --onnx model.onnx \
                   --output model.nnc \
                   --precision int8
```

## 💻 C++ Usage

```cpp
xinfer::zoo::vision::DetectorConfig config;
config.target = xinfer::Target::SAMSUNG_EXYNOS;
config.model_path = "models/yolov8.nnc";

// Optional: Enable Boost Mode for low latency
config.vendor_params = { "POWER=BOOST" };
```

## ⚠️ Memory Optimization

Exynos NPUs are highly sensitive to memory allocation.
*   **Best Performance:** Use **ION Buffers** (Android) or **DMABUF** (Linux). xInfer detects if `inputs[0]` is a `CmaContiguous` tensor and passes the file descriptor directly to the NPU driver, enabling Zero-Copy inference.
*   **Standard Performance:** If you pass standard `malloc` memory, the driver must perform a copy to NPU-accessible SRAM/DRAM, which adds latency.
