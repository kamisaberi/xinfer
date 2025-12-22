# Rockchip RKNN Backend for xInfer

This backend executes models on **RK3588**, **RK3568**, and **RV1126** SoCs using the **RKNPU2** interface. It allows for multi-core NPU scheduling and is optimized for the **Blackbox SIEM** edge collectors.

## 🛠️ Prerequisites

1.  **Hardware:** Rockchip-based board (Orange Pi 5, Radxa Rock 5, Firefly).
2.  **Driver:** The `librknnrt.so` library must be installed. This usually comes with the Ubuntu/Debian image provided by the board vendor.
    *   Check location: `/usr/lib/librknnrt.so`

## ⚙️ Model Conversion

Rockchip NPU requires models converted to **`.rknn`** format.

**Using xInfer CLI:**
```bash
xinfer-cli compile --target rockchip-rknn \
                   --onnx model.onnx \
                   --output model.rknn \
                   --precision int8 \
                   --calibrate ./data/calib_images
```
*(This command wraps the `rknn-toolkit2` Python package inside the xInfer Compiler Docker)*

## 💻 C++ Usage

```cpp
xinfer::zoo::vision::DetectorConfig config;
config.target = xinfer::Target::ROCKCHIP_RKNN;
config.model_path = "models/yolov8_rk3588.rknn";

// Optional: Force execution on all 3 NPU cores (Max Throughput)
config.vendor_params = { "CORE=ALL" };
```

## ⚠️ Performance Note

*   **Zero-Copy (Advanced):**
    For 4K video analysis, standard `memcpy` is too slow. The current backend uses `rknn_inputs_set` (copy mode). To enable true Zero-Copy, you must allocate memory using DRM/DMA-BUF (`dma_alloc`) and pass the file descriptor to the `xInfer` Tensor.
    *   *This requires xInfer to be compiled with `XINFER_ENABLE_DRM`.*
