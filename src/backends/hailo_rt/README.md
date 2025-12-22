# HailoRT Backend for xInfer

This backend executes models on **Hailo-8** and **Hailo-8L** AI processors. It communicates with the hardware via the **HailoRT** C++ library over PCIe or USB.

## 🛠️ Hardware Setup

1.  **Install Hailo PCIe Driver:**
    Download `hailo-pcie-driver` from the Hailo Developer Zone.
    ```bash
    sudo dpkg -i hailo-pcie-driver_*.deb
    sudo modprobe hailo_pci
    ```

2.  **Install HailoRT Library:**
    Download `hailort_*.deb` (Runtime).
    ```bash
    sudo dpkg -i hailort_*.deb
    ```
    Ensure `libhailort.so` is in `/usr/lib`.

## ⚙️ Model Compilation (.hef)

Hailo requires models to be compiled into **HEF (Hailo Executable Format)**.

**Using xInfer CLI:**
```bash
xinfer-cli compile --target hailo-rt \
                   --onnx model.onnx \
                   --output model.hef \
                   --precision int8 \
                   --calibrate ./calib_data/
```

This uses the Hailo Dataflow Compiler (DFC) internally.

## 💻 C++ Usage

```cpp
xinfer::zoo::vision::DetectorConfig config;
config.target = xinfer::Target::HAILO_RT;
config.model_path = "yolov8.hef";

// Optional: Pass raw UINT8 image bytes for max speed (Zero-Copy)
config.vendor_params = { "FORMAT=UINT8" };
```

## ⚠️ Performance Note

*   **VStreams Latency:** The initial creation of VStreams takes a few hundred milliseconds. Do not destroy/recreate the backend per frame.
*   **Batch Size:** The batch size is **baked into the HEF** at compile time. If you compile for Batch=1, you cannot run Batch=8 at runtime.
