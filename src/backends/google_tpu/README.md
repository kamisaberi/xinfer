# Google Edge TPU Backend for xInfer

This backend executes models on **Google Coral** hardware (USB Accelerator, M.2 Card, Dev Board). It uses the **TensorFlow Lite C++ API** and the **libedgetpu** delegate.

## ⚠️ Model Requirements

Models **must** be:
1.  Full Integer Quantized (INT8).
2.  Compiled with the `edgetpu_compiler`.

Files usually end in `_edgetpu.tflite`.

## 🛠️ Installation on Linux (Debian/Ubuntu)

1.  **Add Google Coral Repos:**
    ```bash
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
    sudo apt-get update
    ```

2.  **Install Runtime & Headers:**
    ```bash
    # Minimal runtime
    sudo apt-get install libedgetpu1-std 
    
    # Development headers (Required for building xInfer)
    sudo apt-get install libedgetpu-dev
    
    # TensorFlow Lite headers (often manual install needed, or via package)
    # xInfer's CMake will attempt to find them.
    ```

## ⚙️ Usage

### 1. Compile Model (CLI)
Use `xinfer-cli` (which wraps `edgetpu_compiler`):

```bash
xinfer-cli compile --target google-tpu \
                   --onnx model.onnx \
                   --output model_edgetpu.tflite \
                   --precision int8 \
                   --calibrate ./data/
```

### 2. C++ Usage
```cpp
xinfer::zoo::vision::DetectorConfig config;
config.target = xinfer::Target::GOOGLE_TPU;
config.model_path = "models/mobilenet_v2_ssd_quant_edgetpu.tflite";

// Optional: Select 2nd TPU if you have multiple
config.vendor_params = { "DEVICE_INDEX=1" }; 
```
