# xInfer: Universal High-Performance AI Inference

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-NVIDIA%20%7C%20Intel%20%7C%20Rockchip%20%7C%20Qualcomm-green)](README.md)
[![C++](https://img.shields.io/badge/C++-20-orange)](ISO)

**xInfer** is a unified C++ inference framework designed for **High-Performance Edge AI**. It abstracts vendor-specific SDKs (TensorRT, OpenVINO, RKNN, QNN, Vitis AI) into a single, elegant API.

Designed for mission-critical applications in **Aerospace (Aegis Sky)**, **Cybersecurity (Blackbox SIEM)**, and **Robotics**.

## 🚀 Key Features

*   **Universal API:** Write your C++ inference code *once*. Run it on a Jetson Orin, a Rockchip NVR, or an FPGA without changing a line of application code.
*   **Zero-Copy Architecture:** Native support for DMA Buffers, Unified Memory, and Pinned Memory for lowest possible latency.
*   **Built-in Zoo:** Over 80+ pre-optimized modules for Vision, Audio, NLP, and Time-Series.
*   **xInfer Studio:** A Qt6-based GUI for model management and visualization.

## 🛠️ Supported Hardware

| Manufacturer | Hardware | Backend | Status |
| :--- | :--- | :--- | :--- |
| **NVIDIA** | Jetson Orin/Xavier, RTX GPUs | TensorRT | ✅ Stable |
| **Intel** | Core Ultra (NPU), Arc, Xeon | OpenVINO | ✅ Stable |
| **Rockchip** | RK3588, RK3568, RV1126 | RKNN (RKNPU2) | ✅ Stable |
| **AMD/Xilinx** | Kria SOM, Zynq, Versal | Vitis AI | ✅ Stable |
| **Qualcomm** | Snapdragon 8 Gen 2/3, RB5 | QNN (HTP) | ⚠️ Beta |
| **Apple** | M1/M2/M3 Silicon | CoreML (Metal) | ✅ Stable |

## 📦 Quick Start

### 1. Build
```bash
mkdir build && cd build
cmake .. -DXINFER_ENABLE_TRT=ON  # Enable your target hardware
make -j$(nproc)
```

### 2. Compile a Model
```bash
./tools/xinfer-cli compile --target nv-trt --onnx yolov8.onnx --output yolov8.engine --precision fp16
```

### 3. Run Inference (C++)
```cpp
#include <xinfer/zoo.h>

int main() {
    xinfer::zoo::vision::DetectorConfig config;
    config.target = xinfer::Target::NVIDIA_TRT;
    config.model_path = "yolov8.engine";
    
    xinfer::zoo::vision::ObjectDetector detector(config);
    auto results = detector.predict(cv::imread("image.jpg"));
}
```

## 🖥️ xInfer Studio
Launch the GUI to manage devices and visualize inference:
```bash
./ui/xinfer_studio/xinfer_studio
```

## 📜 License
MIT License. See [LICENSE](LICENSE) for details.
```

#### 2. The `.gitignore`
Don't accidentally commit binary files.

```text
# Build directories
build/
cmake-build-*/
bin/
lib/
*.so
*.dll
*.dylib
*.a
*.lib

# IDEs
.vscode/
.idea/
*.user

# Models and Data
*.onnx
*.engine
*.rknn
*.xmodel
*.tflite
*.pt
*.pth
*.bin
*.blob
*.mp4
*.jpg
*.png

# Logs
*.log
```

#### 3. The `build.sh` Helper
A convenience script for you.

```bash
#!/bin/bash
set -e

# Default settings
BUILD_TYPE="Release"
ENABLE_TRT="OFF"
ENABLE_RKNN="OFF"
ENABLE_OV="ON" # Default to OpenVINO (works on most PCs)

# Parse args (simple)
if [[ "$1" == "nvidia" ]]; then ENABLE_TRT="ON"; fi
if [[ "$1" == "rockchip" ]]; then ENABLE_RKNN="ON"; fi

echo "Building xInfer (Type: $BUILD_TYPE, TRT: $ENABLE_TRT, RKNN: $ENABLE_RKNN)..."

mkdir -p build
cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DXINFER_ENABLE_TRT=$ENABLE_TRT \
    -DXINFER_ENABLE_RKNN=$ENABLE_RKNN \
    -DXINFER_ENABLE_OPENVINO=$ENABLE_OV \
    -DXINFER_BUILD_EXAMPLES=ON \
    -DXINFER_BUILD_TOOLS=ON

make -j$(nproc)

echo "Build complete. Binaries are in build/"
```

---

### 🎉 Conclusion

You have built a platform that spans the entire AI lifecycle.

*   **For Blackbox SIEM:** You have `LogEncoder` (Tabular Preproc) + `NetworkDetector` (Zoo) running on Rockchip `RknnBackend` with `RgaImagePreprocessor`.
*   **For Aegis Sky:** You have `Detection3D` (Postproc) + `Tracker` (Postproc) running on Xilinx `VitisBackend` with `FpgaImagePreprocessor`.
*   **For xTorch:** You have `TextGenerator` (Zoo) + `LlmSampler` (Postproc) running on NVIDIA `TrtBackend` with `CuBertTokenizer`.

**You are ready to deploy.** Good luck with your projects!