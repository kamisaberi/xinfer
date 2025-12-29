# xInfer: The Universal High-Performance AI Runtime

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![C++ Standard](https://img.shields.io/badge/C++-20-blue.svg)](https://isocpp.org/std/the-standard)
[![Platforms](https://img.shields.io/badge/Supported%20Platforms-15+-success)](README.md#hardware-support-matrix)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

**xInfer is the industry's first truly universal C++ inference framework.** It abstracts the fragmented world of Edge AI SDKs into a single, high-performance API.

Designed for mission-critical applications in **Aerospace**, **Cybersecurity**, **Robotics**, and **Medical Imaging**, xInfer allows you to write your application logic **once** and deploy it natively to **15+ hardware architectures** without changing a single line of C++ code.

---

## 🚀 Why xInfer?

The Edge AI landscape is fragmented. Deploying a model usually requires learning a new SDK for every chip: TensorRT for NVIDIA, OpenVINO for Intel, RKNN for Rockchip, Vitis for FPGA, etc.

**xInfer solves this.**

*   **Unified C++20 API:** One interface to rule them all.
*   **Zero-Copy Architecture:** Native support for DMA Buffers, Unified Memory, and Pinned Memory. No unnecessary CPU-GPU copies.
*   **Built-in Model Zoo:** Over 80+ pre-optimized, task-specific modules (YOLO, FaceMesh, BERT, etc.).
*   **Full MLOps Stack:** Includes a Compiler, Preprocessor, Postprocessor, and Deployment Manager.

---

## 🏛️ Hardware Support Matrix

xInfer supports native acceleration on the following **15 platforms** via specialized backend drivers.

| Manufacturer | Target Hardware | Backend / SDK | Engine File Format | Best For |
| :--- | :--- | :--- | :--- | :--- |
| **NVIDIA** | GeForce RTX, Jetson Orin/Xavier | **TensorRT** (10.x) | `.engine` | High-Performance Vision, LLMs |
| **Intel** | Core Ultra (NPU), Xeon, Arc GPU | **OpenVINO** | `.xml` + `.bin` | PC/Laptop, Edge Servers |
| **Rockchip** | RK3588, RK3568, RV1126 | **RKNN** (RKNPU2) | `.rknn` | Smart Cameras, NVRs, **SIEM** |
| **AMD (Xilinx)** | Kria SOM, Zynq MPSoC, Versal | **Vitis AI** (DPU) | `.xmodel` | Low-Latency Robotics, **Aegis Sky** |
| **Qualcomm** | Snapdragon 8 Gen 2/3, RB5 | **QNN** (HTP) | `.bin` / `.so` | Drones, Mobile, AR/VR |
| **Apple** | M1, M2, M3, A-Series | **CoreML** (Metal) | `.mlmodelc` | iOS/macOS Applications |
| **AMD (Ryzen)** | Ryzen 7040/8040 (Phoenix) | **Ryzen AI** (IPU) | `.onnx` (Vitis EP) | AI PCs, Windows Laptops |
| **MediaTek** | Genio 1200, Dimensity | **NeuroPilot** (Neuron) | `.dla` / `.pte` | IoT, Smart Home |
| **Hailo** | Hailo-8, Hailo-8L, Hailo-10 | **HailoRT** | `.hef` | High-FPS Edge Box, RPi 5 Add-on |
| **Ambarella** | CV2, CV3, CV5 | **CVFlow** | `.cavalry` | Professional Security, Automotive |
| **Samsung** | Exynos 2200/2400 (NPU) | **ENN** (Eden) | `.nnc` | Mobile, Automotive Cockpit |
| **Google** | Coral USB / M.2 / Dev Board | **Edge TPU** | `.tflite` (Int8) | Low-Power IoT |
| **Intel (FPGA)** | Agilex, Stratix 10, Arria 10 | **FPGA AI Suite** (DLA) | `.bin` + `.aocx` | High-Throughput Data Center |
| **Microchip** | PolarFire SoC / FPGA | **VectorBlox** | `.blob` | Defense, Ultra-Low Power, Radiation |
| **Lattice** | CrossLink-NX, Certus-NX | **sensAI** | `.bin` | TinyML, Always-On Sensors |

---

## ⚡ Performance

xInfer is designed to be **faster than Python** and **as fast as native SDKs**.

**Benchmark: YOLOv8 Object Detection (640x640)**

| Platform | Framework | Latency | Overhead |
| :--- | :--- | :--- | :--- |
| **Jetson Orin** | Python (PyTorch) | 24.5 ms | High |
| **Jetson Orin** | **xInfer (TensorRT)** | **6.2 ms** | **Zero** |
| **Rockchip RK3588**| Python (RKNN-Lite)| 35.0 ms | High |
| **Rockchip RK3588**| **xInfer (RKNN)** | **14.1 ms** | **Zero** |
| **Kria KV260** | Python (VART) | 18.0 ms | Medium |
| **Kria KV260** | **xInfer (Vitis AI)**| **9.8 ms** | **Zero** |

---

## 🛠️ The xInfer Ecosystem

### 1. The Compiler (`xinfer-cli`)
Cross-compile models for any target from your development machine.

```bash
# Compile for NVIDIA Jetson
xinfer-cli compile --target nv-trt --onnx model.onnx --output model.engine --precision fp16

# Compile for Rockchip NPU (Requires INT8 calibration)
xinfer-cli compile --target rockchip-rknn --onnx model.onnx --output model.rknn --precision int8 --calibrate ./images/
```

### 2. The Model Zoo
Over 80+ task-specific modules ready to use.

```cpp
#include <xinfer/zoo.h>

// 1. Configure for your target (Swapping this enum changes the entire backend!)
xinfer::zoo::vision::DetectorConfig config;
config.target = xinfer::Target::ROCKCHIP_RKNN; 
config.model_path = "yolov8.rknn";

// 2. Instantiate (Factories handle the rest)
xinfer::zoo::vision::ObjectDetector detector(config);

// 3. Run
auto results = detector.predict(cv::imread("image.jpg"));
```

### 3. The Deployment Manager
Push updates to your fleet of devices via SSH directly from the CLI or Studio.

```bash
# Deploy app and model to a remote drone
xinfer-cli deploy --device "drone_fleet_01" --app ./build/aegis_sky --onnx ./models/tracker.onnx
```

### 4. xInfer Studio (GUI)
A Qt6-based IDE to visualize inference, manage devices, and benchmark models.

<p align="center">
  <img src="docs/images/xinfer_studio_preview.png" alt="xInfer Studio UI" width="800"/>
</p>

---

## 📦 Installation

### Prerequisites
*   CMake 3.20+
*   C++20 Compiler (GCC 11+, Clang 14+, MSVC 19.30+)
*   OpenCV 4.x
*   (Optional) Vendor SDKs: CUDA, TensorRT, RKNPU2, Vitis AI, etc.

### Building
```bash
git clone https://github.com/kamisaberi/xinfer.git
cd xinfer && mkdir build && cd build

# Enable the backends you have installed
cmake .. \
    -DXINFER_ENABLE_TRT=ON \
    -DXINFER_ENABLE_RKNN=OFF \
    -DXINFER_ENABLE_OPENVINO=ON \
    -DXINFER_BUILD_TOOLS=ON

make -j$(nproc)
```

---

## 📂 Project Structure

*   `src/core`: Hardware-agnostic Tensor and Memory management.
*   `src/backends`: The 15 hardware driver implementations.
*   `src/compiler`: The `xinfer-cli` build logic.
*   `src/preproc`: HW-accelerated data processing (CUDA, RGA, NEON, Metal).
*   `src/postproc`: HW-accelerated decoding (NMS, Softmax).
*   `src/zoo`: High-level application modules (Vision, Audio, NLP, etc.).
*   `ui/`: The xInfer Studio Qt6 application.

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to add support for new backends or Zoo models.

## 📜 License

Copyright (c) 2025 Kami Saberi.
Licensed under the **MIT** License.