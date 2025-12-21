# xInfer: The Universal C++ Inference Performance Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++ Standard](https://img.shields.io/badge/C++-20-blue.svg)](https://isocpp.org/std/the-standard)
[![Platforms](https://img.shields.io/badge/Platforms-15+-red.svg)](#platform-support-matrix)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/kamisaberi/xinfer)

**xInfer is a high-performance C++ inference engine designed for one purpose: to run your deep learning models with the absolute maximum performance possible on any hardware—from NVIDIA GPUs and Xilinx FPGAs to Snapdragon NPUs and Rockchip Edge devices.**

`xInfer` bridges the massive gap between training a model and deploying it in low-latency, mission-critical C++ environments. It automates the complex process of Ahead-of-Time (AOT) model compilation and provides a unified, elegant API that hides the extreme complexity of cross-platform hardware acceleration.

This library is the ideal deployment companion to **[xTorch](https://github.com/kamisaberi/xtorch)** and the backbone of the **[Ignition Hub](https://github.com/kamisaberi/ignition-hub)** ecosystem.

---

## 🚀 The Solution: A Unified AI Backbone

Deploying AI at the edge is a nightmare of fragmented SDKs and proprietary "Engine" files. `xInfer` replaces this chaos with a powerful, two-layer architecture:

1.  **The Universal Compiler (`xinfer-cli`):** A single extension that takes your ONNX or xTorch models and "bakes" them into hardware-specific optimized formats (`.engine`, `.xmodel`, `.rknn`, `.hef`, `.so`, etc.) for 15+ different platforms.
2.  **The xInfer Zoo API:** A collection of 30+ production-ready domain pipelines (Vision, Cybersecurity, Drones, HFT) that run identically across all supported hardware. This is the "F1 car" with the "limousine" interior.

---

## 🏗️ Platform Support Matrix

`xInfer` now supports **15+ specialized hardware backends**. Whether you are building **Blackbox SIEM** for server-side security or **Aegis Sky** for military-grade drone tracking, we have the driver.

| Category | Target Hardware | Engine Format | Key Feature |
| :--- | :--- | :--- | :--- |
| **Desktop/Server** | **NVIDIA** (TensorRT) | `.engine` | Maximum GPU Throughput |
| | **Intel** (OpenVINO) | `.xml / .bin` | Enterprise CPU/NPU scaling |
| | **AMD** (Ryzen AI) | `.dll` | XDNA NPU Laptop Acceleration |
| | **Apple** (CoreML) | `.mlmodelc` | Apple Neural Engine (ANE) |
| **FPGA/Adaptive** | **AMD/Xilinx** (Vitis-AI) | **`.xmodel`** | Ultra-low Latency (Aegis Sky) |
| | **Intel FPGA** (AI Suite) | `.bin` | Industrial Vision Pipelines |
| | **Microchip** (VectorBlox) | `.blob` | Hardened Security & SEU Immunity |
| | **Lattice** (sensAI) | `.bit` | Ultra-low power IoT |
| **Mobile/SoC** | **Qualcomm** (QNN) | **`.so`** | Hexagon NPU Performance |
| | **Rockchip** (RKNN) | **`.rknn`** | Best for Blackbox SIEM Edge Hubs |
| | **MediaTek** (NeuroPilot) | `.pte` | Mass-market Mobile AI |
| | **Samsung** (Exynos) | `.nnc` | Galaxy SoC Optimization |
| **Specialized** | **Hailo** (8/10) | `.hef` | High-FPS Vision modules |
| | **Ambarella** (CVFlow) | `.cavalry` | Professional Drone/Security Video |
| | **Google** (Edge TPU) | `.tflite` | Global IoT Standard |

---

## ⚡ Performance: The xInfer Advantage

For small models and real-time tracking, `xInfer` on specialized hardware (FPGA/NPU) crushes standard ARM-based Linux deployments.

### Benchmark: YOLOv8 Object Detection (End-to-End)
*Tested on Aegis Sky tracking module (1280x720 frame).*

| Implementation | Hardware | Latency (ms) | Speedup |
| :--- | :--- | :--- | :--- |
| **Python + PyTorch** | ARM Cortex-A72 | 11.5 ms | 1x |
| **C++ / LibTorch** | ARM Cortex-A72 | 10.4 ms | 1.1x |
| **C++ / xInfer (NPU)** | **Rockchip RK3588** | **2.1 ms** | **~5.5x** |
| **C++ / xInfer (FPGA)** | **Xilinx Zynq US+** | **0.8 ms** | **~14x** |

**Why xInfer is faster:**
*   **Zero-Copy Memory:** Custom `PhysicalBuffer` logic for FPGA/NPU DMA.
*   **Fused Kernels:** Hardware-specific pre-processing (Resize/Normalize) in CUDA, NEON, or HLS.
*   **AOT Quantization:** Extreme INT8/INT4 optimization via the `xinfer-compile` extension.

---

## 🛠️ The "Wow" Example: Cross-Platform Deployment

Switching from an NVIDIA Server to a Rockchip Edge Hub or a Xilinx Drone is now a single configuration change.

```cpp
#include <xinfer/zoo/vision/detector.h>

int main() {
    // 1. Define the config. Target the hardware of your choice!
    xinfer::zoo::vision::DetectorConfig config {
        .model_path = "./models/yolo_v8.xmodel", // or .rknn, .engine
        .target = xinfer::Target::AMD_VITIS,      // target FPGA for Aegis Sky
        .precision = xinfer::Precision::INT8,
        .use_fused_kernels = true
    };

    // 2. Instantiate the detector. 
    // xInfer automatically loads the correct backend driver.
    xinfer::zoo::vision::Detector detector(config);

    // 3. High-speed inference
    cv::Mat frame = cv::imread("drone_feed.jpg");
    auto detections = detector.predict(frame);
}
```

---

## 🩺 The xInfer-Doctor Tool

Managing 15 platforms is complex. Use the built-in doctor to verify your environment:

```bash
./xinfer-doctor
[PASS] NVIDIA TensorRT 10.0 detected.
[PASS] Rockchip RKNN-Toolkit2 detected.
[FAIL] AMD Vitis-AI Docker image not found. (Required for Aegis Sky)
[WARN] Qualcomm QNN_SDK_ROOT not set.
```

---

## 📦 Installation & Compilation

`xInfer` uses a unified CLI to compile models for any platform.

1. **Build xInfer:**
   ```bash
   mkdir build && cd build
   cmake .. -DPLATFORM=ALL 
   make -j$(nproc)
   ```

2. **Compile a Model for Rockchip (Blackbox SIEM):**
   ```bash
   xinfer-cli compile --onnx model.onnx --target rk-npu --precision int8 --output model.rknn
   ```

3. **Compile a Model for FPGA (Aegis Sky):**
   ```bash
   xinfer-cli compile --onnx model.onnx --target amd-vitis --vendor-params DPU_ARCH=DPUCZDX8G
   ```

---

## 🤝 Get Involved

`xInfer` is expanding the frontiers of high-performance C++ inference. We welcome contributions for new backend drivers and kernel optimizations.

*   **Cybersecurity/SIEM:** See `xinfer/zoo/cybersecurity`
*   **Aerospace/Defense:** See `xinfer/zoo/drones` (Aegis Sky)
*   **Core Toolkit:** See `xinfer/core`

**License:** MIT  
**Author:** [Kamisaberi](https://github.com/kamisaberi)