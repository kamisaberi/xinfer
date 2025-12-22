# Qualcomm QNN Backend for xInfer

This backend executes models on **Snapdragon** devices via the **AI Engine Direct (QNN)** SDK. It is the successor to SNPE and provides the highest performance for the **Hexagon HTP**.

## 🛠️ Prerequisites

1.  **QNN SDK:**
    Download the QNN SDK from [Qualcomm Create](https://qpm.qualcomm.com/).
    Set `export QNN_SDK_ROOT=/opt/qcom/qnn`.
2.  **Target Device:**
    A device with a supported Hexagon DSP (Snapdragon 8 Gen 2+, RB5, etc.).

## ⚙️ Model Conversion

QNN does not load ONNX files directly. You must convert them to a **Context Binary** (`.bin`) for the specific SoC.

**1. ONNX to QNN (Intermediate):**
```bash
qnn-onnx-converter --input_network model.onnx --output_path model_qnn.cpp
```

**2. Compile to Context Binary:**
```bash
qnn-context-binary-generator --model model_qnn.so --backend libQnnHtp.so --output model.bin
```

**Using xInfer CLI:**
```bash
xinfer-cli compile --target qcom-qnn \
                   --onnx model.onnx \
                   --output model.bin \
                   --vendor-params BACKEND=HTP
```

## 💻 C++ Usage

```cpp
xinfer::zoo::vision::DetectorConfig config;
config.target = xinfer::Target::QUALCOMM_QNN;
config.model_path = "models/yolov8.bin"; // The Context Binary

// Optional: High Performance Burst Mode
config.vendor_params = { "PERF=BURST" };
```

## ⚠️ Runtime Libraries

QNN requires the backend `.so` files to be present on the target device at runtime.
Ensure `libQnnHtp.so`, `libQnnSystem.so`, and `libQnnHtpV73.so` (or specific arch version) are in your `LD_LIBRARY_PATH`.
