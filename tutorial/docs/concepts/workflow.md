# The `xInfer` Workflow: Build Once, Infer Fast

The core philosophy of `xInfer` is to do as much work as possible **ahead of time** so that your final inference pipeline is as lean and fast as humanly possible. This is achieved through a clear, two-stage workflow.

![xInfer Workflow Diagram](https://your-image-host.com/xinfer_workflow.png)
*(You would create a simple diagram for this: [Model Training] -> [ONNX] -> [xInfer Build Step] -> [TensorRT Engine] -> [xInfer Inference Step])*

---

### **Stage 1: The Build Step (Offline)**

This is the "factory" where you create your hyper-optimized "F1 car" engine. This is a **one-time, offline process** that you run during development or as part of your CI/CD pipeline.

The goal of this stage is to convert a trained model from a flexible, high-level format into a rigid, hardware-specific, and incredibly fast TensorRT engine file.

#### **1a. Start with a Trained Model**

Your journey begins with a model that has already been trained in a framework like PyTorch, TensorFlow, or your own **`xTorch`**. The output of this training process is a set of learned weights.

#### **1b. Export to a Standard Format: ONNX**

TensorRT, the core optimization engine used by `xInfer`, works best with a standard, open-source format called **ONNX (Open Neural Network Exchange)**. An ONNX file describes the architecture of your model in a way that different tools can understand.

- **From `xTorch`:** You can use the `xinfer::builders::export_to_onnx()` function to convert your trained `xTorch` models.
- **From Python:** You would use the standard `torch.onnx.export()` function.

This step gives you a portable, framework-agnostic representation of your model (e.g., `yolov8n.onnx`).

#### **1c. Build the TensorRT Engine with `xInfer`**

This is the most critical part of the build step and where `xInfer` provides its first major value. You use the `xinfer-cli` tool or the `xinfer::builders::EngineBuilder` C++ class to perform the final compilation.

This step does several powerful things:
- **Parses the ONNX Graph:** It reads your model's architecture.
- **Applies Optimizations:** It performs graph-level optimizations like **layer fusion**, combining `Conv+BN+ReLU` into a single operation.
- **Applies Quantization:** If you enable it, it will convert the model's weights to faster, lower-precision formats like **FP16** or **INT8**.
- **Hardware-Specific Tuning:** It selects the absolute fastest, hand-tuned CUDA kernels (called "tactics") for each layer on **your specific GPU architecture**.
- **Serializes the Engine:** It saves the final, fully optimized, and compiled model into a single binary file (e.g., `yolov8n_fp16.engine`).

**The output of this stage is a single `.engine` file. This file is the only artifact you need to ship with your final application.**

---

### **Stage 2: The Inference Step (Real-Time)**

This is what happens inside your final, deployed C++ application. This stage is designed to be **extremely fast, lightweight, and have minimal dependencies.** Your application does not need the heavy ONNX parser or TensorRT builder libraries; it only needs the much smaller TensorRT runtime.

#### **2a. Load the Engine**

Your application uses the `xinfer::zoo` or `xinfer::core` API to load the pre-built `.engine` file. This is a very fast operation, as it simply deserializes the compiled graph into GPU memory.

```cpp
#include <xinfer/zoo/vision/detector.h>

// The application only needs the final .engine file.
xinfer::zoo::vision::DetectorConfig config;
config.engine_path = "yolov8n_fp16.engine";
config.labels_path = "coco.names";

xinfer::zoo::vision::ObjectDetector detector(config);
```

#### **2b. Pre-process Input Data**

User input (like a camera frame or a piece of audio) needs to be converted into a GPU tensor. `xInfer` provides hyper-optimized, fused CUDA kernels for these tasks in the `xinfer::preproc` module to avoid CPU bottlenecks. The `zoo` classes handle this for you automatically.

#### **2c. Run Inference**

You call the `.predict()` method. This is where the magic happens. The `xinfer::core::InferenceEngine` takes your input tensor and executes the pre-compiled, hyper-optimized graph on the GPU. This is the fastest part of the entire process.

#### **2d. Post-process Output Data**

The raw output from the model (logits) is often not the final answer. It needs to be converted into a human-readable format. `xInfer` provides fused CUDA kernels for common post-processing bottlenecks (like NMS for object detection or ArgMax for segmentation) in the `xinfer::postproc` module.

```cpp
#include <opencv2/opencv.hpp>

// The .predict() call handles pre-processing, inference, and post-processing.
cv::Mat image = cv::imread("my_image.jpg");
auto detections = detector.predict(image); // Returns a clean vector of bounding boxes
```

---

### **Summary of the Workflow**

| Stage | **What Happens** | **Tools Used** | **Output** |
| :--- | :--- | :--- | :--- |
| **Build Step (Offline)** | A slow, one-time process of compiling and optimizing a trained model for a specific hardware target. | `xTorch`/PyTorch, `xinfer-cli`, `xinfer::builders` | **A single `.engine` file** |
| **Inference Step (Runtime)**| A blazing-fast, real-time process of loading the engine and running it on new data. | `xinfer::zoo`, `xinfer::core`, `xinfer::preproc`, `xinfer::postproc` | **The final prediction** |

By separating these two concerns, `xInfer` allows you to pay the high cost of optimization **once**, during development, and then reap the benefits of extreme performance in your final, lightweight C++ application.
