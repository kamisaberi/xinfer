# xInfer: Maximum Performance, Minimum Effort.

**xInfer is a C++ inference toolkit designed for a single purpose: to run your trained deep learning models with the absolute maximum performance possible on NVIDIA GPUs.**

It is the definitive solution for C++ developers who need to deploy AI in latency-critical, high-throughput, or power-constrained environments.

---

### The Problem: The AI Deployment Wall

Deploying AI models in production-grade C++ is brutally difficult. Developers hit a wall of complexity that forces them to choose between performance and productivity.

- **High-Level Frameworks (like LibTorch):** Easy to use, but their eager-mode execution has overhead, and they lack a streamlined path to state-of-the-art inference optimization.
- **Low-Level Libraries (like TensorRT):** Incredibly powerful, but have a notoriously steep learning curve, a verbose API, and require you to write complex, boilerplate-heavy code for even simple tasks.

This leaves a massive gap for developers who need both **the simplicity of a high-level API and the bare-metal speed of a low-level one.**

### The Solution: xInfer

`xInfer` fills this gap by providing a complete, two-layer solution built on top of NVIDIA TensorRT.

#### 1. The `xInfer::zoo` - The "Easy Button" API

A comprehensive library of pre-packaged, task-oriented pipelines for the most common AI models. With the `zoo`, you can get a hyper-performant, state-of-the-art object detector or diffusion model running in just a few lines of C++.

**It's this simple:**
```cpp
#include <xinfer/zoo/vision/detector.h>
#include <opencv2/opencv.hpp>

int main() {
    // 1. Configure the detector to use your pre-built TensorRT engine
    xinfer::zoo::vision::DetectorConfig config;
    config.engine_path = "yolov8n.engine";
    config.labels_path = "coco.names";

    // 2. Initialize the detector. All complexity is handled internally.
    xinfer::zoo::vision::ObjectDetector detector(config);

    // 3. Load an image and predict in one line.
    cv::Mat image = cv::imread("my_image.jpg");
    auto detections = detector.predict(image);
}