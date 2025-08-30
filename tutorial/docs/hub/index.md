# The Ignition Hub: Pre-Built Engines, On Demand.

Welcome to the future of C++ AI deployment. The **Ignition Hub** is a cloud-based repository of pre-built, hyper-optimized TensorRT engine files for the world's most popular open-source models.

It is designed to completely eliminate the slowest, most complex, and most error-prone step in the entire inference pipeline: **the engine build process.**

---

### The Problem: The "Build Barrier"

Every developer who has used NVIDIA TensorRT knows the pain of the build step:

- **It's Slow:** Building an engine, especially with INT8 calibration, can take many minutes or even hours.
- **It's Heavy:** It requires you to have the full, multi-gigabyte CUDA Toolkit, cuDNN, and TensorRT SDKs installed and correctly configured on your machine.
- **It's Brittle:** An engine is a compiled binary. An engine built for an RTX 4090 with TensorRT 10 **will not work** on a Jetson Orin with TensorRT 8.6.

This "Build Barrier" makes rapid prototyping, collaboration, and deployment across different hardware targets a massive challenge.

### The Solution: Download, Don't Build.

The Ignition Hub solves this problem by treating the engine build process as a **centralized, cloud-native service.** We run the slow, complex build process on our massive cloud build farm, so you don't have to.

The workflow is transformed:

**Old Workflow:**
1.  Find and download a model's ONNX file.
2.  Install all the heavy SDKs (CUDA, cuDNN, TensorRT).
3.  Write complex C++ build code using the `xinfer::builders` API.
4.  Wait 10 minutes for the engine to build.
5.  Finally, run your application.

**The Ignition Hub Workflow:**
1.  Find your model on the Hub.
2.  Call a single C++ function: `xinfer::hub::download_engine(...)`.
3.  Run your application **instantly**.

---

### How It Works

The Ignition Hub is a massive, curated catalog of engine files. For every major open-source model (like Llama 3 or YOLOv8), we have pre-built and stored a matrix of engines for every common combination of:

- **GPU Architecture:** From the Jetson Nano (`sm_52`) to the H100 (`sm_90`).
- **TensorRT Version:** From legacy 8.x versions to the latest 10.x.
- **Precision:** `FP32`, `FP16`, and `INT8`.

When you request an engine, our service delivers the one, single, perfectly-optimized binary that is guaranteed to work on your specific hardware and software configuration.

### Example: The "Magic" of the Hub-Integrated `zoo`

The true power of the Hub is its seamless integration with the `xinfer::zoo` API. The `zoo` classes have special constructors that can download models directly from the Hub.

This is the future of C++ AI deployment.

```cpp
#include <xinfer/zoo/vision/detector.h>
#include <xinfer/hub/model_info.h> // For the HardwareTarget struct
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    try {
        // 1. Define the model we want and the exact hardware we are running on.
        std::string model_id = "yolov8n-coco";
        xinfer::hub::HardwareTarget my_target = {
            .gpu_architecture = "Jetson_Orin_Nano",
            .tensorrt_version = "10.0.1",
            .precision = "INT8"
        };
        
        // 2. Instantiate the detector.
        //    This one line of code will:
        //    - Connect to the Ignition Hub.
        //    - Find the perfect pre-built engine for our exact hardware.
        //    - Download and cache it locally.
        //    - Load it into the InferenceEngine.
        std::cout << "Initializing detector from Ignition Hub...\n";
        xinfer::zoo::vision::ObjectDetector detector(model_id, my_target);

        // 3. The detector is now ready to run at maximum performance.
        std::cout << "Detector ready. Running inference...\n";
        cv::Mat image = cv::imread("my_image.jpg");
        auto detections = detector.predict(image);

        std::cout << "Found " << detections.size() << " objects.\n";

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
```

---

## Get Started

Ready to stop building and start inferring?

- **[Usage Guide](./usage.md):** Learn how to use the `xinfer::hub` C++ API.
- **Browse the Hub:** (Link to your future web UI)
