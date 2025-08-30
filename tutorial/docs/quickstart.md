# 🚀 Quickstart: Real-Time Object Detection in 5 Minutes

Welcome to the `xInfer` quickstart guide! This tutorial will walk you through the entire high-performance pipeline, from downloading a standard ONNX model to running it for real-time object detection in a C++ application.

By the end of this guide, you will have a clear understanding of the core `xInfer` workflow and the power of the `zoo` API.

**What You'll Accomplish:**
1.  **Download** a pre-trained YOLOv8 object detection model.
2.  **Optimize** it into a hyper-performant TensorRT engine using the `xinfer-cli` tool.
3.  **Use** the `xinfer::zoo::vision::Detector` in a simple C++ application to run inference on an image.

**Prerequisites:**
- You have successfully installed `xInfer` and its dependencies by following the **[Installation](./installation.md)** guide.
- You have the `xinfer-cli` and `xinfer_example` executables in your `build` directory.

---

### **Step 1: Get a Pre-Trained Model**

First, we need a model to optimize. We'll use the popular **YOLOv8-Nano**, a small and fast object detector trained on the COCO dataset. It's perfect for a quickstart guide.

Let's download the ONNX version of the model.

```bash
# From your project's root directory
mkdir -p assets
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx -O assets/yolov8n.onnx
```

We also need the list of class names for the COCO dataset.

```bash
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names -O assets/coco.names
```

You should now have an `assets` directory containing `yolov8n.onnx` and `coco.names`.

---

### **Step 2: Optimize with `xinfer-cli` (The "F1 Car" Build)**

This is where the magic happens. We will use the `xinfer-cli` tool to convert the standard `.onnx` file into a hyper-optimized `.engine` file. We'll enable **FP16 precision**, which provides a ~2x speedup on modern NVIDIA GPUs.

```bash
# Navigate to your build directory where the CLI tool was created
cd build/tools/xinfer-cli

# Run the build command
./xinfer-cli --build \
    --onnx ../../assets/yolov8n.onnx \
    --save_engine ../../assets/yolov8n_fp16.engine \
    --fp16
```

You will see output from the TensorRT builder as it analyzes, optimizes, and fuses the layers of the model. After a minute or two, you will have a new file: `assets/yolov8n_fp16.engine`. This file is a fully compiled, self-contained inference engine tuned for *your specific GPU*.

!!! success "What just happened?"
You just performed a complex, ahead-of-time compilation that would have required hundreds of lines of verbose C++ TensorRT code. `xinfer-cli` automated this entire process with a single, clear command.

---

### **Step 3: Use the Engine in C++ (The "Easy Button" API)**

Now, we'll write a very simple C++ program to use our new engine. This demonstrates the power and simplicity of the `xinfer::zoo` API.

Create a new file named `quickstart_detector.cpp` in your `examples` directory.

**File: `examples/quickstart_detector.cpp`**
```cpp
#include <xinfer/zoo/vision/detector.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdexcept>

int main() {
    try {
        // 1. Configure the detector to use our new engine and labels.
        xinfer::zoo::vision::DetectorConfig config;
        config.engine_path = "assets/yolov8n_fp16.engine";
        config.labels_path = "assets/coco.names";
        config.confidence_threshold = 0.5f;

        // 2. Initialize the detector.
        // This is a fast, one-time setup that loads the optimized engine.
        std::cout << "Loading object detector...\n";
        xinfer::zoo::vision::ObjectDetector detector(config);

        // 3. Load an image to run inference on.
        // (Create a simple dummy image for this test)
        cv::Mat image = cv::Mat(480, 640, CV_8UC3, cv::Scalar(114, 144, 154));
        cv::putText(image, "xInfer Quickstart!", cv::Point(50, 240), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(255, 255, 255), 3);
        cv::imwrite("quickstart_input.jpg", image);
        std::cout << "Created a dummy image: quickstart_input.jpg\n";

        // 4. Predict in a single line of code.
        // xInfer handles all the pre-processing, inference, and NMS post-processing.
        std::cout << "Running prediction...\n";
        std::vector<xinfer::zoo::vision::BoundingBox> detections = detector.predict(image);

        // 5. Print and draw the results.
        std::cout << "\nFound " << detections.size() << " objects (this will be 0 on a dummy image).\n";
        for (const auto& box : detections) {
            std::cout << " - " << box.label << " (Confidence: " << box.confidence << ")\n";
            cv::rectangle(image, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2), cv::Scalar(0, 255, 0), 2);
        }

        cv::imwrite("quickstart_output.jpg", image);
        std::cout << "Saved annotated image to quickstart_output.jpg\n";

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
```

You will need to add this new example to your root `CMakeLists.txt` to build it:
```cmake
# In your root CMakeLists.txt
# ... after the existing add_executable(xinfer_example ...)
add_executable(quickstart_detector examples/quickstart_detector.cpp)
target_link_libraries(quickstart_detector PRIVATE xinfer)
```

Now, rebuild and run your new example:
```bash
# From your build directory
cmake ..
make

# Run the quickstart
./examples/quickstart_detector
```

---

### **Conclusion**

Congratulations! In just a few minutes, you have:
- Taken a standard open-source model.
- Converted it into a hyper-performant engine tuned for your hardware.
- Used it in a clean, simple, and production-ready C++ application.

This is the core workflow that `xInfer` is designed to perfect. You are now ready to explore the rest of the **[Model Zoo](./zoo-api/index.md)** or dive into the **[How-to Guides](./guides/building-engines.md)** to learn how to optimize your own custom models.
