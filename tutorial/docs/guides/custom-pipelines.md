# How-To Guide: Building Custom Pipelines

The **[`xInfer::zoo`](../zoo-api/index.md)** provides incredible, one-line solutions for common AI tasks. But what if your task is unique? What if you have a custom model with a non-standard input or a complex, multi-stage post-processing logic?

For these scenarios, `xInfer` provides the low-level **Core Toolkit**. This guide will show you how to use these powerful building blocks—`core`, `preproc`, and `postproc`—to build a completely custom, high-performance inference pipeline from scratch.

This is the "power user" path. It gives you maximum control and flexibility.

---

## The Goal: A Custom Multi-Model Pipeline

Let's imagine a real-world robotics task that isn't in the `zoo`: **"Find the largest object in a scene and classify it."**

This requires a two-stage pipeline:
1.  **Run an Object Detector:** To find all objects and their bounding boxes.
2.  **Run an Image Classifier:** To classify the content of the *largest* bounding box found.

We will build this pipeline step-by-step using the `xInfer` Core Toolkit.

**Prerequisites:**
- You have already used `xinfer-cli` to build two separate engine files:
  - `yolov8n.engine`: An object detection model.
  - `resnet18.engine`: An image classification model.

---

### Step 1: Initialize the Engines and Processors

First, we load our pre-built engines and set up the necessary pre-processing modules.

```cpp
#include <xinfer/core/engine.h>
#include <xinfer/preproc/image_processor.h>
#include <xinfer/postproc/detection.h>
#include <xinfer/postproc/yolo_decoder.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm> // For std::max_element

int main() {
    try {
        // --- 1. Load all necessary engines ---
        std::cout << "Loading engines...\n";
        xinfer::core::InferenceEngine detector_engine("yolov8n.engine");
        xinfer::core::InferenceEngine classifier_engine("resnet18.engine");

        // --- 2. Set up pre-processors for each model ---
        // The detector uses a 640x640 input with letterboxing
        xinfer::preproc::ImageProcessor detector_preprocessor(640, 640, true);
        // The classifier uses a 224x224 input with standard ImageNet normalization
        xinfer::preproc::ImageProcessor classifier_preprocessor(224, 224, {0.485, 0.456, 0.406}, {0.229, 0.224, 0.225});

        // --- 3. Load input image ---
        cv::Mat image = cv::imread("my_scene.jpg");
        if (image.empty()) {
            throw std::runtime_error("Failed to load image.");
        }
```

### Step 2: Run the First Stage (Detection)

Now, we execute the first part of our custom pipeline: find all objects.

```cpp
        // --- 4. Run the detection pipeline ---
        std::cout << "Running detection stage...\n";
        
        // 4a. Prepare the input tensor for the detector
        auto det_input_shape = detector_engine.get_input_shape(0);
        xinfer::core::Tensor det_input_tensor(det_input_shape, xinfer::core::DataType::kFLOAT);
        detector_preprocessor.process(image, det_input_tensor);

        // 4b. Run inference
        auto det_output_tensors = detector_engine.infer({det_input_tensor});

        // 4c. Run the custom post-processing kernels
        // We use intermediate GPU tensors to avoid slow CPU round-trips
        const int MAX_BOXES = 1024;
        xinfer::core::Tensor decoded_boxes({MAX_BOXES, 4}, xinfer::core::DataType::kFLOAT);
        xinfer::core::Tensor decoded_scores({MAX_BOXES}, xinfer::core::DataType::kFLOAT);
        xinfer::core::Tensor decoded_classes({MAX_BOXES}, xinfer::core::DataType::kINT32);
        
        xinfer::postproc::yolo::decode(det_output_tensors, 0.5f, decoded_boxes, decoded_scores, decoded_classes);
        std::vector<int> nms_indices = xinfer::postproc::detection::nms(decoded_boxes, decoded_scores, 0.5f);

        if (nms_indices.empty()) {
            std::cout << "No objects found.\n";
            return 0;
        }
```

### Step 3: Connect the Pipelines (Custom Logic)

This is where the power of the Core API shines. We can now implement our custom logic: find the largest box and prepare it for the next stage.

```cpp
        // --- 5. Custom Logic: Find the largest detected object ---
        std::cout << "Finding largest object...\n";

        // Download only the necessary box data to the CPU
        std::vector<float> h_boxes(decoded_boxes.num_elements());
        decoded_boxes.copy_to_host(h_boxes.data());

        float max_area = 0.0f;
        cv::Rect largest_box_roi;

        for (int idx : nms_indices) {
            float x1 = h_boxes[idx * 4 + 0];
            float y1 = h_boxes[idx * 4 + 1];
            float x2 = h_boxes[idx * 4 + 2];
            float y2 = h_boxes[idx * 4 + 3];
            float area = (x2 - x1) * (y2 - y1);

            if (area > max_area) {
                max_area = area;
                // Scale coordinates back to original image size
                float scale_x = (float)image.cols / 640.0f;
                float scale_y = (float)image.rows / 640.0f;
                largest_box_roi = cv::Rect(x1 * scale_x, y1 * scale_y, (x2-x1) * scale_x, (y2-y1) * scale_y);
            }
        }
```

### Step 4: Run the Second Stage (Classification)

Finally, we run the classifier on the cropped image of the largest object.

```cpp
        // --- 6. Run the classification pipeline on the cropped image ---
        std::cout << "Running classification stage on the largest object...\n";

        cv::Mat largest_object_patch = image(largest_box_roi);

        // 6a. Prepare the input tensor for the classifier
        auto cls_input_shape = classifier_engine.get_input_shape(0);
        xinfer::core::Tensor cls_input_tensor(cls_input_shape, xinfer::core::DataType::kFLOAT);
        classifier_preprocessor.process(largest_object_patch, cls_input_tensor);

        // 6b. Run inference
        auto cls_output_tensors = classifier_engine.infer({cls_input_tensor});

        // 6c. Post-process the result on the CPU
        std::vector<float> logits(cls_output_tensors.num_elements());
        cls_output_tensors.copy_to_host(logits.data());
        auto max_it = std::max_element(logits.begin(), logits.end());
        int class_id = std::distance(logits.begin(), max_it);

        std::cout << "\n--- Custom Pipeline Result ---\n";
        std::cout << "The largest object was classified as: Class " << class_id << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "An error occurred in the custom pipeline: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
```

---

### **Conclusion**

As you can see, the Core Toolkit provides the ultimate level of control. By composing the low-level `InferenceEngine`, `ImageProcessor`, and `postproc` functions, you can build sophisticated, multi-stage pipelines that are tailored to your exact needs, all while maintaining the hyper-performance of the underlying CUDA and TensorRT components.

This is the power of `xInfer`: it provides a simple, elegant "easy button" with the `zoo`, but it never takes away the expert's ability to open the hood and build their own F1 car from scratch.
