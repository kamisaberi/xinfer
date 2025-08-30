# Zoo API: Computer Vision

The `xinfer::zoo::vision` module provides a comprehensive suite of high-level, hyper-optimized pipelines for the most common computer vision tasks.

Each class in this module is an end-to-end solution that handles all the complexity of pre-processing, TensorRT inference, and GPU-accelerated post-processing, giving you the final, human-readable answer with a single `.predict()` call.

## `ImageClassifier`

Performs image classification, identifying the primary subject of an image.

**Header:** `#include <xinfer/zoo/vision/classifier.h>`

```cpp
#include <xinfer/zoo/vision/classifier.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 1. Configure the classifier
    xinfer::zoo::vision::ClassifierConfig config;
    config.engine_path = "assets/resnet50.engine";
    config.labels_path = "assets/imagenet_labels.txt";

    // 2. Initialize
    xinfer::zoo::vision::ImageClassifier classifier(config);

    // 3. Predict
    cv::Mat image = cv::imread("assets/dog.jpg");
    auto results = classifier.predict(image, 3); // Get top 3 results

    // 4. Print results
    std::cout << "Top 3 Predictions:\n";
    for (const auto& result : results) {
        printf(" - Label: %s, Confidence: %.4f\n", result.label.c_str(), result.confidence);
    }
}
```
**Config Struct:** `ClassifierConfig`
**Output Struct:** `ClassificationResult`

---

## `ObjectDetector`

Detects and localizes multiple objects within an image.

**Header:** `#include <xinfer/zoo/vision/detector.h>`

```cpp
#include <xinfer/zoo/vision/detector.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 1. Configure the detector
    xinfer::zoo::vision::DetectorConfig config;
    config.engine_path = "assets/yolov8n.engine";
    config.labels_path = "assets/coco.names";
    config.confidence_threshold = 0.5f;

    // 2. Initialize
    xinfer::zoo::vision::ObjectDetector detector(config);

    // 3. Predict
    cv::Mat image = cv::imread("assets/street.jpg");
    auto detections = detector.predict(image);

    // 4. Draw results
    for (const auto& box : detections) {
        cv::rectangle(image, { (int)box.x1, (int)box.y1 }, { (int)box.x2, (int)box.y2 }, {0, 255, 0}, 2);
    }
    cv::imwrite("detections_output.jpg", image);
    std::cout << "Saved annotated image to detections_output.jpg\n";
}
```
**Config Struct:** `DetectorConfig`
**Output Struct:** `BoundingBox`

---

## `Segmenter`

Performs semantic segmentation, assigning a class label to every pixel in an image.

**Header:** `#include <xinfer/zoo/vision/segmenter.h>`

```cpp
#include <xinfer/zoo/vision/segmenter.h>
#include <opencv2/opencv.hpp>

int main() {
    // 1. Configure the segmenter
    xinfer::zoo::vision::SegmenterConfig config;
    config.engine_path = "assets/segformer.engine";

    // 2. Initialize
    xinfer::zoo::vision::Segmenter segmenter(config);

    // 3. Predict
    cv::Mat image = cv::imread("assets/cityscape.jpg");
    cv::Mat class_mask = segmenter.predict(image); // Returns a CV_8UC1 mask

    // 4. Visualize the result
    cv::Mat color_mask;
    // (Here you would apply a colormap for visualization)
    cv::imwrite("segmentation_output.png", class_mask);
}
```
**Config Struct:** `SegmenterConfig`
**Output:** `cv::Mat` (single-channel, 8-bit integer mask)

---

## `InstanceSegmenter`

Performs instance segmentation, detecting individual object instances and providing a per-pixel mask for each one.

**Header:** `#include <xinfer/zoo/vision/instance_segmenter.h>`

```cpp
#include <xinfer/zoo/vision/instance_segmenter.h>
#include <opencv2/opencv.hpp>

int main() {
    // 1. Configure the instance segmenter
    xinfer::zoo::vision::InstanceSegmenterConfig config;
    config.engine_path = "assets/mask_rcnn.engine";
    config.labels_path = "assets/coco.names";

    // 2. Initialize
    xinfer::zoo::vision::InstanceSegmenter segmenter(config);

    // 3. Predict
    cv::Mat image = cv::imread("assets/people.jpg");
    auto results = segmenter.predict(image);

    // 4. Draw results
    for (const auto& instance : results) {
        // (Draw the instance.mask and instance.bounding_box on the image)
    }
    cv::imwrite("instance_segmentation_output.jpg", image);
}
```
**Config Struct:** `InstanceSegmenterConfig`
**Output Struct:** `InstanceSegmentationResult` (contains box, mask, label, etc.)

---

## `PoseEstimator`

Estimates the 2D keypoints of a human pose.

**Header:** `#include <xinfer/zoo/vision/pose_estimator.h>`

```cpp
#include <xinfer/zoo/vision/pose_estimator.h>
#include <opencv2/opencv.hpp>

int main() {
    xinfer::zoo::vision::PoseEstimatorConfig config;
    config.engine_path = "assets/rtmpose.engine";

    xinfer::zoo::vision::PoseEstimator estimator(config);

    cv::Mat image = cv::imread("assets/person_running.jpg");
    auto poses = estimator.predict(image);

    // Draw the keypoints for the first detected person
    if (!poses.empty()) {
        for (const auto& keypoint : poses) {
            if (keypoint.confidence > 0.5f) {
                cv::circle(image, { (int)keypoint.x, (int)keypoint.y }, 3, {0, 0, 255}, -1);
            }
        }
    }
    cv::imwrite("pose_output.jpg", image);
}
```
**Config Struct:** `PoseEstimatorConfig`
**Output:** `std::vector<Pose>` where `Pose` is a `std::vector<Keypoint>`

---

## And More...

This module provides many more specialized pipelines, each with a simple, consistent API.

- **`DepthEstimator`**: Predicts a dense depth map from a single RGB image.
- **`FaceDetector`**: A lightweight and fast detector specifically for faces.
- **`FaceRecognizer`**: Generates a 512-d feature embedding for a face, used for identification.
- **`HandTracker`**: Detects and tracks hands and their keypoints in real-time.
- **`OCR`**: A full, two-stage pipeline for detecting and recognizing text.
- **`ImageDeblur`**: Sharpens blurry images using a generative model.
- **`LowLightEnhancer`**: Brightens and denoises dark or nighttime images.
- **`SmokeFlameDetector`**: A specialized detector for industrial safety and wildfire monitoring.

*Each of these would have its own section with a code example, just like the ones above.*
