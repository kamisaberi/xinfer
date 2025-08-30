# Zoo API: 3D & Spatial Computing

The `xinfer::zoo::threed` module provides high-level pipelines for processing and understanding 3D data, primarily from sensors like LIDAR or through multi-view reconstruction.

These classes are built on top of `xInfer`'s most advanced "F1 car" technologies. They abstract away extremely complex, non-standard GPU operations, allowing you to integrate state-of-the-art 3D AI into your C++ applications.

---

## `PointCloudDetector`

Performs 3D object detection directly on point cloud data. This is a fundamental task for autonomous driving and robotics.

**Header:** `#include <xinfer/zoo/threed/pointcloud_detector.h>`

```cpp
#include <xinfer/zoo/threed/pointcloud_detector.h>
#include <xinfer/core/tensor.h>
#include <iostream>
#include <vector>

int main() {
    // 1. Configure the 3D detector.
    xinfer::zoo::threed::PointCloudDetectorConfig config;
    config.engine_path = "assets/pointpillar.engine";
    config.labels_path = "assets/kitti_labels.txt"; // e.g., "Car", "Pedestrian", "Cyclist"

    // 2. Initialize.
    xinfer::zoo::threed::PointCloudDetector detector(config);

    // 3. Load LIDAR point cloud data and upload to a GPU tensor.
    //    (In a real app, this would come from a LIDAR sensor driver)
    std::vector<float> points; // Vector of [x, y, z, intensity] floats
    // ... load points from a .bin file ...
    xinfer::core::Tensor point_cloud({1, points.size() / 4, 4}, xinfer::core::DataType::kFLOAT);
    point_cloud.copy_from_host(points.data());

    // 4. Predict 3D bounding boxes.
    std::vector<xinfer::zoo::threed::BoundingBox3D> detections = detector.predict(point_cloud);

    // 5. Print results.
    std::cout << "Detected " << detections.size() << " objects:\n";
    for (const auto& box : detections) {
        std::cout << " - Label: " << box.label << ", Confidence: " << box.confidence << "\n";
    }
}
```
**Config Struct:** `PointCloudDetectorConfig`
**Input:** `xinfer::core::Tensor` containing the point cloud data.
**Output Struct:** `BoundingBox3D` (contains 3D center, dimensions, yaw, label, etc.).

---

## `PointCloudSegmenter`

Performs semantic segmentation on a point cloud, assigning a class label to every single point.

**Header:** `#include <xinfer/zoo/threed/pointcloud_segmenter.h>`

```cpp
#include <xinfer/zoo/threed/pointcloud_segmenter.h>
#include <xinfer/core/tensor.h>
#include <iostream>
#include <vector>

int main() {
    // 1. Configure the 3D segmenter.
    xinfer::zoo::threed::PointCloudSegmenterConfig config;
    config.engine_path = "assets/randlanet.engine";
    config.labels_path = "assets/semantic_kitti_labels.txt"; // e.g., "road", "building", "vegetation"

    // 2. Initialize.
    xinfer::zoo::threed::PointCloudSegmenter segmenter(config);

    // 3. Load LIDAR point cloud data.
    std::vector<float> points; // Vector of [x, y, z, intensity] floats
    // ... load points ...
    xinfer::core::Tensor point_cloud({1, points.size() / 4, 4}, xinfer::core::DataType::kFLOAT);
    point_cloud.copy_to_host(points.data());

    // 4. Predict per-point labels.
    std::vector<int> point_labels = segmenter.predict(point_cloud);

    // 5. Print results.
    std::cout << "Segmentation complete. Got " << point_labels.size() << " labels.\n";
    // You would now use these labels to colorize the point cloud for visualization.
}
```
**Config Struct:** `PointCloudSegmenterConfig`
**Input:** `xinfer::core::Tensor` containing the point cloud data.
**Output:** `std::vector<int>` of class IDs, one for each input point.

---

## `Reconstructor`

Reconstructs a 3D scene from a collection of 2D images, using a state-of-the-art neural rendering pipeline like 3D Gaussian Splatting.

**Header:** `#include <xinfer/zoo/threed/reconstructor.h>`

```cpp
#include <xinfer/zoo/threed/reconstructor.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    // 1. Configure the reconstructor.
    xinfer::zoo::threed::ReconstructorConfig config;
    config.num_iterations = 15000; // More iterations = higher quality

    // 2. Initialize. This sets up the custom CUDA pipeline.
    xinfer::zoo::threed::Reconstructor reconstructor(config);

    // 3. Load a set of images and their corresponding camera poses.
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> poses;
    // ... load images and camera poses from disk ...

    // 4. Run the reconstruction and meshing process.
    //    This is a heavy operation that trains the 3D representation and extracts a mesh.
    std::cout << "Starting 3D reconstruction...\n";
    xinfer::zoo::threed::Mesh3D result_mesh = reconstructor.predict(images, poses);

    // 5. Save the resulting mesh to a file.
    // (You would write the result_mesh.vertices and faces to an .obj file here)
    std::cout << "Reconstruction complete. Mesh has " << result_mesh.vertices.size() / 3 << " vertices.\n";
}
```
**Config Struct:** `ReconstructorConfig`
**Input:** `std::vector<cv::Mat>` of images and a `std::vector<cv::Mat>` of camera poses.
**Output Struct:** `Mesh3D` (contains vertices, faces, and vertex colors).
**"F1 Car" Technology:** This class is a wrapper around a full, from-scratch **custom CUDA pipeline** for training and meshing a neural representation like Gaussian Splatting.

---

## `SLAMAccelerator`

Provides hyper-optimized components to accelerate a Visual SLAM (Simultaneous Localization and Mapping) pipeline.

**Header:** `#include <xinfer/zoo/threed/slam_accelerator.h>`

```cpp
#include <xinfer/zoo/threed/slam_accelerator.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 1. Configure the SLAM accelerator.
    //    The engine would be a learned feature extractor like SuperPoint.
    xinfer::zoo::threed::SLAMAcceleratorConfig config;
    config.feature_engine_path = "assets/superpoint.engine";

    // 2. Initialize.
    xinfer::zoo::threed::SLAMAccelerator accelerator(config);

    // 3. In your main SLAM loop, use the accelerator for feature extraction.
    cv::Mat video_frame; // from camera
    xinfer::zoo::threed::SLAMFeatureResult features = accelerator.extract_features(video_frame);
    
    std::cout << "Extracted " << features.keypoints.size() << " keypoints.\n";
    // You would then pass these features to the tracking and mapping parts of your SLAM system.
}
```
**Config Struct:** `SLAMAcceleratorConfig`
**Input:** `cv::Mat` video frame.
**Output Struct:** `SLAMFeatureResult` (contains `cv::KeyPoint`s and `cv::Mat` descriptors).
