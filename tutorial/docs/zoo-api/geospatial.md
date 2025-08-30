# Zoo API: Geospatial

The `xinfer::zoo::geospatial` module provides a suite of high-performance pipelines for analyzing geospatial imagery from satellites, airplanes, and drones.

Processing massive, multi-channel satellite images is a significant computational challenge. The `zoo` classes in this module are built on `xInfer`'s hyper-optimized C++/TensorRT core to enable rapid, large-scale analysis that is often impractical with standard Python-based frameworks.

---

## `BuildingSegmenter`

Performs semantic segmentation on satellite imagery to extract the footprints of buildings.

**Header:** `#include <xinfer/zoo/geospatial/building_segmenter.h>`

### Use Case: Urban Planning and Insurance Risk Assessment

An urban planner needs to create a map of all buildings in a city, or an insurance company needs to assess the number of properties in a high-risk flood zone.

```cpp
#include <xinfer/zoo/geospatial/building_segmenter.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 1. Configure the building segmenter.
    xinfer::zoo::geospatial::BuildingSegmenterConfig config;
    config.engine_path = "assets/building_segmenter.engine";
    config.probability_threshold = 0.5f;

    // 2. Initialize.
    xinfer::zoo::geospatial::BuildingSegmenter segmenter(config);

    // 3. Load a large satellite or aerial image.
    cv::Mat satellite_image = cv::imread("assets/city_orthomosaic.tif");

    // 4. Predict the binary mask of all buildings.
    cv::Mat building_mask = segmenter.predict_mask(satellite_image);

    // 5. (Optional) Convert the mask to vector polygons for use in GIS software.
    std::vector<xinfer::zoo::geospatial::BuildingPolygon> polygons = segmenter.predict_polygons(satellite_image);

    std::cout << "Found " << polygons.size() << " building footprints.\n";
    cv::imwrite("building_footprints.png", building_mask);
}
```
**Config Struct:** `BuildingSegmenterConfig`
**Methods:**
- `predict_mask(const cv::Mat&)` returns a `cv::Mat` binary mask.
- `predict_polygons(const cv::Mat&)` returns a `std::vector<BuildingPolygon>`.
  **"F1 Car" Technology:** This class internally tiles the large input image, runs a high-performance U-Net engine on each tile, and stitches the results back together. The polygonization step uses efficient OpenCV algorithms.

---

## `RoadExtractor`

Performs semantic segmentation to extract the road network from satellite imagery.

**Header:** `#include <xinfer/zoo/geospatial/road_extractor.h>`

### Use Case: Logistics and Infrastructure Mapping

A logistics company wants to create an up-to-date road map for a remote area, or a government agency needs to assess road infrastructure.

```cpp
#include <xinfer/zoo/geospatial/road_extractor.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    xinfer::zoo::geospatial::RoadExtractorConfig config;
    config.engine_path = "assets/road_extractor.engine";

    xinfer::zoo::geospatial::RoadExtractor extractor(config);

    cv::Mat satellite_image = cv::imread("assets/rural_area.tif");
    cv::Mat road_mask = extractor.predict_mask(satellite_image);
    
    // The mask can be further processed into a graph-based road network.
    std::cout << "Road network mask generated.\n";
    cv::imwrite("road_network.png", road_mask);
}
```
**Config Struct:** `RoadExtractorConfig`
**Methods:**
- `predict_mask(const cv::Mat&)` returns a `cv::Mat` binary mask.
- `predict_segments(const cv::Mat&)` returns a `std::vector<RoadSegment>` (contours).

---

## `MaritimeDetector`

Detects and classifies maritime objects (ships, boats) in satellite or aerial imagery.

**Header:** `#include <xinfer/zoo/geospatial/maritime_detector.h>`

### Use Case: Port Security and Maritime Surveillance

A coast guard or port authority needs to monitor maritime traffic and identify specific vessels in a large area of open water or a crowded port.

```cpp
#include <xinfer/zoo/geospatial/maritime_detector.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    xinfer::zoo::geospatial::MaritimeDetectorConfig config;
    config.engine_path = "assets/ship_detector.engine";
    config.labels_path = "assets/ship_classes.txt"; // e.g., "Cargo Ship", "Tanker", "Fishing Vessel"

    xinfer::zoo::geospatial::MaritimeDetector detector(config);

    cv::Mat coastal_image = cv::imread("assets/port_image.jpg");
    auto detections = detector.predict(coastal_image);

    std::cout << "Detected " << detections.size() << " maritime objects:\n";
    for (const auto& obj : detections) {
        // Draw the rotated bounding box
        cv::polylines(coastal_image, obj.contour, true, {0, 255, 0}, 2);
        std::cout << " - " << obj.label << " (Confidence: " << obj.confidence << ")\n";
    }
    cv::imwrite("maritime_detections.jpg", coastal_image);
}
```
**Config Struct:** `MaritimeDetectorConfig`
**Input:** `cv::Mat` satellite/aerial image.
**Output Struct:** `DetectedObject` (contains class, confidence, and a contour for rotated bounding boxes).

---

## `DisasterAssessor`

Compares pre- and post-disaster satellite images to automatically map and assess the extent of damage.

**Header:** `#include <xinfer/zoo/geospatial/disaster_assessor.h>`

### Use Case: Emergency Response and Insurance Assessment

After a hurricane or wildfire, emergency response teams and insurance companies need to rapidly determine which buildings have been damaged or destroyed.

```cpp
#include <xinfer/zoo/geospatial/disaster_assessor.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    xinfer::zoo::geospatial::DisasterAssessorConfig config;
    config.engine_path = "assets/damage_assessor.engine";

    xinfer::zoo::geospatial::DisasterAssessor assessor(config);

    cv::Mat image_before = cv::imread("assets/town_before_hurricane.png");
    cv::Mat image_after = cv::imread("assets/town_after_hurricane.png");

    // The model takes both images as input and identifies the changes.
    cv::Mat damage_mask = assessor.predict(image_before, image_after);

    std::cout << "Damage assessment mask generated.\n";
    cv::imwrite("damage_mask.png", damage_mask);
}
```
**Config Struct:** `DisasterAssessorConfig`
**Input:** Two `cv::Mat` images (pre- and post-disaster).
**Output:** A `cv::Mat` binary mask where white pixels indicate damaged areas.

---

## `CropMonitor`

Analyzes multi-spectral satellite imagery to assess crop health and calculate common agricultural indices.

**Header:** `#include <xinfer/zoo/geospatial/crop_monitor.h>`

### Use Case: Precision Agriculture

A large farm or agricultural cooperative needs to monitor the health of thousands of acres of crops to apply fertilizer and water more efficiently.

```cpp
#include <xinfer/zoo/geospatial/crop_monitor.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // In a real app, this would be a multi-channel GeoTIFF file
    cv::Mat multispectral_image; // Assume this is loaded with B, G, R, NIR channels

    // You can perform classic analysis without a model:
    cv::Mat ndvi_map = xinfer::zoo::geospatial::CropMonitor::calculate_ndvi(multispectral_image);
    cv::imwrite("ndvi_map.png", ndvi_map);
    std::cout << "NDVI map calculated and saved.\n";

    // Or use a trained AI model for a more advanced health prediction:
    xinfer::zoo::geospatial::CropMonitorConfig config;
    config.engine_path = "assets/crop_health_model.engine";
    xinfer::zoo::geospatial::CropMonitor monitor(config);
    cv::Mat health_map = monitor.predict_health_map(multispectral_image);
    cv::imwrite("health_map.png", health_map);
    std::cout << "AI-based health map generated and saved.\n";
}
```
**Config Struct:** `CropMonitorConfig`
**Methods:**
- `predict_health_map(const cv::Mat&)` returns a `cv::Mat` health score map.
- `calculate_ndvi(const cv::Mat&)` is a static utility function that returns a `cv::Mat` NDVI map.
