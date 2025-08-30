# Zoo API: Medical Imaging

The `xinfer::zoo::medical` module provides a suite of high-performance, specialized pipelines for medical image analysis.

Developing AI for healthcare requires the highest standards of performance, reliability, and precision. The classes in this module are designed to be integrated into clinical research, diagnostic workflows, and medical devices. They are built on top of `xInfer`'s hyper-optimized C++/TensorRT core to provide the low-latency, high-throughput processing that is essential for modern medical applications.

!!! warning "For Research Use Only"
    The `zoo::medical` pipelines are powerful tools for research and development. They are not certified as medical devices. Any clinical or diagnostic application built with `xInfer` must undergo its own separate and rigorous regulatory approval process (e.g., FDA, CE).

---

## `TumorDetector`

Performs 3D object detection on medical scans (like CT or MRI) to identify and localize potential tumors.

**Header:** `#include <xinfer/zoo/medical/tumor_detector.h>`

```cpp
#include <xinfer/zoo/medical/tumor_detector.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    // 1. Configure the 3D tumor detector.
    xinfer::zoo::medical::TumorDetectorConfig config;
    config.engine_path = "assets/lung_nodule_detector_3d.engine";
    config.labels_path = "assets/nodule_types.txt";

    // 2. Initialize.
    xinfer::zoo::medical::TumorDetector detector(config);

    // 3. Load a series of 2D CT scan slices that form a 3D volume.
    std::vector<cv::Mat> ct_scan_slices;
    // ... load slices from DICOM files ...

    // 4. Predict 3D bounding boxes for all potential tumors.
    std::vector<xinfer::zoo::medical::Tumor> tumors = detector.predict(ct_scan_slices);

    // 5. Print results.
    std::cout << "Found " << tumors.size() << " potential tumors:\n";
    for (const auto& tumor : tumors) {
        std::cout << " - Type: " << tumor.label << ", Confidence: " << tumor.confidence << "\n";
    }
}
```
**Config Struct:** `TumorDetectorConfig`
**Input:** `std::vector<cv::Mat>` representing the 3D scan volume.
**Output Struct:** `Tumor` (contains 3D bounding box, class, and confidence).
**"F1 Car" Technology:** This pipeline is built to run a 3D CNN (like a 3D U-Net) and would use `xInfer`'s planned **3D NMS** post-processing kernel for maximum performance.

---

## `CellSegmenter`

Performs instance segmentation on microscope images to identify and count individual cells.

**Header:** `#include <xinfer/zoo/medical/cell_segmenter.h>`

```cpp
#include <xinfer/zoo/medical/cell_segmenter.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 1. Configure the cell segmenter.
    xinfer::zoo::medical::CellSegmenterConfig config;
    config.engine_path = "assets/cell_unet.engine";
    config.probability_threshold = 0.5f;

    // 2. Initialize.
    xinfer::zoo::medical::CellSegmenter segmenter(config);

    // 3. Process a microscope image.
    cv::Mat microscope_image = cv::imread("assets/blood_smear.png");
    auto result = segmenter.predict(microscope_image);

    // 4. Print the result and save the instance mask.
    std::cout << "Detected " << result.cell_count << " individual cells.\n";
    // result.instance_mask is a CV_32S Mat where each cell has a unique integer ID.
    // (This can be colorized for visualization).
    cv::imwrite("cell_instance_mask.png", result.instance_mask);
}
```
**Config Struct:** `CellSegmenterConfig`
**Input:** `cv::Mat` microscope image.
**Output Struct:** `CellSegmentationResult` (contains an instance mask, cell count, and contours).

---

## `RetinaScanner`

Analyzes retinal fundus images to detect and grade signs of diabetic retinopathy.

**Header:** `#include <xinfer/zoo/medical/retina_scanner.h>`

```cpp
#include <xinfer/zoo/medical/retina_scanner.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    xinfer::zoo::medical::RetinaScannerConfig config;
    config.engine_path = "assets/retinopathy_classifier.engine";

    xinfer::zoo::medical::RetinaScanner scanner(config);

    cv::Mat fundus_image = cv::imread("assets/retina_scan.jpg");
    auto result = scanner.predict(fundus_image);

    std::cout << "Retina Scan Analysis:\n";
    std::cout << " - Diagnosis: " << result.diagnosis
              << " (Grade " << result.severity_grade << ")\n"
              << " - Confidence: " << result.confidence << "\n";
    
    // The heatmap can be overlaid on the original image to show areas the AI focused on.
    cv::imwrite("lesion_heatmap.png", result.lesion_heatmap);
}
```
**Config Struct:** `RetinaScannerConfig`
**Input:** `cv::Mat` fundus image.
**Output Struct:** `RetinaScanResult` (contains diagnosis, severity grade, confidence, and a `cv::Mat` heatmap).

---

## `UltrasoundGuide`

Performs real-time segmentation on ultrasound video feeds to assist medical professionals.

**Header:** `#include <xinfer/zoo/medical/ultrasound_guide.h>`

```cpp
#include <xinfer/zoo/medical/ultrasound_guide.h>
#include <opencv2/opencv.hpp>

int main() {
    xinfer::zoo::medical::UltrasoundGuideConfig config;
    config.engine_path = "assets/ultrasound_nerve_segmenter.engine";

    xinfer::zoo::medical::UltrasoundGuide guide(config);

    cv::VideoCapture cap(0); // Open a live camera/ultrasound feed
    cv::Mat frame;
    while (cap.read(frame)) {
        // Run the segmentation pipeline in real-time
        auto result = guide.predict(frame);

        // Overlay the segmentation mask on the live feed for guidance
        cv::addWeighted(frame, 1.0, result.segmentation_mask, 0.4, 0.0, frame);
        cv::imshow("Ultrasound Guidance", frame);
        if (cv::waitKey(1) == 27) break; // Exit on ESC
    }
}
```
**Config Struct:** `UltrasoundGuideConfig`
**Input:** `cv::Mat` ultrasound video frame.
**Output Struct:** `UltrasoundGuideResult` (contains a binary segmentation mask and contours).

---

## `PathologyAssistant`

Analyzes gigapixel whole-slide images to detect and quantify mitotic activity, assisting pathologists in cancer grading.

**Header:** `#include <xinfer/zoo/medical/pathology_assistant.h>`

```cpp
#include <xinfer/zoo/medical/pathology_assistant.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    xinfer::zoo::medical::PathologyAssistantConfig config;
    config.engine_path = "assets/mitosis_detector.engine";
    config.batch_size = 16; // Process 16 tiles at a time for high throughput

    xinfer::zoo::medical::PathologyAssistant assistant(config);

    // In a real app, this would use a library like OpenSlide to read a gigapixel .svs file
    cv::Mat whole_slide_image = cv::imread("assets/pathology_slide_large.tif");

    auto result = assistant.predict(whole_slide_image);

    std::cout << "Pathology Slide Analysis:\n";
    std::cout << " - Overall Mitotic Score: " << result.overall_mitotic_score << std::endl;
    
    cv::imwrite("mitotic_heatmap.png", result.mitotic_heatmap);
}
```
**Config Struct:** `PathologyAssistantConfig`
**Input:** `cv::Mat` representing a whole-slide image.
**Output Struct:** `PathologyResult` (contains a summary score and a `cv::Mat` heatmap).
**"F1 Car" Technology:** This class internally handles the complex logic of **tiling** a massive image, running inference on batches of tiles, and stitching the results back together into a final heatmap.