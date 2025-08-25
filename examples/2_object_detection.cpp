#include <include/zoo/vision/detector.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Pre-build your YOLO engine:
    // xinfer-cli build --onnx yolov8.onnx --save_engine yolov8.engine --fp16

    xinfer::zoo::vision::DetectorConfig config;
    config.engine_path = "assets/yolov8.engine";
    config.labels_path = "assets/coco.names";
    config.confidence_threshold = 0.5f;

    xinfer::zoo::vision::ObjectDetector detector(config);

    cv::Mat image = cv::imread("assets/street_scene.jpg");
    std::vector<xinfer::zoo::vision::BoundingBox> detections = detector.predict(image);

    std::cout << "Found " << detections.size() << " objects in street_scene.jpg:\n";
    for (const auto& box : detections) {
        std::cout << " - " << box.label << " (Confidence: " << box.confidence << ")\n";
        // Draw the bounding box on the image
        cv::rectangle(image, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2), cv::Scalar(0, 255, 0), 2);
        cv::putText(image, box.label, cv::Point(box.x1, box.y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }

    cv::imwrite("detection_output.jpg", image);
    std::cout << "Saved annotated image to detection_output.jpg\n";
    return 0;
}