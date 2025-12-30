#include <xinfer/flow/pipeline.h>
#include <xinfer/core/tensor.h>
#include <xinfer/zoo.h> // Access the whole zoo

#include <opencv2/opencv.hpp>
#include <iostream>

namespace xinfer::flow {

// =============================================================================
// Source Node: Camera
// =============================================================================
class CameraNode : public INode {
    cv::VideoCapture cap;
    int cam_id = 0;
public:
    void init(const std::map<std::string, std::string>& params) override {
        if (params.count("id")) cam_id = std::stoi(params.at("id"));
        cap.open(cam_id);
        if(!cap.isOpened()) throw std::runtime_error("Failed to open camera " + std::to_string(cam_id));
    }

    Packet process(const Packet& input) override {
        Packet out = input;
        cv::Mat frame;
        if (!cap.read(frame)) {
            out.is_eos = true;
        } else {
            // Put raw CV Mat into Any
            out.data["image"] = frame;
        }
        return out;
    }
};

// =============================================================================
// Inference Node: Object Detector (YOLO)
// =============================================================================
class DetectorNode : public INode {
    std::unique_ptr<zoo::vision::ObjectDetector> detector;
public:
    void init(const std::map<std::string, std::string>& params) override {
        zoo::vision::DetectorConfig cfg;
        cfg.model_path = params.at("model");

        // Parse target string to Enum
        std::string t = params.at("target"); // "nv-trt", etc.
        cfg.target = compiler::stringToTarget(t);

        detector = std::make_unique<zoo::vision::ObjectDetector>(cfg);
    }

    Packet process(const Packet& input) override {
        Packet out = input;

        // Extract image
        if (input.data.count("image")) {
            cv::Mat img = std::any_cast<cv::Mat>(input.data.at("image"));

            // Run Zoo Module
            auto results = detector->predict(img);

            // Store results
            out.data["detections"] = results;
        }
        return out;
    }
};

// =============================================================================
// Sink Node: Display
// =============================================================================
class DisplayNode : public INode {
public:
    void init(const std::map<std::string, std::string>& params) override {
        // Init window
    }

    Packet process(const Packet& input) override {
        if (input.data.count("image")) {
            cv::Mat img = std::any_cast<cv::Mat>(input.data.at("image"));

            // Draw boxes if available
            if (input.data.count("detections")) {
                auto dets = std::any_cast<std::vector<zoo::vision::BoundingBox>>(input.data.at("detections"));
                for(auto& d : dets) {
                    cv::rectangle(img, cv::Rect(d.x1, d.y1, d.x2-d.x1, d.y2-d.y1), cv::Scalar(0,255,0), 2);
                }
            }

            cv::imshow("xInfer Flow", img);
            if (cv::waitKey(1) == 27) {
                Packet p = input;
                p.is_eos = true;
                return p;
            }
        }
        return input;
    }
};

// =============================================================================
// Auto-Registration
// =============================================================================
// This static block runs on startup to register the nodes
struct NodeRegistrar {
    NodeRegistrar() {
        Pipeline::register_node("CameraSource", [](){ return std::make_unique<CameraNode>(); });
        Pipeline::register_node("ObjectDetector", [](){ return std::make_unique<DetectorNode>(); });
        Pipeline::register_node("DisplaySink", [](){ return std::make_unique<DisplayNode>(); });
    }
};
static NodeRegistrar registrar;

} // namespace xinfer::flow