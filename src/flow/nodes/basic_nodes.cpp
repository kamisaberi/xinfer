#include <xinfer/flow/pipeline.h>
#include <xinfer/core/tensor.h>
#include <xinfer/core/logging.h>
#include <xinfer/zoo.h>

#include <opencv2/opencv.hpp>
#include <iostream>

namespace xinfer::flow {

// =============================================================================
// Source Node: Camera / Video File
// =============================================================================
class CameraSource : public INode {
    cv::VideoCapture cap;
    int cam_id = 0;
    std::string file_path;
    bool is_file = false;

public:
    void init(const std::map<std::string, std::string>& params) override {
        if (params.count("id")) {
            cam_id = std::stoi(params.at("id"));
            cap.open(cam_id);
            XINFER_LOG_INFO("Opened Camera ID: " + std::to_string(cam_id));
        } else if (params.count("file")) {
            file_path = params.at("file");
            cap.open(file_path);
            is_file = true;
            XINFER_LOG_INFO("Opened Video File: " + file_path);
        }

        if(!cap.isOpened()) {
            throw std::runtime_error("Failed to open video source.");
        }
    }

    Packet process(const Packet& input) override {
        Packet out = input;
        cv::Mat frame;
        if (!cap.read(frame)) {
            // Loop for files? Or Stop?
            // Simple loop logic:
            if (is_file) {
                cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                cap.read(frame);
            } else {
                out.is_eos = true;
            }
        }

        if (!frame.empty()) {
            out.data["image"] = frame;
        }
        return out;
    }
};

// =============================================================================
// Processing Node: Object Detector (Zoo Wrapper)
// =============================================================================
class ObjectDetectorNode : public INode {
    std::unique_ptr<zoo::vision::ObjectDetector> detector;
public:
    void init(const std::map<std::string, std::string>& params) override {
        zoo::vision::DetectorConfig cfg;

        if (params.count("model")) cfg.model_path = params.at("model");
        else throw std::runtime_error("DetectorNode: 'model' param required.");

        if (params.count("target")) {
            std::string t = params.at("target");
            // Simple parsing (In real app, use compiler::stringToTarget)
            if (t == "nv-trt") cfg.target = Target::NVIDIA_TRT;
            else if (t == "rockchip-rknn") cfg.target = Target::ROCKCHIP_RKNN;
            else if (t == "intel-ov") cfg.target = Target::INTEL_OV;
        }

        if (params.count("conf_threshold")) {
            cfg.confidence_threshold = std::stof(params.at("conf_threshold"));
        }

        detector = std::make_unique<zoo::vision::ObjectDetector>(cfg);
    }

    Packet process(const Packet& input) override {
        Packet out = input;

        if (input.data.count("image")) {
            cv::Mat img = std::any_cast<cv::Mat>(input.data.at("image"));

            // Run Zoo Module
            auto results = detector->predict(img);

            // Store results in packet
            out.data["detections"] = results;
        }
        return out;
    }
};

// =============================================================================
// Sink Node: Display Window
// =============================================================================
class DisplaySink : public INode {
    std::string window_name = "xInfer Output";
    bool draw_boxes = true;

public:
    void init(const std::map<std::string, std::string>& params) override {
        if (params.count("window_name")) window_name = params.at("window_name");
        if (params.count("draw_boxes")) draw_boxes = (params.at("draw_boxes") == "true");

        cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    }

    Packet process(const Packet& input) override {
        if (input.data.count("image")) {
            // Get image from packet (make a copy to draw on)
            cv::Mat img = std::any_cast<cv::Mat>(input.data.at("image")).clone();

            // Visualization Logic
            if (draw_boxes && input.data.count("detections")) {
                auto dets = std::any_cast<std::vector<zoo::vision::BoundingBox>>(input.data.at("detections"));
                for(const auto& d : dets) {
                    cv::Rect r((int)d.x1, (int)d.y1, (int)(d.x2-d.x1), (int)(d.y2-d.y1));
                    cv::rectangle(img, r, cv::Scalar(0, 255, 0), 2);
                    cv::putText(img, d.label, r.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,0), 2);
                }
            }

            cv::imshow(window_name, img);

            if (cv::waitKey(1) == 27) { // ESC key
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
// This static block executes when the library is loaded, registering nodes.
struct BasicNodesRegistrar {
    BasicNodesRegistrar() {
        Pipeline::register_node("CameraSource", [](){ return std::make_unique<CameraSource>(); });
        Pipeline::register_node("ObjectDetector", [](){ return std::make_unique<ObjectDetectorNode>(); });
        Pipeline::register_node("DisplaySink", [](){ return std::make_unique<DisplaySink>(); });
    }
};
static BasicNodesRegistrar registrar;

} // namespace xinfer::flow