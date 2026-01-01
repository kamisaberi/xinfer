#include <xinfer/flow/pipeline.h>
#include <xinfer/core/logging.h>
#include <xinfer/zoo.h>

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

namespace xinfer::flow {

// -----------------------------------------------------------------------------
// Display Sink (GUI Window)
// -----------------------------------------------------------------------------
class DisplaySink : public INode {
    std::string window_name = "xInfer Flow";
    bool draw_boxes = true;

public:
    void init(const std::map<std::string, std::string>& params) override {
        if (params.count("window_name")) window_name = params.at("window_name");
        if (params.count("draw_boxes")) draw_boxes = (params.at("draw_boxes") == "true");

        // Only create window if we are in a GUI environment
        // In headless mode (server), this should probably log or no-op
        cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    }

    Packet process(const Packet& input) override {
        if (input.data.count("image")) {
            // Clone the image so we don't modify the one passed to other nodes
            cv::Mat img = std::any_cast<cv::Mat>(input.data.at("image")).clone();

            // Visualization Logic: Object Detection
            if (draw_boxes && input.data.count("detections")) {
                auto dets = std::any_cast<std::vector<zoo::vision::BoundingBox>>(input.data.at("detections"));
                for(const auto& d : dets) {
                    cv::rectangle(img, cv::Point(d.x1, d.y1), cv::Point(d.x2, d.y2), cv::Scalar(0, 255, 0), 2);
                    cv::putText(img, d.label, cv::Point(d.x1, d.y1-5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,0), 2);
                }
            }

            // Visualization Logic: Face Mesh
            if (input.data.count("face_landmarks")) {
                auto mesh = std::any_cast<zoo::vision::FaceMeshResult>(input.data.at("face_landmarks"));
                zoo::vision::FaceMesh::draw_mesh(img, mesh);
            }

            cv::imshow(window_name, img);
            int key = cv::waitKey(1);
            if (key == 27) { // ESC
                Packet out = input;
                out.is_eos = true;
                return out;
            }
        }
        return input;
    }
};

// -----------------------------------------------------------------------------
// Logger Sink (Print/Save Metadata)
// -----------------------------------------------------------------------------
class LoggerSink : public INode {
public:
    void init(const std::map<std::string, std::string>& params) override {
        // Init file if needed
    }

    Packet process(const Packet& input) override {
        if (input.data.count("detections")) {
            auto dets = std::any_cast<std::vector<zoo::vision::BoundingBox>>(input.data.at("detections"));
            if (!dets.empty()) {
                XINFER_LOG_INFO("Timestamp " + std::to_string(input.timestamp) +
                                " | Detected " + std::to_string(dets.size()) + " objects.");
            }
        }
        return input;
    }
};

// Register
struct SinkNodesRegistrar {
    SinkNodesRegistrar() {
        Pipeline::register_node("DisplaySink", [](){ return std::make_unique<DisplaySink>(); });
        Pipeline::register_node("LoggerSink", [](){ return std::make_unique<LoggerSink>(); });
    }
};
static SinkNodesRegistrar registrar;

} // namespace xinfer::flow