#include <xinfer/flow/pipeline.h>
#include <xinfer/core/logging.h>
#include <opencv2/opencv.hpp>
#include <iostream>

namespace xinfer::flow {

// -----------------------------------------------------------------------------
// Camera Source (Webcam / CSI Camera)
// -----------------------------------------------------------------------------
class CameraSource : public INode {
    cv::VideoCapture cap;
    int cam_id = 0;
public:
    void init(const std::map<std::string, std::string>& params) override {
        if (params.count("id")) cam_id = std::stoi(params.at("id"));

        cap.open(cam_id);
        if(!cap.isOpened()) {
            throw std::runtime_error("CameraSource: Failed to open ID " + std::to_string(cam_id));
        }

        // Optional: Set Resolution
        if(params.count("width")) cap.set(cv::CAP_PROP_FRAME_WIDTH, std::stoi(params.at("width")));
        if(params.count("height")) cap.set(cv::CAP_PROP_FRAME_HEIGHT, std::stoi(params.at("height")));
    }

    Packet process(const Packet& input) override {
        Packet out = input;
        cv::Mat frame;
        if (!cap.read(frame)) {
            out.is_eos = true;
        } else {
            out.data["image"] = frame;
        }
        return out;
    }
};

// -----------------------------------------------------------------------------
// File Source (Video File)
// -----------------------------------------------------------------------------
class FileSource : public INode {
    cv::VideoCapture cap;
    bool loop = false;
public:
    void init(const std::map<std::string, std::string>& params) override {
        if (!params.count("path")) throw std::runtime_error("FileSource: 'path' param missing.");

        std::string path = params.at("path");
        if (params.count("loop")) loop = (params.at("loop") == "true");

        cap.open(path);
        if(!cap.isOpened()) throw std::runtime_error("FileSource: Failed to open " + path);
    }

    Packet process(const Packet& input) override {
        Packet out = input;
        cv::Mat frame;
        if (!cap.read(frame)) {
            if (loop) {
                cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                cap.read(frame);
                if (!frame.empty()) out.data["image"] = frame;
            } else {
                out.is_eos = true;
            }
        } else {
            out.data["image"] = frame;
        }
        return out;
    }
};

// Register
struct SourceNodesRegistrar {
    SourceNodesRegistrar() {
        Pipeline::register_node("CameraSource", [](){ return std::make_unique<CameraSource>(); });
        Pipeline::register_node("FileSource", [](){ return std::make_unique<FileSource>(); });
    }
};
static SourceNodesRegistrar registrar;

} // namespace xinfer::flow