#include <xinfer/flow/pipeline.h>
#include <xinfer/core/logging.h>
#include <xinfer/zoo.h>
#include <iostream>

namespace xinfer::flow {

// -----------------------------------------------------------------------------
// Object Detector Node (Wraps zoo/vision/detector)
// -----------------------------------------------------------------------------
class ObjectDetectorNode : public INode {
    std::unique_ptr<zoo::vision::ObjectDetector> detector;
public:
    void init(const std::map<std::string, std::string>& params) override {
        zoo::vision::DetectorConfig cfg;

        if (!params.count("model")) throw std::runtime_error("DetectorNode: 'model' param missing");
        cfg.model_path = params.at("model");

        if (params.count("target")) {
            std::string t = params.at("target");
            cfg.target = compiler::stringToTarget(t); // Helper from base_compiler
        }

        if (params.count("conf_threshold")) cfg.confidence_threshold = std::stof(params.at("conf_threshold"));
        if (params.count("nms_threshold")) cfg.nms_iou_threshold = std::stof(params.at("nms_threshold"));

        detector = std::make_unique<zoo::vision::ObjectDetector>(cfg);
    }

    Packet process(const Packet& input) override {
        Packet out = input;
        if (input.data.count("image")) {
            cv::Mat img = std::any_cast<cv::Mat>(input.data.at("image"));

            // Run Inference
            auto results = detector->predict(img);

            // Store results
            out.data["detections"] = results;
        }
        return out;
    }
};

// -----------------------------------------------------------------------------
// Face Mesh Node (Wraps zoo/vision/face_mesh)
// -----------------------------------------------------------------------------
class FaceMeshNode : public INode {
    std::unique_ptr<zoo::vision::FaceMesh> mesher;
public:
    void init(const std::map<std::string, std::string>& params) override {
        zoo::vision::FaceMeshConfig cfg;
        cfg.model_path = params.at("model");
        if(params.count("target")) cfg.target = compiler::stringToTarget(params.at("target"));

        mesher = std::make_unique<zoo::vision::FaceMesh>(cfg);
    }

    Packet process(const Packet& input) override {
        Packet out = input;
        if (input.data.count("image")) {
            cv::Mat img = std::any_cast<cv::Mat>(input.data.at("image"));
            // NOTE: FaceMesh usually expects a crop.
            // In a real pipeline, previous node should crop or we run on full image.
            auto result = mesher->estimate(img);
            out.data["face_landmarks"] = result;
        }
        return out;
    }
};

// Register
struct InferNodesRegistrar {
    InferNodesRegistrar() {
        Pipeline::register_node("ObjectDetector", [](){ return std::make_unique<ObjectDetectorNode>(); });
        Pipeline::register_node("FaceMesh", [](){ return std::make_unique<FaceMeshNode>(); });
    }
};
static InferNodesRegistrar registrar;

} // namespace xinfer::flow