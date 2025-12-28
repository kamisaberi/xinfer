#include <xinfer/zoo/accessibility/visual_assistant.h>
#include <xinfer/core/logging.h>

// --- We compose multiple Zoo modules ---
#include <xinfer/zoo/vision/detector.h>
#include <xinfer/zoo/vision/depth_estimator.h>
#include <xinfer/zoo/vision/ocr.h>
#include <xinfer/zoo/nlp/text_generator.h>

#include <iostream>
#include <sstream>
#include <algorithm>

namespace xinfer::zoo::accessibility {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct VisualAssistant::Impl {
    AssistantConfig config_;

    // --- Components ---
    std::unique_ptr<vision::ObjectDetector> detector_;
    std::unique_ptr<vision::DepthEstimator> depth_estimator_;
    std::unique_ptr<vision::OcrRecognizer> ocr_;
    std::unique_ptr<nlp::TextGenerator> llm_;

    Impl(const AssistantConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Init Detector
        vision::DetectorConfig det_cfg;
        det_cfg.target = config_.target;
        det_cfg.model_path = config_.detector_path;
        det_cfg.labels_path = config_.det_labels_path;
        detector_ = std::make_unique<vision::ObjectDetector>(det_cfg);

        // 2. Init Depth Estimator
        vision::DepthEstimatorConfig depth_cfg;
        depth_cfg.target = config_.target;
        depth_cfg.model_path = config_.depth_model_path;
        depth_estimator_ = std::make_unique<vision::DepthEstimator>(depth_cfg);

        // 3. Init OCR
        vision::OcrConfig ocr_cfg;
        ocr_cfg.target = config_.target;
        ocr_cfg.model_path = config_.ocr_model_path;
        // ... set ocr vocab etc.
        ocr_ = std::make_unique<vision::OcrRecognizer>(ocr_cfg);

        // 4. Init LLM
        nlp::TextGenConfig llm_cfg;
        llm_cfg.target = config_.target;
        llm_cfg.model_path = config_.llm_path;
        llm_cfg.tokenizer_path = config_.llm_tokenizer_path;
        llm_cfg.max_new_tokens = 64; // Short descriptions
        llm_cfg.temperature = 0.5f;
        llm_ = std::make_unique<nlp::TextGenerator>(llm_cfg);
    }

    // --- Core Logic: Structured Data -> Natural Language Prompt ---
    std::string build_prompt(const std::vector<vision::BoundingBox>& objects,
                             const cv::Mat& depth_map,
                             const std::vector<std::string>& ocr_texts) {

        std::stringstream ss;
        ss << "You are a visual assistant. Describe the scene based on the following data. Be concise.\n";
        ss << "OBJECTS:\n";

        for (const auto& obj : objects) {
            // Find distance from depth map
            int cx = (int)(obj.x1 + obj.x2) / 2;
            int cy = (int)(obj.y1 + obj.y2) / 2;

            float distance = 0.0f;
            if (cy < depth_map.rows && cx < depth_map.cols) {
                distance = depth_map.at<float>(cy, cx);
            }

            ss << "- A " << obj.label << " is " << std::fixed << std::setprecision(1) << distance << " meters away.\n";
        }

        if (!ocr_texts.empty()) {
            ss << "TEXTS:\n";
            for (const auto& txt : ocr_texts) {
                ss << "- \"" << txt << "\"\n";
            }
        }

        ss << "SUMMARY:";
        return ss.str();
    }
};

// =================================================================================
// Public API
// =================================================================================

VisualAssistant::VisualAssistant(const AssistantConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

VisualAssistant::~VisualAssistant() = default;
VisualAssistant::VisualAssistant(VisualAssistant&&) noexcept = default;
VisualAssistant& VisualAssistant::operator=(VisualAssistant&&) noexcept = default;

SceneDescription VisualAssistant::describe_scene(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("VisualAssistant is null.");

    SceneDescription result;

    // 1. Run Vision Models (can be parallelized on GPU)
    auto objects = pimpl_->detector_->predict(image);
    auto depth = pimpl_->depth_estimator_->estimate(image);

    // For OCR, we can be smart and only run it on detected "sign" objects
    // or run it on the whole image (slower). Here, we simulate full-image OCR.
    // auto ocr_res = pimpl_->ocr_->recognize(image);
    // result.readable_text.push_back(ocr_res.text);

    // 2. Immediate Obstacle Check
    result.has_obstacles = false;
    for (const auto& obj : objects) {
        int cx = (int)(obj.x1 + obj.x2) / 2;
        int cy = (int)(obj.y1 + obj.y2) / 2;
        float dist = depth.depth_raw.at<float>(cy, cx);

        if (dist > 0 && dist < pimpl_->config_.obstacle_distance_m) {
            result.has_obstacles = true;
            result.immediate_obstacle_info = "Obstacle ahead: " + obj.label + " at " + std::to_string(dist) + " meters.";
            break; // Prioritize closest obstacle
        }
    }

    // 3. Build Prompt & Generate Narrative
    std::string prompt = pimpl_->build_prompt(objects, depth.depth_raw, result.readable_text);
    result.narrative = pimpl_->llm_->generate(prompt);

    return result;
}

} // namespace xinfer::zoo::accessibility