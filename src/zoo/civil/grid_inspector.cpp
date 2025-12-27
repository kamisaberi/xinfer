#include <xinfer/zoo/civil/grid_inspector.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/classification_interface.h>

// We reuse the generic detector
#include <xinfer/zoo/vision/detector.h>

#include <iostream>
#include <fstream>

namespace xinfer::zoo::civil {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct GridInspector::Impl {
    GridConfig config_;

    // --- Components ---
    std::unique_ptr<vision::ObjectDetector> component_detector_;
    std::unique_ptr<backends::IBackend> fault_classifier_;
    std::unique_ptr<preproc::IImagePreprocessor> cls_preproc_;
    std::unique_ptr<postproc::IClassificationPostprocessor> cls_postproc_;

    // --- Tensors ---
    core::Tensor cls_input, cls_output;

    Impl(const GridConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Setup Component Detector
        vision::DetectorConfig det_cfg;
        det_cfg.target = config_.target;
        det_cfg.model_path = config_.detector_path;
        det_cfg.labels_path = config_.component_labels_path;
        det_cfg.input_width = config_.det_input_width;
        det_cfg.input_height = config_.det_input_height;
        det_cfg.confidence_threshold = config_.det_conf_thresh;

        component_detector_ = std::make_unique<vision::ObjectDetector>(det_cfg);

        // 2. Setup Fault Classifier
        fault_classifier_ = backends::BackendFactory::create(config_.target);
        xinfer::Config c_cfg; c_cfg.model_path = config_.classifier_path;

        if (!fault_classifier_->load_model(c_cfg.model_path)) {
            throw std::runtime_error("GridInspector: Failed to load classifier.");
        }

        cls_preproc_ = preproc::create_image_preprocessor(config_.target);
        preproc::ImagePreprocConfig cp_cfg;
        cp_cfg.target_width = config_.cls_input_width;
        cp_cfg.target_height = config_.cls_input_height;
        cls_preproc_->init(cp_cfg);

        cls_postproc_ = postproc::create_classification(config_.target);
        postproc::ClassificationConfig cpost_cfg;
        cpost_cfg.top_k = 1;
        cpost_cfg.apply_softmax = true;

        // Load fault labels
        std::vector<std::string> fault_labels;
        std::ifstream file(config_.fault_labels_path);
        if (file.is_open()) {
            std::string line;
            while(std::getline(file, line)) fault_labels.push_back(line);
        }
        cpost_cfg.labels = fault_labels;
        cls_postproc_->init(cpost_cfg);
    }
};

// =================================================================================
// Public API
// =================================================================================

GridInspector::GridInspector(const GridConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

GridInspector::~GridInspector() = default;
GridInspector::GridInspector(GridInspector&&) noexcept = default;
GridInspector& GridInspector::operator=(GridInspector&&) noexcept = default;

GridInspectionResult GridInspector::inspect(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("GridInspector is null.");

    GridInspectionResult result;
    result.annotated_image = image.clone();
    result.requires_maintenance = false;

    // 1. Detect Components
    auto components = pimpl_->component_detector_->predict(image);

    // 2. Classify Each Component
    for (const auto& comp : components) {
        // Crop component with a small margin
        int x1 = std::max(0, (int)comp.x1 - 10);
        int y1 = std::max(0, (int)comp.y1 - 10);
        int x2 = std::min(image.cols, (int)comp.x2 + 10);
        int y2 = std::min(image.rows, (int)comp.y2 + 10);

        if (x2 - x1 <= 0 || y2 - y1 <= 0) continue;

        cv::Mat crop = image(cv::Rect(x1, y1, x2 - x1, y2 - y1));

        // Preprocess crop for classifier
        preproc::ImageFrame frame{crop.data, crop.cols, crop.rows, preproc::ImageFormat::BGR};
        pimpl_->cls_preproc_->process(frame, pimpl_->cls_in);

        // Inference
        pimpl_->cls_postproc_->engine_->predict({pimpl_->cls_in}, {pimpl_->cls_out});

        // Postprocess
        auto cls_results = pimpl_->cls_postproc_->process(pimpl_->cls_out);

        if (!cls_results.empty() && !cls_results[0].empty()) {
            auto& top1 = cls_results[0][0];

            // Only report if it's a fault (not "OK")
            if (top1.label != "OK" && top1.score > pimpl_->config_.cls_conf_thresh) {
                GridFault fault;
                fault.component_type = comp.label;
                fault.fault_type = top1.label;
                fault.confidence = top1.score;
                fault.box = { (float)x1, (float)y1, (float)x2, (float)y2, comp.confidence, comp.class_id };

                result.faults.push_back(fault);
                result.requires_maintenance = true;
            }
        }
    }

    // 3. Visualization
    for (const auto& f : result.faults) {
        cv::Rect r((int)f.box.x1, (int)f.box.y1, (int)(f.box.x2-f.box.x1), (int)(f.box.y2-f.box.y1));
        cv::rectangle(result.annotated_image, r, cv::Scalar(0, 0, 255), 2);

        std::string text = f.component_type + ": " + f.fault_type;
        cv::putText(result.annotated_image, text, r.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,255), 2);
    }

    return result;
}

} // namespace xinfer::zoo::civil