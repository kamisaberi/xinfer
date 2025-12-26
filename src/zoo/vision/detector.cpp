#include <xinfer/zoo/vision/detector.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars of xInfer ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>

#include <fstream>
#include <iostream>

namespace xinfer::zoo::vision {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct ObjectDetector::Impl {
    DetectorConfig config_;

    // Abstract Interfaces (Polymorphic)
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preprocessor_;
    std::unique_ptr<postproc::IDetectionPostprocessor> postprocessor_;

    // Reusable Tensors (avoid allocation loop)
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    // Helper data
    std::vector<std::string> class_labels_;

    Impl(const DetectorConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Initialize Runtime Backend
        // ---------------------------------------------------------
        engine_ = backends::BackendFactory::create(config_.target);

        // Pass vendor params (e.g. "DLA=0" for Jetson, "CORE=ALL" for Rockchip)
        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("Failed to load model: " + config_.model_path);
        }

        // 2. Initialize Preprocessor
        // ---------------------------------------------------------
        // The factory automatically picks CUDA for NVIDIA, RGA for Rockchip, etc.
        preprocessor_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::RGB;
        // Most models expect NCHW. TFLite/Rockchip sometimes expect NHWC.
        // We assume NCHW standard here, or check target type.
        pre_cfg.layout_nchw = true;

        preprocessor_->init(pre_cfg);

        // 3. Initialize Postprocessor
        // ---------------------------------------------------------
        // The factory picks CUDA for NVIDIA, CPU (AVX/NEON) for others.
        postprocessor_ = postproc::create_detection(config_.target);

        postproc::DetectionConfig post_cfg;
        post_cfg.conf_threshold = config_.confidence_threshold;
        post_cfg.nms_threshold = config_.nms_iou_threshold;
        postprocessor_->init(post_cfg);

        // 4. Load Labels
        // ---------------------------------------------------------
        if (!config_.labels_path.empty()) {
            load_labels(config_.labels_path);
        }
    }

    void load_labels(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            XINFER_LOG_WARN("Could not open labels file: " + path);
            return;
        }
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            class_labels_.push_back(line);
        }
    }
};

// =================================================================================
// Public API
// =================================================================================

ObjectDetector::ObjectDetector(const DetectorConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

ObjectDetector::~ObjectDetector() = default;
ObjectDetector::ObjectDetector(ObjectDetector&&) noexcept = default;
ObjectDetector& ObjectDetector::operator=(ObjectDetector&&) noexcept = default;

std::vector<BoundingBox> ObjectDetector::predict(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("ObjectDetector is null.");

    // --- STEP 1: Abstraction Layer - Preprocessing ---
    // This call might run on CPU (NEON), GPU (CUDA), or Hardware 2D (RGA)
    // It handles resizing, normalization, and memory layout automatically.
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    frame.format = preproc::ImageFormat::BGR; // OpenCV default

    // Process writes directly to pimpl_->input_tensor
    pimpl_->preprocessor_->process(frame, pimpl_->input_tensor);

    // --- STEP 2: Abstraction Layer - Inference ---
    // Runs on TRT, OpenVINO, RKNN, QNN, etc.
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // --- STEP 3: Abstraction Layer - Postprocessing ---
    // If NVIDIA: Decodes & NMS on GPU, copies small result list to CPU.
    // If Rockchip: Decodes & NMS on CPU using optimized NEON/OpenCV.
    auto raw_results = pimpl_->postprocessor_->process({pimpl_->output_tensor});

    // --- STEP 4: Coordinate Mapping & Labeling ---
    // Map the boxes (0..640) back to original image size (e.g. 1920x1080)
    std::vector<BoundingBox> final_boxes;
    final_boxes.reserve(raw_results.size());

    float scale_x = (float)image.cols / pimpl_->config_.input_width;
    float scale_y = (float)image.rows / pimpl_->config_.input_height;

    for (const auto& res : raw_results) {
        BoundingBox box;
        // Scale back coordinates
        box.x1 = res.x1 * scale_x;
        box.y1 = res.y1 * scale_y;
        box.x2 = res.x2 * scale_x;
        box.y2 = res.y2 * scale_y;

        box.confidence = res.confidence;
        box.class_id = res.class_id;

        // Label lookup
        if (box.class_id >= 0 && box.class_id < (int)pimpl_->class_labels_.size()) {
            box.label = pimpl_->class_labels_[box.class_id];
        } else {
            box.label = "Class " + std::to_string(box.class_id);
        }

        final_boxes.push_back(box);
    }

    return final_boxes;
}

} // namespace xinfer::zoo::vision