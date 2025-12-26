#include <xinfer/zoo/vision/face_detector.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>

#include <iostream>

namespace xinfer::zoo::vision {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct FaceDetector::Impl {
    FaceDetectorConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::IDetectionPostprocessor> postproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    Impl(const FaceDetectorConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("FaceDetector: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::RGB;
        pre_cfg.layout_nchw = true;
        preproc_->init(pre_cfg);

        // 3. Setup Postprocessor
        // We use the generic Detection processor (YOLO style)
        postproc_ = postproc::create_detection(config_.target);

        postproc::DetectionConfig post_cfg;
        post_cfg.conf_threshold = config_.conf_threshold;
        post_cfg.nms_threshold = config_.nms_threshold;
        // Face detection usually implies 1 class (Face), but model might have more.
        post_cfg.num_classes = 1;
        postproc_->init(post_cfg);
    }
};

// =================================================================================
// Public API
// =================================================================================

FaceDetector::FaceDetector(const FaceDetectorConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

FaceDetector::~FaceDetector() = default;
FaceDetector::FaceDetector(FaceDetector&&) noexcept = default;
FaceDetector& FaceDetector::operator=(FaceDetector&&) noexcept = default;

std::vector<FaceResult> FaceDetector::detect(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("FaceDetector is null.");

    // 1. Preprocess
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    frame.format = preproc::ImageFormat::BGR;

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess
    // The generic detector returns BoundingBox structs
    auto raw_detections = pimpl_->postproc_->process({pimpl_->output_tensor});

    // 4. Map to FaceResult & Rescale
    std::vector<FaceResult> faces;
    faces.reserve(raw_detections.size());

    float scale_x = (float)image.cols / pimpl_->config_.input_width;
    float scale_y = (float)image.rows / pimpl_->config_.input_height;

    for (const auto& det : raw_detections) {
        FaceResult face;
        // Scale coordinates back to original image size
        face.x1 = det.x1 * scale_x;
        face.y1 = det.y1 * scale_y;
        face.x2 = det.x2 * scale_x;
        face.y2 = det.y2 * scale_y;
        face.confidence = det.confidence;

        faces.push_back(face);
    }

    return faces;
}

} // namespace xinfer::zoo::vision