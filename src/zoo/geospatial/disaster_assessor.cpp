#include <xinfer/zoo/geospatial/disaster_assessor.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/segmentation_interface.h>

#include <iostream>
#include <numeric>
#include <map>

namespace xinfer::zoo::geospatial {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct DisasterAssessor::Impl {
    DisasterConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    // We reuse the standard image preprocessor for each image
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;

    // Custom post-processing logic is needed for damage quantification
    // But we can reuse Segmentation for the raw mask
    std::unique_ptr<postproc::ISegmentationPostprocessor> postproc_;

    // Data Containers
    core::Tensor input_tensor; // 6-Channel
    core::Tensor output_tensor;

    Impl(const DisasterConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("DisasterAssessor: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor (for single images)
        preproc_ = preproc::create_image_preprocessor(config_.target);
        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::RGB;
        pre_cfg.layout_nchw = true;
        preproc_->init(pre_cfg);

        // 3. Setup Post-processor
        postproc_ = postproc::create_segmentation(config_.target);
        postproc::SegmentationConfig post_cfg;
        post_cfg.target_width = config_.input_width;
        post_cfg.target_height = config_.input_height;
        postproc_->init(post_cfg);
    }

    // --- Core Logic ---
    void prepare_input(const cv::Mat& pre, const cv::Mat& post) {
        // Allocate 6-channel tensor
        input_tensor.resize({1, 6, (int64_t)config_.input_height, (int64_t)config_.input_width}, core::DataType::kFLOAT);

        // Temporary tensors for individual preprocessing
        core::Tensor pre_tensor, post_tensor;

        // Preprocess "Before" image
        preproc::ImageFrame pre_frame{pre.data, pre.cols, pre.rows, preproc::ImageFormat::BGR};
        preproc_->process(pre_frame, pre_tensor);

        // Preprocess "After" image
        preproc::ImageFrame post_frame{post.data, post.cols, post.rows, preproc::ImageFormat::BGR};
        preproc_->process(post_frame, post_tensor);

        // Concatenate
        char* dst_ptr = static_cast<char*>(input_tensor.data());
        size_t single_size_bytes = 3 * config_.input_height * config_.input_width * sizeof(float);

        std::memcpy(dst_ptr, pre_tensor.data(), single_size_bytes);
        std::memcpy(dst_ptr + single_size_bytes, post_tensor.data(), single_size_bytes);
    }

    void analyze_mask(const cv::Mat& mask, AssessmentResult& result) {
        // Find contours for each damage level
        for (int level = 1; level <= 3; ++level) { // Skip "No Damage"
            cv::Mat level_mask;
            cv::inRange(mask, cv::Scalar(level), cv::Scalar(level), level_mask);

            if (cv::countNonZero(level_mask) == 0) continue;

            // Morphology to clean up noise
            cv::morphologyEx(level_mask, level_mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5)));

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(level_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            for (const auto& cnt : contours) {
                if (cv::contourArea(cnt) > 50) { // Min area filter
                    DamageSite site;
                    site.box.x1 = cv::boundingRect(cnt).x;
                    site.box.y1 = cv::boundingRect(cnt).y;
                    // ... etc. for box
                    site.level = static_cast<DamageLevel>(level);
                    site.confidence = 0.9f; // Placeholder
                    result.damaged_sites.push_back(site);
                }
            }
        }

        result.total_structures_affected = result.damaged_sites.size();

        // Calculate total damage area
        cv::Mat all_damage_mask;
        cv::inRange(mask, cv::Scalar(1), cv::Scalar(3), all_damage_mask);
        result.damage_area_percent = (float)cv::countNonZero(all_damage_mask) / (mask.rows * mask.cols);
    }
};

// =================================================================================
// Public API
// =================================================================================

DisasterAssessor::DisasterAssessor(const DisasterConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

DisasterAssessor::~DisasterAssessor() = default;
DisasterAssessor::DisasterAssessor(DisasterAssessor&&) noexcept = default;
DisasterAssessor& DisasterAssessor::operator=(DisasterAssessor&&) noexcept = default;

AssessmentResult DisasterAssessor::assess(const cv::Mat& pre_disaster_img, const cv::Mat& post_disaster_img) {
    if (!pimpl_) throw std::runtime_error("DisasterAssessor is null.");

    // 1. Prepare Input
    pimpl_->prepare_input(pre_disaster_img, post_disaster_img);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess
    auto seg_res = pimpl_->postproc_->process(pimpl_->output_tensor);

    // Convert to Mat
    int h = seg_res.mask.shape()[1];
    int w = seg_res.mask.shape()[2];
    const uint8_t* ptr = static_cast<const uint8_t*>(seg_res.mask.data());
    cv::Mat mask_low(h, w, CV_8U, const_cast<uint8_t*>(ptr));

    // 4. Analysis
    AssessmentResult result;
    pimpl_->analyze_mask(mask_low, result);

    // 5. Visualization (overlay on "After" image)
    result.annotated_image = post_disaster_img.clone();
    // ... (Drawing logic similar to other segmentation examples) ...

    return result;
}

} // namespace xinfer::zoo::geospatial