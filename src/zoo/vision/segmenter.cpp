#include <xinfer/zoo/vision/segmenter.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>

#include <iostream>
#include <algorithm>
#include <chrono>
#include <random>

namespace xinfer::zoo::vision {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct Segmenter::Impl {
    SegmenterConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::ISegmentationPostprocessor> postproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    // LUT for visualization
    std::vector<cv::Vec3b> color_lut_;

    Impl(const SegmenterConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("Segmenter: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::RGB;
        pre_cfg.layout_nchw = true;

        pre_cfg.norm_params.mean = config_.mean;
        pre_cfg.norm_params.std = config_.std;
        pre_cfg.norm_params.scale_factor = 1.0f; // Input 0-255

        preproc_->init(pre_cfg);

        // 3. Setup Post-processor
        postproc_ = postproc::create_segmentation(config_.target);

        postproc::SegmentationConfig post_cfg;
        // We initially process at model resolution for speed
        post_cfg.target_width = config_.input_width;
        post_cfg.target_height = config_.input_height;
        post_cfg.apply_softmax = false; // ArgMax directly on logits is faster/equivalent

        postproc_->init(post_cfg);

        // 4. Initialize Colors
        init_colors();
    }

    void init_colors() {
        if (!config_.class_colors.empty()) {
            for (const auto& c : config_.class_colors) {
                if (c.size() >= 3) color_lut_.push_back(cv::Vec3b(c[2], c[1], c[0])); // RGB->BGR
            }
        } else {
            // Generate Random Colors
            std::mt19937 rng(42);
            std::uniform_int_distribution<int> dist(0, 255);
            for (int i = 0; i < 256; ++i) { // Support up to 256 classes default
                color_lut_.push_back(cv::Vec3b(dist(rng), dist(rng), dist(rng)));
            }
            // Class 0 (Background) usually black
            color_lut_[0] = cv::Vec3b(0, 0, 0);
        }
    }

    cv::Mat colorize(const cv::Mat& mask_idx) {
        cv::Mat color_mask(mask_idx.size(), CV_8UC3);

        // Map indices to colors
        // Fast iterator access
        int rows = mask_idx.rows;
        int cols = mask_idx.cols;

        if (mask_idx.isContinuous() && color_mask.isContinuous()) {
            cols *= rows;
            rows = 1;
        }

        const uint8_t* p_idx = mask_idx.ptr<uint8_t>(0);
        cv::Vec3b* p_dst = color_mask.ptr<cv::Vec3b>(0);
        size_t lut_size = color_lut_.size();

        for (int i = 0; i < cols * rows; ++i) {
            uint8_t id = p_idx[i];
            if (id < lut_size) {
                p_dst[i] = color_lut_[id];
            } else {
                p_dst[i] = cv::Vec3b(255, 255, 255); // Unknown class white
            }
        }
        return color_mask;
    }
};

// =================================================================================
// Public API
// =================================================================================

Segmenter::Segmenter(const SegmenterConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

Segmenter::~Segmenter() = default;
Segmenter::Segmenter(Segmenter&&) noexcept = default;
Segmenter& Segmenter::operator=(Segmenter&&) noexcept = default;

SegmenterResult Segmenter::segment(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("Segmenter is null.");

    auto start = std::chrono::high_resolution_clock::now();

    // 1. Preprocess
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    frame.format = preproc::ImageFormat::BGR;

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess (ArgMax)
    // Result mask is at model resolution (e.g. 512x512)
    auto raw_res = pimpl_->postproc_->process(pimpl_->output_tensor);

    // Convert Tensor -> cv::Mat
    auto shape = raw_res.mask.shape();
    int h = (int)shape[1];
    int w = (int)shape[2];

    // Wrap mask data (UINT8)
    const uint8_t* ptr = static_cast<const uint8_t*>(raw_res.mask.data());
    cv::Mat low_res_mask(h, w, CV_8U, const_cast<uint8_t*>(ptr));

    // 4. Resize to Original
    SegmenterResult result;

    // Nearest neighbor to preserve class IDs
    if (h != image.rows || w != image.cols) {
        cv::resize(low_res_mask, result.mask, image.size(), 0, 0, cv::INTER_NEAREST);
    } else {
        result.mask = low_res_mask.clone();
    }

    // 5. Visualization
    result.color_mask = pimpl_->colorize(result.mask);

    auto end = std::chrono::high_resolution_clock::now();
    result.inference_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    return result;
}

cv::Mat Segmenter::blend(const cv::Mat& image, const cv::Mat& mask, float alpha) {
    cv::Mat out;
    cv::addWeighted(image, 1.0 - alpha, mask, alpha, 0.0, out);
    return out;
}

} // namespace xinfer::zoo::vision