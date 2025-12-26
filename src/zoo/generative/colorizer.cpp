#include <xinfer/zoo/generative/colorizer.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
// Preproc/Postproc is custom Lab color space math, not using factories.

#include <iostream>
#include <vector>

namespace xinfer::zoo::generative {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct Colorizer::Impl {
    ColorizerConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;

    // Data Containers
    core::Tensor input_tensor;  // L channel
    core::Tensor output_tensor; // a, b channels

    // Cached original L channel for merging
    cv::Mat original_l_channel_;

    Impl(const ColorizerConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("Colorizer: Failed to load model " + config_.model_path);
        }

        // 2. Allocate Tensors
        input_tensor.resize({1, 1, (int64_t)config_.input_height, (int64_t)config_.input_width}, core::DataType::kFLOAT);
    }

    // --- Custom Preprocessing: BGR -> Lab -> L Channel Tensor ---
    void preprocess(const cv::Mat& img) {
        // 1. Ensure input is 3-channel
        cv::Mat bgr_img;
        if (img.channels() == 1) {
            cv::cvtColor(img, bgr_img, cv::COLOR_GRAY2BGR);
        } else {
            bgr_img = img;
        }

        // 2. Resize to model input size
        cv::Mat resized;
        cv::resize(bgr_img, resized, cv::Size(config_.input_width, config_.input_height));

        // 3. Convert to Lab color space
        cv::Mat lab_img;
        resized.convertTo(lab_img, CV_32F, 1.0/255.0); // Convert to float [0,1] first
        cv::cvtColor(lab_img, lab_img, cv::COLOR_BGR2Lab);

        // 4. Split Channels
        std::vector<cv::Mat> lab_planes(3);
        cv::split(lab_img, lab_planes);

        // 5. Prepare L Channel for model
        // L channel is [0, 100], normalize to [-1, 1] as expected by most GANs
        // (L' = L / 50.0 - 1.0)
        cv::Mat l_norm = (lab_planes[0] / 50.0f) - 1.0f;

        // Copy to input tensor
        float* ptr = static_cast<float*>(input_tensor.data());
        std::memcpy(ptr, l_norm.data, l_norm.total() * sizeof(float));

        // Cache original L channel for merging later
        // It needs to be resized back to original image size
        cv::resize(lab_planes[0], original_l_channel_, img.size());
    }

    // --- Custom Post-processing: Merge L + (a,b) -> Lab -> BGR ---
    cv::Mat postprocess(const cv::Size& original_size) {
        // 1. Get predicted 'ab' channels
        // Output shape is [1, 2, H, W]
        auto shape = output_tensor.shape();
        int h = (int)shape[2];
        int w = (int)shape[3];
        int plane_size = h * w;

        const float* data = static_cast<const float*>(output_tensor.data());

        // Model usually outputs 'ab' in range [-1, 1], scale back to Lab's range [-128, 127]
        float scale = 128.0f;

        cv::Mat a_channel(h, w, CV_32F);
        cv::Mat b_channel(h, w, CV_32F);

        const float* a_ptr = data;
        const float* b_ptr = data + plane_size;

        for(int i=0; i<plane_size; ++i) {
            a_channel.at<float>(i) = a_ptr[i] * scale;
            b_channel.at<float>(i) = b_ptr[i] * scale;
        }

        // 2. Resize predicted 'ab' to original image size
        cv::Mat a_resized, b_resized;
        cv::resize(a_channel, a_resized, original_size, 0, 0, cv::INTER_CUBIC);
        cv::resize(b_channel, b_resized, original_size, 0, 0, cv::INTER_CUBIC);

        // 3. Merge with original L channel
        std::vector<cv::Mat> final_lab_planes = {original_l_channel_, a_resized, b_resized};

        cv::Mat final_lab;
        cv::merge(final_lab_planes, final_lab);

        // 4. Convert back to BGR
        cv::Mat final_bgr;
        cv::cvtColor(final_lab, final_bgr, cv::COLOR_Lab2BGR);

        // Convert to uint8
        cv::Mat final_image;
        final_bgr.convertTo(final_image, CV_8U, 255.0);

        return final_image;
    }
};

// =================================================================================
// Public API
// =================================================================================

Colorizer::Colorizer(const ColorizerConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

Colorizer::~Colorizer() = default;
Colorizer::Colorizer(Colorizer&&) noexcept = default;
Colorizer& Colorizer::operator=(Colorizer&&) noexcept = default;

cv::Mat Colorizer::colorize(const cv::Mat& gray_image) {
    if (!pimpl_) throw std::runtime_error("Colorizer is null.");

    // 1. Preprocess
    pimpl_->preprocess(gray_image);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess
    return pimpl_->postprocess(gray_image.size());
}

} // namespace xinfer::zoo::generative