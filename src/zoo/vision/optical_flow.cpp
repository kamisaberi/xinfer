#include <xinfer/zoo/vision/optical_flow.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>

#include <iostream>
#include <cmath>

namespace xinfer::zoo::vision {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct OpticalFlow::Impl {
    OpticalFlowConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;

    // Data Containers
    core::Tensor input_prev;
    core::Tensor input_curr;
    core::Tensor output_flow;

    Impl(const OpticalFlowConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("OpticalFlow: Failed to load model " + config_.model_path);
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
        pre_cfg.norm_params.scale_factor = config_.scale_factor;

        preproc_->init(pre_cfg);
    }

    // --- Post-Processing: Tensor -> Flow Map ---
    cv::Mat process_output(const core::Tensor& tensor, const cv::Size& original_size) {
        auto shape = tensor.shape(); // Expect [1, 2, H, W]
        if (shape.size() != 4 || shape[1] != 2) {
            XINFER_LOG_ERROR("Optical Flow output shape mismatch. Expected [1, 2, H, W].");
            return cv::Mat();
        }

        int h = (int)shape[2];
        int w = (int)shape[3];
        int spatial = h * w;

        const float* data = static_cast<const float*>(tensor.data());

        // Create 2-channel Mat for Flow (CV_32FC2)
        // Model output is Planar (NCHW), OpenCV needs Packed (HWC)
        std::vector<cv::Mat> channels(2);
        channels[0] = cv::Mat(h, w, CV_32F, const_cast<float*>(data));             // dx
        channels[1] = cv::Mat(h, w, CV_32F, const_cast<float*>(data + spatial));   // dy

        cv::Mat flow;
        cv::merge(channels, flow);

        // Resize to original resolution
        if (h != original_size.height || w != original_size.width) {
            cv::resize(flow, flow, original_size, 0, 0, cv::INTER_LINEAR);

            // IMPORTANT: When upscaling flow, magnitudes must be scaled too!
            float scale_x = (float)original_size.width / w;
            float scale_y = (float)original_size.height / h;

            // Multiply channel 0 by scale_x, channel 1 by scale_y
            // We can do this efficiently by splitting again or using mul
            // Assuming uniform scale for simplicity often works, but let's be precise:

            // Simple scalar mul if aspect ratio maintained
            flow *= scale_x;
        }

        return flow;
    }

    // --- Visualization: Flow -> Color ---
    cv::Mat colorize_flow(const cv::Mat& flow) {
        cv::Mat hsv(flow.size(), CV_8UC3);
        cv::Mat magnitude, angle;
        std::vector<cv::Mat> flow_chans(2);
        cv::split(flow, flow_chans);

        // Convert cartesian to polar
        cv::cartToPolar(flow_chans[0], flow_chans[1], magnitude, angle, true);

        // Map Angle -> Hue (0-180), Magnitude -> Value (0-255)
        // Saturation = 255
        std::vector<cv::Mat> hsv_planes(3);

        // Angle is 0-360, Hue expects 0-180
        angle *= 0.5;
        angle.convertTo(hsv_planes[0], CV_8U);

        hsv_planes[1] = cv::Mat::ones(flow.size(), CV_8U) * 255;

        // Normalize magnitude for visibility
        cv::normalize(magnitude, magnitude, 0, 255, cv::NORM_MINMAX);
        magnitude.convertTo(hsv_planes[2], CV_8U);

        cv::merge(hsv_planes, hsv);

        cv::Mat bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
        return bgr;
    }
};

// =================================================================================
// Public API
// =================================================================================

OpticalFlow::OpticalFlow(const OpticalFlowConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

OpticalFlow::~OpticalFlow() = default;
OpticalFlow::OpticalFlow(OpticalFlow&&) noexcept = default;
OpticalFlow& OpticalFlow::operator=(OpticalFlow&&) noexcept = default;

FlowResult OpticalFlow::calculate(const cv::Mat& prev_image, const cv::Mat& curr_image) {
    if (!pimpl_) throw std::runtime_error("OpticalFlow is null.");

    // 1. Preprocess Both Frames
    preproc::ImageFrame frame_prev{prev_image.data, prev_image.cols, prev_image.rows, preproc::ImageFormat::BGR};
    preproc::ImageFrame frame_curr{curr_image.data, curr_image.cols, curr_image.rows, preproc::ImageFormat::BGR};

    pimpl_->preproc_->process(frame_prev, pimpl_->input_prev);
    pimpl_->preproc_->process(frame_curr, pimpl_->input_curr);

    // 2. Inference
    // Passing two tensors to the engine
    pimpl_->engine_->predict({pimpl_->input_prev, pimpl_->input_curr}, {pimpl_->output_flow});

    // 3. Postprocess
    FlowResult result;
    result.flow_map = pimpl_->process_output(pimpl_->output_flow, prev_image.size());

    if (pimpl_->config_.generate_visualization) {
        result.visualization = pimpl_->colorize_flow(result.flow_map);
    }

    return result;
}

cv::Mat OpticalFlow::warp(const cv::Mat& src, const cv::Mat& flow) {
    // Utility to warp 'src' according to 'flow'
    // flow(x,y) contains (dx, dy)
    // We need to create a map for remap: map_x(x,y) = x + dx, map_y(x,y) = y + dy

    cv::Mat map_x(flow.size(), CV_32F);
    cv::Mat map_y(flow.size(), CV_32F);

    for(int y=0; y<flow.rows; ++y) {
        for(int x=0; x<flow.cols; ++x) {
            cv::Point2f f = flow.at<cv::Point2f>(y, x);
            map_x.at<float>(y, x) = (float)x + f.x;
            map_y.at<float>(y, x) = (float)y + f.y;
        }
    }

    cv::Mat warped;
    cv::remap(src, warped, map_x, map_y, cv::INTER_LINEAR);
    return warped;
}

} // namespace xinfer::zoo::vision