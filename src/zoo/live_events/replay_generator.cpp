#include <xinfer/zoo/live_events/replay_generator.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/classification_interface.h>

#include <iostream>
#include <fstream>
#include <numeric>

namespace xinfer::zoo::live_events {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct ReplayGenerator::Impl {
    ReplayConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    // We reuse the standard image preprocessor for each frame
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::IClassificationPostprocessor> postproc_;

    // Data Containers
    core::Tensor input_tensor; // 5D: [1, T, C, H, W]
    core::Tensor output_tensor;

    // --- State Buffers ---
    // Stores original, high-res frames for saving the clip
    std::deque<cv::Mat> replay_buffer_;
    // Stores downsampled, normalized frames for the AI model
    std::deque<core::Tensor> inference_buffer_;

    // Frame counter for striding
    int frame_count_ = 0;
    // Labels
    std::vector<std::string> labels_;
    // Avoid re-triggering for the same event
    bool event_on_cooldown_ = false;
    int cooldown_frames_ = 0;

    Impl(const ReplayConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);
        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("ReplayGenerator: Failed to load model.");
        }

        // 2. Setup Preprocessor (for single frames)
        preproc_ = preproc::create_image_preprocessor(config_.target);
        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        preproc_->init(pre_cfg);

        // 3. Setup Post-processor
        postproc_ = postproc::create_classification(config_.target);
        postproc::ClassificationConfig post_cfg;
        post_cfg.top_k = 3;
        load_labels(config_.labels_path);
        post_cfg.labels = labels_;
        postproc_->init(post_cfg);
    }

    void load_labels(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) return;
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            labels_.push_back(line);
        }
    }

    void update_buffers(const cv::Mat& image) {
        // Update Replay Buffer (high-res)
        replay_buffer_.push_back(image.clone());
        // Assuming ~30fps, calc buffer size from seconds
        size_t replay_capacity = 30 * (config_.pre_roll_sec + config_.post_roll_sec);
        if (replay_buffer_.size() > replay_capacity) {
            replay_buffer_.pop_front();
        }

        // Update Inference Buffer (low-res, strided)
        frame_count_++;
        if (frame_count_ % config_.frame_stride == 0) {
            // Preprocess and add to buffer
            core::Tensor processed_frame;
            preproc::ImageFrame frame{image.data, image.cols, image.rows, preproc::ImageFormat::BGR};
            preproc_->process(frame, processed_frame);

            inference_buffer_.push_back(processed_frame);
            if (inference_buffer_.size() > (size_t)config_.window_size) {
                inference_buffer_.pop_front();
            }
        }
    }

    void prepare_input_tensor() {
        // Stack frames from inference_buffer into the 5D input tensor
        int T = config_.window_size;
        int C = 3;
        int H = config_.input_height;
        int W = config_.input_width;

        input_tensor.resize({1, (int64_t)T, (int64_t)C, (int64_t)H, (int64_t)W}, core::DataType::kFLOAT);

        char* dst_ptr = static_cast<char*>(input_tensor.data());
        size_t frame_bytes = C * H * W * sizeof(float);

        for(int t=0; t < T; ++t) {
            std::memcpy(dst_ptr + (t * frame_bytes), inference_buffer_[t].data(), frame_bytes);
        }
    }
};

// =================================================================================
// Public API
// =================================================================================

ReplayGenerator::ReplayGenerator(const ReplayConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

ReplayGenerator::~ReplayGenerator() = default;
ReplayGenerator::ReplayGenerator(ReplayGenerator&&) noexcept = default;
ReplayGenerator& ReplayGenerator::operator=(ReplayGenerator&&) noexcept = default;

std::vector<HighlightEvent> ReplayGenerator::process_frame(const cv::Mat& image, long long timestamp_ms) {
    if (!pimpl_) throw std::runtime_error("ReplayGenerator is null.");

    std::vector<HighlightEvent> events;

    // 1. Update Buffers
    pimpl_->update_buffers(image);

    // Manage cooldown
    if (pimpl_->event_on_cooldown_) {
        pimpl_->cooldown_frames_--;
        if (pimpl_->cooldown_frames_ <= 0) pimpl_->event_on_cooldown_ = false;
        return events;
    }

    // Check if we have enough frames to run inference
    if (pimpl_->inference_buffer_.size() < (size_t)pimpl_->config_.window_size) {
        return events;
    }

    // 2. Prepare Input
    pimpl_->prepare_input_tensor();

    // 3. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 4. Postprocess
    auto results = pimpl_->postproc_->process(pimpl_->output_tensor);

    if (results.empty() || results[0].empty()) return events;

    // 5. Logic
    for (const auto& res : results[0]) {
        // Ignore background class (usually "Normal")
        if (res.label == "Normal" || res.label == "Background") continue;

        if (res.score > pimpl_->config_.event_threshold) {
            HighlightEvent evt;
            evt.event_type = res.label;
            evt.confidence = res.score;
            evt.timestamp_ms = timestamp_ms;

            // Copy frames from buffer
            for (const auto& f : pimpl_->replay_buffer_) {
                evt.replay_frames.push_back(f.clone());
            }

            events.push_back(evt);

            // Set cooldown to prevent spamming triggers for the same goal
            pimpl_->event_on_cooldown_ = true;
            pimpl_->cooldown_frames_ = 30 * 5; // 5 seconds at 30fps
            break; // Only trigger one event per inference
        }
    }

    return events;
}

} // namespace xinfer::zoo::live_events