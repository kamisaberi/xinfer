#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <deque>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::live_events {

    /**
     * @brief A detected highlight event.
     */
    struct HighlightEvent {
        std::string event_type; // "Goal", "Foul", "Corner_Kick"
        float confidence;
        long long timestamp_ms; // When the event occurred

        // The frames that make up the replay clip
        std::vector<cv::Mat> replay_frames;
    };

    struct ReplayConfig {
        // Hardware Target (GPU/NPU required for 3D CNNs)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // Model Path (e.g., resnet3d_sports.engine)
        // A video action classifier
        std::string model_path;
        std::string labels_path; // Action labels ("Goal", "Pass", "Background")

        // Temporal Window
        int window_size = 16;      // Model takes 16 frames as input
        int frame_stride = 2;      // Sample every 2nd frame from live feed

        // Input Specs
        int input_width = 224;
        int input_height = 224;

        // Trigger Settings
        float event_threshold = 0.8f;

        // How many seconds of footage before the event to save?
        float pre_roll_sec = 5.0f;
        // How many seconds after?
        float post_roll_sec = 2.0f;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class ReplayGenerator {
    public:
        explicit ReplayGenerator(const ReplayConfig& config);
        ~ReplayGenerator();

        // Move semantics
        ReplayGenerator(ReplayGenerator&&) noexcept;
        ReplayGenerator& operator=(ReplayGenerator&&) noexcept;
        ReplayGenerator(const ReplayGenerator&) = delete;
        ReplayGenerator& operator=(const ReplayGenerator&) = delete;

        /**
         * @brief Process a live frame and check for highlight events.
         *
         * This function must be called for every frame of the stream.
         *
         * @param image The current video frame.
         * @param timestamp_ms Current stream time.
         * @return A HighlightEvent if a new one was just triggered, otherwise returns an empty/invalid event.
         */
        std::vector<HighlightEvent> process_frame(const cv::Mat& image, long long timestamp_ms);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::live_events```

---

### 2. Implementation: `src/zoo/live_events/replay_generator.cpp`

This implementation manages two **Ring Buffers (Deques)**:
1.  **Inference Buffer:** Holds the sparse, resized frames for the AI model (`window_size` frames).
2.  **Replay Buffer:** Holds the original, high-resolution frames for the clip (`pre_roll + post_roll` seconds worth).

```cpp
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