#include <xinfer/zoo/audio/event_detector.h>
#include <xinfer/core/logging.h>

// --- We reuse the Audio Classifier module internally ---
#include <xinfer/zoo/audio/classifier.h>

#include <iostream>
#include <vector>
#include <deque>
#include <numeric>

namespace xinfer::zoo::audio {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct EventDetector::Impl {
    EventDetectorConfig config_;

    // We reuse the Classifier as our core engine
    std::unique_ptr<Classifier> classifier_;

    // --- Streaming Buffers ---
    // A ring buffer to hold the continuous audio stream
    std::deque<float> stream_buffer_;

    // Size of buffers in samples
    size_t window_samples_;
    size_t stride_samples_;

    long long current_timestamp_ = 0;

    Impl(const EventDetectorConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // Configure the underlying Classifier
        ClassifierConfig cls_cfg;
        cls_cfg.target = config_.target;
        cls_cfg.model_path = config_.model_path;
        cls_cfg.labels_path = config_.labels_path;
        cls_cfg.sample_rate = config_.sample_rate;
        cls_cfg.duration_sec = config_.window_sec;
        cls_cfg.n_fft = config_.n_fft;
        cls_cfg.hop_length = config_.hop_length;
        cls_cfg.n_mels = config_.n_mels;
        cls_cfg.top_k = 5; // Get a few top candidates
        cls_cfg.confidence_threshold = config_.confidence_threshold;

        classifier_ = std::make_unique<Classifier>(cls_cfg);

        // Calculate buffer sizes in samples
        window_samples_ = config_.sample_rate * config_.window_sec;
        stride_samples_ = config_.sample_rate * config_.stride_sec;
    }

    // Simple energy-based Voice Activity Detection (VAD)
    bool is_active(const std::vector<float>& chunk) {
        if (chunk.empty()) return false;

        double sum_sq = 0.0;
        for (float s : chunk) sum_sq += s * s;

        double rms = std::sqrt(sum_sq / chunk.size());
        return (rms > config_.vad_energy_threshold);
    }
};

// =================================================================================
// Public API
// =================================================================================

EventDetector::EventDetector(const EventDetectorConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

EventDetector::~EventDetector() = default;
EventDetector::EventDetector(EventDetector&&) noexcept = default;
EventDetector& EventDetector::operator=(EventDetector&&) noexcept = default;

void EventDetector::reset() {
    if (pimpl_) pimpl_->stream_buffer_.clear();
}

std::vector<AudioEvent> EventDetector::process_stream(const std::vector<float>& pcm_chunk, long long timestamp_ms) {
    if (!pimpl_ || !pimpl_->classifier_) throw std::runtime_error("EventDetector is null.");

    // 1. Add new data to the buffer
    pimpl_->stream_buffer_.insert(pimpl_->stream_buffer_.end(), pcm_chunk.begin(), pcm_chunk.end());

    if (pimpl_->current_timestamp_ == 0) pimpl_->current_timestamp_ = timestamp_ms;

    std::vector<AudioEvent> detected_events;

    // 2. Process in sliding windows
    while (pimpl_->stream_buffer_.size() >= pimpl_->window_samples_) {
        // A. Extract a window
        std::vector<float> window(pimpl_->window_samples_);
        std::copy(pimpl_->stream_buffer_.begin(),
                  pimpl_->stream_buffer_.begin() + pimpl_->window_samples_,
                  window.begin());

        // B. Run VAD to save power
        if (pimpl_->is_active(window)) {
            // C. Classify the window
            auto results = pimpl_->classifier_->classify(window);

            // D. Log Events
            for (const auto& res : results) {
                // Ignore "background" or "silence" classes
                if (res.label.find("Speech") != std::string::npos ||
                    res.label.find("Silence") != std::string::npos) {
                    continue;
                }

                AudioEvent evt;
                evt.label = res.label;
                evt.confidence = res.confidence;
                evt.start_time_ms = pimpl_->current_timestamp_;
                evt.end_time_ms = pimpl_->current_timestamp_ + (long long)(pimpl_->config_.window_sec * 1000);

                detected_events.push_back(evt);
            }
        }

        // E. Slide the window
        pimpl_->stream_buffer_.erase(pimpl_->stream_buffer_.begin(),
                                     pimpl_->stream_buffer_.begin() + pimpl_->stride_samples_);

        pimpl_->current_timestamp_ += (long long)(pimpl_->config_.stride_sec * 1000);
    }

    return detected_events;
}

} // namespace xinfer::zoo::audio