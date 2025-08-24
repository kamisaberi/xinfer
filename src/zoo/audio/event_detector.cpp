#include <include/zoo/audio/event_detector.h>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <cmath>

#include <include/core/engine.h>

namespace xinfer::zoo::audio {

struct EventDetector::Impl {
    EventDetectorConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::AudioProcessor> preprocessor_;
    std::vector<std::string> class_labels_;
};

EventDetector::EventDetector(const EventDetectorConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Audio event detector engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    pimpl_->preprocessor_ = std::make_unique<preproc::AudioProcessor>(pimpl_->config_.audio_config);

    if (!pimpl_->config_.labels_path.empty()) {
        std::ifstream labels_file(pimpl_->config_.labels_path);
        if (!labels_file) throw std::runtime_error("Could not open labels file: " + pimpl_->config_.labels_path);
        std::string line;
        while (std::getline(labels_file, line)) {
            pimpl_->class_labels_.push_back(line);
        }
    }
}

EventDetector::~EventDetector() = default;
EventDetector::EventDetector(EventDetector&&) noexcept = default;
EventDetector& EventDetector::operator=(EventDetector&&) noexcept = default;

std::vector<AudioEvent> EventDetector::predict(const std::vector<float>& waveform) {
    if (!pimpl_) throw std::runtime_error("EventDetector is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor spectrogram_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(waveform, spectrogram_tensor);

    auto output_tensors = pimpl_->engine_->infer({spectrogram_tensor});
    const core::Tensor& probability_tensor = output_tensors[0];

    auto prob_shape = probability_tensor.shape();
    const int num_frames = prob_shape[1];
    const int num_classes = prob_shape[2];

    std::vector<float> probabilities(probability_tensor.num_elements());
    probability_tensor.copy_to_host(probabilities.data());

    std::vector<AudioEvent> events;
    float frame_duration_seconds = (float)pimpl_->config_.audio_config.hop_length / pimpl_->config_.audio_config.sample_rate;

    for (int c = 0; c < num_classes; ++c) {
        bool in_event = false;
        float start_time = 0.0f;
        for (int t = 0; t < num_frames; ++t) {
            float prob = probabilities[t * num_classes + c];
            if (prob > pimpl_->config_.event_threshold && !in_event) {
                in_event = true;
                start_time = (float)t * frame_duration_seconds;
            } else if (prob < pimpl_->config_.event_threshold && in_event) {
                in_event = false;
                AudioEvent event;
                event.class_id = c;
                event.start_time_seconds = start_time;
                event.end_time_seconds = (float)t * frame_duration_seconds;
                event.confidence = 1.0f;
                if (c < pimpl_->class_labels_.size()) {
                    event.label = pimpl_->class_labels_[c];
                }
                events.push_back(event);
            }
        }
        if (in_event) {
            AudioEvent event;
            event.class_id = c;
            event.start_time_seconds = start_time;
            event.end_time_seconds = (float)num_frames * frame_duration_seconds;
            event.confidence = 1.0f;
            if (c < pimpl_->class_labels_.size()) {
                event.label = pimpl_->class_labels_[c];
            }
            events.push_back(event);
        }
    }

    return events;
}

} // namespace xinfer::zoo::audio