#pragma once


#include <string>
#include <vector>
#include <memory>
#include <include/preproc/audio_processor.h>

namespace xinfer::zoo::audio {

    struct AudioClassificationResult {
        int class_id;
        float confidence;
        std::string label;
    };

    struct ClassifierConfig {
        std::string engine_path;
        std::string labels_path = "";
        preproc::AudioProcessorConfig audio_config;
    };

    class Classifier {
    public:
        explicit Classifier(const ClassifierConfig& config);
        ~Classifier();

        Classifier(const Classifier&) = delete;
        Classifier& operator=(const Classifier&) = delete;
        Classifier(Classifier&&) noexcept;
        Classifier& operator=(Classifier&&) noexcept;

        std::vector<AudioClassificationResult> predict(const std::vector<float>& waveform, int top_k = 5);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::audio

