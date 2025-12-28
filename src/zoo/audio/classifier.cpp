#include <xinfer/zoo/audio/classifier.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
// Reusing vision's classification postproc, as the math is the same
#include <xinfer/postproc/vision/classification_interface.h>

#include <iostream>
#include <fstream>
#include <algorithm>

namespace xinfer::zoo::audio {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct Classifier::Impl {
    AudioClassifierConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IAudioPreprocessor> preproc_;
    std::unique_ptr<postproc::IClassificationPostprocessor> postproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    std::vector<std::string> labels_;

    Impl(const AudioClassifierConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("AudioClassifier: Failed to load model.");
        }

        // 2. Setup Audio Preprocessor
        preproc_ = preproc::create_audio_preprocessor(config_.target);

        preproc::AudioPreprocConfig aud_cfg;
        aud_cfg.sample_rate = config_.sample_rate;
        aud_cfg.feature_type = preproc::AudioFeatureType::MEL_SPECTROGRAM;
        aud_cfg.n_fft = config_.n_fft;
        aud_cfg.hop_length = config_.hop_length;
        aud_cfg.n_mels = config_.n_mels;
        aud_cfg.log_mel = true; // CNNs work best with log-mel

        preproc_->init(aud_cfg);

        // 3. Setup Post-processor
        postproc_ = postproc::create_classification(config_.target);

        postproc::ClassificationConfig cls_cfg;
        cls_cfg.top_k = config_.top_k;
        cls_cfg.apply_softmax = true;

        if (!config_.labels_path.empty()) {
            load_labels(config_.labels_path);
            cls_cfg.labels = labels_;
        }

        postproc_->init(cls_cfg);
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

    // Helper: Pad or crop audio to the fixed duration the model expects
    std::vector<float> normalize_length(const std::vector<float>& pcm) {
        size_t target_len = config_.sample_rate * config_.duration_sec;

        if (pcm.size() == target_len) return pcm;

        std::vector<float> fixed(target_len, 0.0f); // Pad with silence

        if (pcm.size() > target_len) {
            // Trim (take center)
            size_t start = (pcm.size() - target_len) / 2;
            std::copy(pcm.begin() + start, pcm.begin() + start + target_len, fixed.begin());
        } else {
            // Pad
            std::copy(pcm.begin(), pcm.end(), fixed.begin());
        }
        return fixed;
    }
};

// =================================================================================
// Public API
// =================================================================================

Classifier::Classifier(const AudioClassifierConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

Classifier::~Classifier() = default;
Classifier::Classifier(Classifier&&) noexcept = default;
Classifier& Classifier::operator=(Classifier&&) noexcept = default;

std::vector<AudioClassResult> Classifier::classify(const std::vector<float>& pcm_data) {
    if (!pimpl_) throw std::runtime_error("AudioClassifier is null.");

    // 1. Fix Audio Length
    std::vector<float> fixed_audio = pimpl_->normalize_length(pcm_data);

    // 2. Preprocess (PCM -> Mel Spectrogram)
    preproc::AudioBuffer buf;
    buf.pcm_data = fixed_audio.data();
    buf.num_samples = fixed_audio.size();
    buf.sample_rate = pimpl_->config_.sample_rate;

    pimpl_->preproc_->process(buf, pimpl_->input_tensor);

    // 3. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 4. Postprocess
    auto raw_results = pimpl_->postproc_->process(pimpl_->output_tensor);

    std::vector<AudioClassResult> results;
    if (!raw_results.empty() && !raw_results[0].empty()) {
        const auto& batch_res = raw_results[0]; // Batch size 1
        results.reserve(batch_res.size());

        for (const auto& item : batch_res) {
            if (item.score >= pimpl_->config_.confidence_threshold) {
                AudioClassResult res;
                res.id = item.id;
                res.confidence = item.score;
                res.label = item.label;
                results.push_back(res);
            }
        }
    }

    return results;
}

} // namespace xinfer::zoo::audio