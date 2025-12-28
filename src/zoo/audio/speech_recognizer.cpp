#include <xinfer/zoo/audio/speech_recognizer.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/text/ocr_interface.h>

#include <iostream>
#include <fstream>
#include <algorithm>

namespace xinfer::zoo::audio {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct SpeechRecognizer::Impl {
    SpeechRecognizerConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IAudioPreprocessor> preproc_;
    std::unique_ptr<postproc::IOcrPostprocessor> postproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    Impl(const SpeechRecognizerConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("SpeechRecognizer: Failed to load model.");
        }

        // 2. Setup Audio Preprocessor
        preproc_ = preproc::create_audio_preprocessor(config_.target);

        preproc::AudioPreprocConfig aud_cfg;
        aud_cfg.sample_rate = config_.sample_rate;
        aud_cfg.feature_type = preproc::AudioFeatureType::MEL_SPECTROGRAM;
        aud_cfg.n_fft = config_.n_fft;
        aud_cfg.hop_length = config_.hop_length;
        aud_cfg.n_mels = config_.n_mels;
        aud_cfg.log_mel = true;

        preproc_->init(aud_cfg);

        // 3. Setup Post-processor (CTC Decoder)
        postproc_ = postproc::create_ocr(config_.target);

        postproc::OcrConfig ctc_cfg;
        ctc_cfg.vocabulary = config_.vocabulary;
        ctc_cfg.blank_index = config_.blank_index;

        postproc_->init(ctc_cfg);
    }
};

// =================================================================================
// Public API
// =================================================================================

SpeechRecognizer::SpeechRecognizer(const SpeechRecognizerConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

SpeechRecognizer::~SpeechRecognizer() = default;
SpeechRecognizer::SpeechRecognizer(SpeechRecognizer&&) noexcept = default;
SpeechRecognizer& SpeechRecognizer::operator=(SpeechRecognizer&&) noexcept = default;

std::vector<std::string> SpeechRecognizer::recognize(const std::vector<float>& pcm_data) {
    if (!pimpl_) throw std::runtime_error("SpeechRecognizer is null.");

    // 1. Preprocess (PCM -> Mel Spectrogram)
    preproc::AudioBuffer buf;
    buf.pcm_data = pcm_data.data();
    buf.num_samples = pcm_data.size();
    buf.sample_rate = pimpl_->config_.sample_rate;

    pimpl_->preproc_->process(buf, pimpl_->input_tensor);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess (CTC Decode)
    return pimpl_->postproc_->process(pimpl_->output_tensor);
}

std::vector<std::string> SpeechRecognizer::recognize_batch(const std::vector<std::vector<float>>& pcm_batch) {
    if (!pimpl_ || pcm_batch.empty()) return {};

    size_t batch_size = pcm_batch.size();

    // 1. Batch Preprocessing
    // A proper batch preprocessor would be more efficient. Here we process one by one
    // and stack the resulting tensors. This is slow but demonstrates the logic.
    std::vector<core::Tensor> batch_inputs;
    for (const auto& pcm : pcm_batch) {
        core::Tensor t;
        preproc::AudioBuffer buf{pcm.data(), pcm.size(), pimpl_->config_.sample_rate};
        pimpl_->preproc_->process(buf, t);
        batch_inputs.push_back(t);
    }

    // Stack tensors into a single batch tensor
    // (Requires a Tensor::concat utility)
    // For now, let's assume we can feed a vector of tensors if the backend supports it,
    // or we run one by one.

    // For simplicity, running in a loop:
    std::vector<std::string> all_results;
    for (const auto& in_tensor : batch_inputs) {
        pimpl_->engine_->predict({in_tensor}, {pimpl_->output_tensor});
        auto res = pimpl_->postproc_->process(pimpl_->output_tensor);
        if (!res.empty()) {
            all_results.push_back(res[0]);
        }
    }

    return all_results;
}

} // namespace xinfer::zoo::audio