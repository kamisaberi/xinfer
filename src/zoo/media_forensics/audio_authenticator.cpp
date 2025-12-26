#include <xinfer/zoo/media_forensics/audio_authenticator.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/classification_interface.h>

#include <iostream>
#include <algorithm>
#include <cmath>

namespace xinfer::zoo::media_forensics {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct AudioAuthenticator::Impl {
    AudioAuthConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IAudioPreprocessor> preproc_;
    std::unique_ptr<postproc::IClassificationPostprocessor> postproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    Impl(const AudioAuthConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("AudioAuthenticator: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor (Audio DSP)
        if (config_.use_spectrogram) {
            preproc_ = preproc::create_audio_preprocessor(config_.target);

            preproc::AudioPreprocConfig aud_cfg;
            aud_cfg.sample_rate = config_.sample_rate;
            aud_cfg.feature_type = preproc::AudioFeatureType::MEL_SPECTROGRAM;
            aud_cfg.n_fft = config_.n_fft;
            aud_cfg.n_mels = config_.n_mels;
            aud_cfg.log_mel = true; // Log-Mel is standard for CNNs
            preproc_->init(aud_cfg);
        }
        // Note: If raw waveform is used, we might handle simple cropping/padding manually
        // or use a preprocessor in RAW mode.

        // 3. Setup Post-processor (Binary Classification)
        postproc_ = postproc::create_classification(config_.target);
        postproc::ClassificationConfig cls_cfg;
        cls_cfg.top_k = 2; // Real vs Fake
        cls_cfg.apply_softmax = true;

        // Define standard labels for clarity
        // Index 0 usually "Real", Index 1 "Fake" (Model dependent, adjust based on training)
        cls_cfg.labels = {"Real", "Fake"};
        postproc_->init(cls_cfg);
    }

    // Helper: Pad or Trim audio to exact duration
    std::vector<float> normalize_length(const std::vector<float>& pcm) {
        size_t target_len = config_.sample_rate * config_.duration_sec;

        if (pcm.size() == target_len) return pcm;

        std::vector<float> fixed(target_len, 0.0f);
        if (pcm.size() > target_len) {
            // Trim (Center crop or just take beginning?)
            // Taking beginning is safer for voice onset
            std::copy(pcm.begin(), pcm.begin() + target_len, fixed.begin());
        } else {
            // Pad / Repeat
            // Loop playback padding is often better than zero padding for spectrograms
            size_t current = 0;
            while (current < target_len) {
                size_t chunk = std::min(pcm.size(), target_len - current);
                std::copy(pcm.begin(), pcm.begin() + chunk, fixed.begin() + current);
                current += chunk;
            }
        }
        return fixed;
    }
};

// =============================================================================