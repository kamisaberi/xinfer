#include <xinfer/zoo/audio/language_identifier.h>
#include <xinfer/core/logging.h>

// --- We reuse the generic Audio Classifier module ---
#include <xinfer/zoo/audio/classifier.h>

#include <iostream>
#include <vector>
#include <map>

namespace xinfer::zoo::audio {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct LanguageIdentifier::Impl {
    LanguageIdConfig config_;

    // The generic classifier does all the work
    std::unique_ptr<Classifier> classifier_;

    // Optional map from full name to short code
    std::map<std::string, std::string> name_to_code_;

    Impl(const LanguageIdConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Configure the underlying classifier
        ClassifierConfig cls_cfg;
        cls_cfg.target = config_.target;
        cls_cfg.model_path = config_.model_path;
        cls_cfg.labels_path = config_.labels_path;
        cls_cfg.sample_rate = config_.sample_rate;
        cls_cfg.duration_sec = config_.duration_sec;
        cls_cfg.n_fft = config_.n_fft;
        cls_cfg.hop_length = config_.hop_length;
        cls_cfg.n_mels = config_.n_mels;
        cls_cfg.top_k = config_.top_k;
        cls_cfg.confidence_threshold = config_.confidence_threshold;

        classifier_ = std::make_unique<Classifier>(cls_cfg);

        // 2. Populate language code map (optional, can be loaded from a file)
        name_to_code_["English"] = "en";
        name_to_code_["Spanish"] = "es";
        name_to_code_["Chinese"] = "zh";
        name_to_code_["German"] = "de";
        name_to_code_["French"] = "fr";
        // ... and so on
    }
};

// =================================================================================
// Public API
// =================================================================================

LanguageIdentifier::LanguageIdentifier(const LanguageIdConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

LanguageIdentifier::~LanguageIdentifier() = default;
LanguageIdentifier::LanguageIdentifier(LanguageIdentifier&&) noexcept = default;
LanguageIdentifier& LanguageIdentifier::operator=(LanguageIdentifier&&) noexcept = default;

std::vector<LanguageResult> LanguageIdentifier::identify(const std::vector<float>& pcm_data) {
    if (!pimpl_ || !pimpl_->classifier_) {
        throw std::runtime_error("LanguageIdentifier is not initialized.");
    }

    // 1. Delegate to the audio classifier
    auto cls_results = pimpl_->classifier_->classify(pcm_data);

    // 2. Map to the specific LanguageResult struct
    std::vector<LanguageResult> results;
    results.reserve(cls_results.size());

    for (const auto& res : cls_results) {
        LanguageResult lr;
        lr.language_name = res.label;
        lr.confidence = res.confidence;

        // Lookup code
        if (pimpl_->name_to_code_.count(res.label)) {
            lr.language_code = pimpl_->name_to_code_.at(res.label);
        } else {
            lr.language_code = "unk"; // Unknown
        }

        results.push_back(lr);
    }

    return results;
}

} // namespace xinfer::zoo::audio