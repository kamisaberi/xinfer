#include <include/zoo/audio/speech_recognizer.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
#include <include/postproc/ctc_decoder.h>

namespace xinfer::zoo::audio {

struct SpeechRecognizer::Impl {
    SpeechRecognizerConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::AudioProcessor> preprocessor_;
    std::vector<std::string> character_map_;
};

SpeechRecognizer::SpeechRecognizer(const SpeechRecognizerConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Speech recognizer engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    pimpl_->preprocessor_ = std::make_unique<preproc::AudioProcessor>(pimpl_->config_.audio_config);

    if (!pimpl_->config_.character_map_path.empty()) {
        std::ifstream labels_file(pimpl_->config_.character_map_path);
        if (!labels_file) throw std::runtime_error("Could not open character map file: " + pimpl_->config_.character_map_path);
        std::string line;
        while (std::getline(labels_file, line)) {
            pimpl_->character_map_.push_back(line);
        }
    }
}

SpeechRecognizer::~SpeechRecognizer() = default;
SpeechRecognizer::SpeechRecognizer(SpeechRecognizer&&) noexcept = default;
SpeechRecognizer& SpeechRecognizer::operator=(SpeechRecognizer&&) noexcept = default;

TranscriptionResult SpeechRecognizer::predict(const std::vector<float>& waveform) {
    if (!pimpl_) throw std::runtime_error("SpeechRecognizer is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor spectrogram_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(waveform, spectrogram_tensor);

    auto output_tensors = pimpl_->engine_->infer({spectrogram_tensor});
    const core::Tensor& logits_tensor = output_tensors[0];

    auto decoded_result = postproc::ctc::decode(logits_tensor, pimpl_->character_map_);

    TranscriptionResult result;
    result.text = decoded_result.first;
    result.confidence = decoded_result.second;

    return result;
}

} // namespace xinfer::zoo::audio