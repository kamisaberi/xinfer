#include <include/zoo/generative/text_to_speech.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
// #include <xinfer/preproc/text_tokenizer.h>

namespace xinfer::zoo::generative {

struct TextToSpeech::Impl {
    TextToSpeechConfig config_;
    std::unique_ptr<core::InferenceEngine> spectrogram_engine_;
    std::unique_ptr<core::InferenceEngine> vocoder_engine_;
    // std::unique_ptr<preproc::TextTokenizer> tokenizer_;
};

TextToSpeech::TextToSpeech(const TextToSpeechConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.spectrogram_engine_path).good()) {
        throw std::runtime_error("TTS Spectrogram engine file not found: " + pimpl_->config_.spectrogram_engine_path);
    }
    if (!std::ifstream(pimpl_->config_.vocoder_engine_path).good()) {
        throw std::runtime_error("TTS Vocoder engine file not found: " + pimpl_->config_.vocoder_engine_path);
    }

    pimpl_->spectrogram_engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.spectrogram_engine_path);
    pimpl_->vocoder_engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.vocoder_engine_path);
    // pimpl_->tokenizer_ = std::make_unique<preproc::TextTokenizer>("path/to/vocab.json");
}

TextToSpeech::~TextToSpeech() = default;
TextToSpeech::TextToSpeech(TextToSpeech&&) noexcept = default;
TextToSpeech& TextToSpeech::operator=(TextToSpeech&&) noexcept = default;

AudioWaveform TextToSpeech::predict(const std::string& text) {
    if (!pimpl_) throw std::runtime_error("TextToSpeech is in a moved-from state.");

    // --- Stage 1: Text to Mel-Spectrogram ---
    // std::vector<int> tokenized_text = pimpl_->tokenizer_->encode(text);
    // core::Tensor input_tensor({1, (int64_t)tokenized_text.size()}, core::DataType::kINT32);
    // input_tensor.copy_from_host(tokenized_text.data());

    // auto mel_spectrogram_tensors = pimpl_->spectrogram_engine_->infer({input_tensor});
    // const core::Tensor& mel_spectrogram = mel_spectrogram_tensors[0];

    // --- Stage 2: Mel-Spectrogram to Waveform (Vocoder) ---
    // auto waveform_tensors = pimpl_->vocoder_engine_->infer({mel_spectrogram});
    // const core::Tensor& waveform_tensor = waveform_tensors[0];

    AudioWaveform waveform;
    // waveform.resize(waveform_tensor.num_elements());
    // waveform_tensor.copy_to_host(waveform.data());

    return waveform;
}

} // namespace xinfer::zoo::generative