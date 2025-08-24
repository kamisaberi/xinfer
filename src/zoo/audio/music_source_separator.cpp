#include <include/zoo/audio/music_source_separator.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>

namespace xinfer::zoo::audio {

struct MusicSourceSeparator::Impl {
    MusicSourceSeparatorConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::AudioProcessor> preprocessor_;
};

MusicSourceSeparator::MusicSourceSeparator(const MusicSourceSeparatorConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Music source separator engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    if (pimpl_->engine_->get_num_outputs() != pimpl_->config_.source_names.size()) {
        throw std::runtime_error("Engine output count does not match the number of source names in config.");
    }

    pimpl_->preprocessor_ = std::make_unique<preproc::AudioProcessor>(pimpl_->config_.audio_config);
}

MusicSourceSeparator::~MusicSourceSeparator() = default;
MusicSourceSeparator::MusicSourceSeparator(MusicSourceSeparator&&) noexcept = default;
MusicSourceSeparator& MusicSourceSeparator::operator=(MusicSourceSeparator&&) noexcept = default;

std::map<std::string, AudioWaveform> MusicSourceSeparator::predict(const std::vector<float>& mixed_waveform) {
    if (!pimpl_) throw std::runtime_error("MusicSourceSeparator is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor spectrogram_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(mixed_waveform, spectrogram_tensor);

    auto output_tensors = pimpl_->engine_->infer({spectrogram_tensor});

    std::map<std::string, AudioWaveform> results;

    // The conversion from spectrogram back to waveform (vocoding) is a very complex process.
    // A full implementation would use a separate vocoder model or an algorithm like Griffin-Lim.
    // For this example, we will treat the output spectrograms as the result, acknowledging
    // that a final vocoding step is needed in a real application.
    for (size_t i = 0; i < pimpl_->config_.source_names.size(); ++i) {
        const auto& source_spectrogram = output_tensors[i];
        AudioWaveform placeholder_waveform(source_spectrogram.num_elements());
        source_spectrogram.copy_to_host(placeholder_waveform.data());
        results[pimpl_->config_.source_names[i]] = placeholder_waveform;
    }

    return results;
}

} // namespace xinfer::zoo::audio