#include <iostream>
#include <vector>
#include <fstream>
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>

using namespace xinfer;

// Helper to read raw PCM audio file
std::vector<float> read_pcm(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<int16_t> buffer(size / 2);
    file.read((char*)buffer.data(), size);

    std::vector<float> pcm(buffer.size());
    for(size_t i=0; i<buffer.size(); ++i) pcm[i] = buffer[i] / 32768.0f; // Norm to [-1, 1]
    return pcm;
}

int main() {
    // Target: Mobile/Edge CPU (e.g., Cortex-A72)
    Target target = Target::INTEL_OV; // Or generic CPU

    // 1. Setup Audio Preprocessor (Mel Spectrogram)
    auto preproc = preproc::create_audio_preprocessor(target);
    preproc::AudioPreprocConfig audio_cfg;
    audio_cfg.sample_rate = 16000;
    audio_cfg.feature_type = preproc::AudioFeatureType::MEL_SPECTROGRAM;
    audio_cfg.n_mels = 64;
    audio_cfg.n_fft = 1024;
    preproc->init(audio_cfg);

    // 2. Load Model (Audio Spectrogram Transformer or light CNN)
    auto engine = backends::BackendFactory::create(target);
    engine->load_model("audio_event_net.xml");

    // 3. Setup Classification Post-processor
    auto postproc = postproc::create_classification(target);
    postproc::ClassificationConfig cls_cfg;
    cls_cfg.top_k = 1;
    cls_cfg.labels = {"Background", "Glass Break", "Gunshot", "Scream"};
    postproc->init(cls_cfg);

    // 4. Processing Loop
    auto pcm_data = read_pcm("test_clip.pcm");

    // Create tensors
    core::Tensor input_spectrogram;
    core::Tensor output_logits;

    // A. Preprocess: Raw PCM -> Mel Spectrogram Tensor
    // NEON optimization happens automatically inside here on ARM
    preproc::AudioBuffer audio_buf{pcm_data.data(), pcm_data.size(), 16000};
    preproc->process(audio_buf, input_spectrogram);

    // B. Inference
    engine->predict({input_spectrogram}, {output_logits});

    // C. Postprocess
    auto results = postproc->process(output_logits);

    std::cout << "Detected Event: " << results[0][0].label
              << " (" << results[0][0].score * 100.0f << "%)" << std::endl;

    return 0;
}