#include <iostream>
#include <vector>
#include <fstream>
#include <string>

// xInfer Headers
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>

using namespace xinfer;

/**
 * @brief Simple helper to read a raw PCM file (16-bit Mono 16kHz)
 * In a real app, use libsndfile or ffmpeg.
 */
std::vector<float> load_audio_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open audio file " << path << std::endl;
        return {};
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Assume 16-bit PCM (2 bytes per sample)
    std::vector<int16_t> raw_buffer(size / 2);
    if (!file.read((char*)raw_buffer.data(), size)) return {};

    // Convert to normalized float [-1.0, 1.0]
    std::vector<float> float_buffer(raw_buffer.size());
    for (size_t i = 0; i < raw_buffer.size(); ++i) {
        float_buffer[i] = raw_buffer[i] / 32768.0f;
    }

    std::cout << "Loaded " << float_buffer.size() << " samples ("
              << (float_buffer.size() / 16000.0f) << " seconds)." << std::endl;
    return float_buffer;
}

int main(int argc, char** argv) {
    // -------------------------------------------------------------------------
    // 1. CONFIGURATION
    // -------------------------------------------------------------------------
    // Target: CPU is usually sufficient for single-stream Audio.
    // Use NVIDIA_TRT for high-throughput batching (call centers).
    Target target = Target::INTEL_OV;

    std::string model_path = "quartznet_15x5.xml"; // OpenVINO IR model
    std::string audio_path = "sample_speech.pcm";

    // -------------------------------------------------------------------------
    // 2. PIPELINE SETUP
    // -------------------------------------------------------------------------

    // A. Backend (Inference)
    auto engine = backends::BackendFactory::create(target);
    if (!engine->load_model(model_path)) {
        std::cerr << "Failed to load ASR model!" << std::endl;
        return -1;
    }

    // B. Preprocessor (Mel Spectrogram)
    // Configuration depends on specific model training (QuartzNet vs Wav2Vec)
    // These settings are typical for QuartzNet/Jasper.
    auto preproc = preproc::create_audio_preprocessor(target);

    preproc::AudioPreprocConfig audio_cfg;
    audio_cfg.sample_rate = 16000;
    audio_cfg.feature_type = preproc::AudioFeatureType::MEL_SPECTROGRAM;
    audio_cfg.n_fft = 512;       // 32ms window
    audio_cfg.hop_length = 160;  // 10ms stride
    audio_cfg.n_mels = 64;       // Input height for model
    audio_cfg.log_mel = true;    // Logarithmic scaling is crucial
    audio_cfg.fmin = 0.0f;
    audio_cfg.fmax = 8000.0f;    // Nyquist

    preproc->init(audio_cfg);

    // C. Postprocessor (CTC Decoder)
    // Converts probabilities [Time, Vocab] -> String
    auto postproc = postproc::create_ocr(target);

    postproc::OcrConfig ctc_cfg;
    // Vocabulary must match training. Index 0 is usually blank.
    ctc_cfg.blank_index = 28; // Example: 26 letters + space + blank = 28
    ctc_cfg.vocabulary = " abcdefghijklmnopqrstuvwxyz'_"; // Mapping index to char
    ctc_cfg.min_confidence = 0.0f; // No thresholding for greedy search

    postproc->init(ctc_cfg);

    // -------------------------------------------------------------------------
    // 3. EXECUTION LOOP
    // -------------------------------------------------------------------------

    // Load Data
    std::vector<float> pcm_data = load_audio_file(audio_path);
    if (pcm_data.empty()) return -1;

    core::Tensor input_spectrogram;
    core::Tensor output_logits;

    // Step 1: Preprocess (PCM -> Spectrogram)
    // If target is NVIDIA, this might run via cuFFT
    std::cout << "Preprocessing..." << std::endl;
    preproc::AudioBuffer audio_buf{pcm_data.data(), pcm_data.size(), 16000};
    preproc->process(audio_buf, input_spectrogram);

    // Step 2: Inference
    std::cout << "Running Inference..." << std::endl;
    // Input shape will be [1, 64, TimeSteps]
    engine->predict({input_spectrogram}, {output_logits});

    // Step 3: Decode (CTC Greedy Search)
    std::cout << "Decoding..." << std::endl;
    auto results = postproc->process(output_logits);

    // -------------------------------------------------------------------------
    // 4. OUTPUT
    // -------------------------------------------------------------------------
    std::cout << "\n----------------------------------------" << std::endl;
    std::cout << "TRANSCRIPT: " << results[0] << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    return 0;
}