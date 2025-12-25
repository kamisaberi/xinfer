#include <iostream>
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>

using namespace xinfer;

int main() {
    // Target: Microchip PolarFire FPGA (VectorBlox IP)
    Target target = Target::MICROCHIP_VECTORBLOX;

    // 1. Setup Engine
    auto engine = backends::BackendFactory::create(target);

    xinfer::Config vbx_config;
    vbx_config.model_path = "kws_v1000.blob"; // Blob compiled for V1000 core
    // VectorBlox specific param: Specify core size to validate binary
    vbx_config.vendor_params = { "CORE=V1000" };

    engine->load_model(vbx_config.model_path);

    // 2. Setup Audio Preproc (MFCC)
    // On RISC-V CPU (PolarFire MSS), NEON isn't available,
    // so this will fall back to the generic C++ KissFFT implementation automatically.
    auto preproc = preproc::create_audio_preprocessor(target);

    preproc::AudioPreprocConfig audio_cfg;
    audio_cfg.sample_rate = 16000;
    audio_cfg.feature_type = preproc::AudioFeatureType::MFCC; // Speech specific
    audio_cfg.n_mfcc = 13;
    preproc->init(audio_cfg);

    // 3. Audio Loop (Mocked)
    std::vector<float> audio_chunk(16000); // 1 sec buffer
    core::Tensor input, output;

    while (true) {
        // fill_audio(audio_chunk);

        // MFCC
        preproc::AudioBuffer buf{audio_chunk.data(), audio_chunk.size(), 16000};
        preproc->process(buf, input);

        // Inference (Runs on FPGA Fabric)
        // Cache coherency is handled inside VectorBloxBackend
        engine->predict({input}, {output});

        // ArgMax
        float* probs = (float*)output.data();
        int k = std::distance(probs, std::max_element(probs, probs+10));

        if (k == 1) std::cout << "Keyword Detected!" << std::endl;
    }
}