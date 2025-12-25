#include <iostream>
#include <xinfer/backends/backend_factory.h>
#include <xinfer/postproc/factory.h> // For Sampler

using namespace xinfer;

int main() {
    // Target: NVIDIA GPU (Required for acceptable speed)
    Target target = Target::NVIDIA_TRT;

    // 1. Load UNet Model
    auto unet = backends::BackendFactory::create(target);
    unet->load_model("sd_v15_unet_fp16.engine");

    // 2. Setup Diffusion Sampler (DDIM)
    auto sampler = postproc::create_sampler(target);
    postproc::SamplerConfig samp_cfg;
    samp_cfg.num_train_timesteps = 1000;
    samp_cfg.type = postproc::SchedulerType::DDIM;
    sampler->init(samp_cfg);

    // 3. Configure Steps
    int num_inference_steps = 20;
    auto timesteps = sampler->set_timesteps(num_inference_steps);

    // 4. Initial Latents (Random Noise)
    // Shape: [1, 4, 64, 64] for 512x512 image
    core::Tensor latents({1, 4, 64, 64}, core::DataType::kFLOAT);
    // ... fill with gaussian noise ...

    // 5. Text Embeddings (Output from CLIP Text Encoder - omitted for brevity)
    core::Tensor text_embeddings;

    // 6. Denoising Loop
    std::cout << "Generating..." << std::endl;
    core::Tensor noise_pred, next_latents;

    for (long t : timesteps) {
        std::cout << "Step " << t << " / " << num_inference_steps << "\r" << std::flush;

        // A. Inference (UNet predicts noise)
        // Inputs: [Latents, Timestep, TextEmbeddings]
        // Note: We need to wrap scalar 't' into a tensor
        core::Tensor t_tensor({1}, core::DataType::kINT32);
        ((int*)t_tensor.data())[0] = (int)t;

        unet->predict({latents, t_tensor, text_embeddings}, {noise_pred});

        // B. Sampling Step (Compute x_t-1)
        // If Target=NVIDIA, this happens on GPU without copying latents to CPU
        sampler->step(noise_pred, t, latents, next_latents);

        // Update current latents
        // (In optimized C++ tensor class, we'd just swap pointers)
        latents = next_latents;
    }

    // 7. VAE Decode (Latents -> Image)
    // auto vae = backends::BackendFactory::create(target); ...

    std::cout << "\nGeneration Complete." << std::endl;
    return 0;
}