#include <include/zoo/generative/diffusion_pipeline.h>
#include <include/utils/image_utils.h> // A hypothetical utility header
#include <iostream>

int main() {
    // Pre-build your U-Net engine:
    // xinfer-cli build --onnx unet.onnx --save_engine unet.engine --fp16

    xinfer::zoo::generative::DiffusionPipelineConfig config;
    config.unet_engine_path = "assets/unet.engine";
    config.num_timesteps = 50;

    xinfer::zoo::generative::DiffusionPipeline pipeline(config);

    std::cout << "Generating image with diffusion model (50 steps)...\n";
    xinfer::core::Tensor image_tensor = pipeline.generate(1);

    // Convert the final GPU tensor to a cv::Mat and save
    cv::Mat final_image = xinfer::utils::tensor_to_mat(image_tensor);
    cv::imwrite("diffusion_output.png", final_image);
    std::cout << "Image saved to diffusion_output.png\n";

    return 0;
}