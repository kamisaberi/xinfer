#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>

using namespace xinfer;

// Helper to colorize class IDs
cv::Mat colorize_mask(const core::Tensor& mask_tensor, int width, int height) {
    // Mask is [1, H, W] uint8
    const uint8_t* data = static_cast<const uint8_t*>(mask_tensor.data());
    cv::Mat mask(height, width, CV_8UC1, const_cast<uint8_t*>(data));

    cv::Mat color_mask;
    // Apply a colormap (e.g., JET or custom LUT)
    // Note: cv::applyColorMap expects 8UC1 or 8UC3
    // We scale IDs to visible range if class count is low
    cv::Mat scaled_mask = mask * 10; // e.g. class 1->10, class 2->20 for visibility
    cv::applyColorMap(scaled_mask, color_mask, cv::COLORMAP_JET);

    return color_mask;
}

int main() {
    // Target: NVIDIA GPU (Good for high-res segmentation)
    Target target = Target::NVIDIA_TRT;
    std::string model_path = "deeplabv3_mobilenet.engine";

    // 1. Setup Engine
    auto engine = backends::BackendFactory::create(target);
    engine->load_model(model_path);

    // 2. Preprocessor (512x512 input)
    auto preproc = preproc::create_image_preprocessor(target);
    preproc::ImagePreprocConfig pre_cfg;
    pre_cfg.target_width = 512;
    pre_cfg.target_height = 512;
    pre_cfg.norm_params = {{123.675, 116.28, 103.53}, {58.395, 57.12, 57.375}}; // Standard mean/std
    preproc->init(pre_cfg);

    // 3. Postprocessor
    // We want the mask resized back to original image size (e.g. 1920x1080)
    auto postproc = postproc::create_segmentation(target);
    postproc::SegmentationConfig post_cfg;
    post_cfg.target_width = 1920;
    post_cfg.target_height = 1080;
    postproc->init(post_cfg);

    // 4. Run
    cv::Mat img = cv::imread("street_scene.jpg"); // 1920x1080
    if (img.empty()) return -1;

    core::Tensor input, output;

    preproc::process({img.data, img.cols, img.rows, preproc::ImageFormat::BGR}, input);
    engine->predict({input}, {output});

    // The post-processor handles the ArgMax and the resizing to 1920x1080
    auto result = postproc->process(output);

    // 5. Visualize
    cv::Mat color_mask = colorize_mask(result.mask, 1920, 1080);

    // Blend with original
    cv::Mat blended;
    cv::addWeighted(img, 0.6, color_mask, 0.4, 0.0, blended);

    cv::imwrite("seg_result.jpg", blended);
    std::cout << "Segmentation mask saved." << std::endl;

    return 0;
}