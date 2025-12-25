#include <iostream>
#include <opencv2/opencv.hpp>

#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>

using namespace xinfer;

int main() {
    // Target: Rockchip RK3588 (Common in industrial cameras)
    Target target = Target::ROCKCHIP_RKNN;
    std::string model_path = "mvtec_ad_bottle.rknn";

    // 1. Setup Engine
    auto engine = backends::BackendFactory::create(target);
    engine->load_model(model_path);

    // 2. Preprocessor (256x256)
    // IMPORTANT: Anomaly detection is very sensitive to normalization.
    // Ensure this matches training exactly.
    auto preproc = preproc::create_image_preprocessor(target);
    preproc::ImagePreprocConfig pre_cfg;
    pre_cfg.target_width = 256;
    pre_cfg.target_height = 256;
    pre_cfg.norm_params.scale_factor = 1.0f / 255.0f; // [0, 1] range
    preproc->init(pre_cfg);

    // 3. Postprocessor (Difference & Heatmap)
    auto postproc = postproc::create_anomaly(target);
    postproc::AnomalyConfig anom_cfg;
    anom_cfg.threshold = 0.45f;   // MSE score above this is anomaly
    anom_cfg.use_smoothing = true; // Blur the heatmap to reduce noise
    postproc->init(anom_cfg);

    // 4. Run Loop
    cv::Mat img = cv::imread("good_bottle.png");
    // cv::Mat img = cv::imread("broken_bottle.png"); // Test this one too

    core::Tensor input, reconstruction;

    // A. Preprocess
    // We keep the input tensor because we need it for comparison later!
    preproc::process({img.data, img.cols, img.rows, preproc::ImageFormat::BGR}, input);

    // B. Inference (Autoencoder: Input -> Bottleneck -> Reconstruction)
    engine->predict({input}, {reconstruction});

    // C. Post-process
    // Compares 'input' vs 'reconstruction'
    // Returns Score, Heatmap (Float), and Segmentation Mask (Binary)
    auto result = postproc->process(input, reconstruction);

    // 5. Report
    if (result.is_anomaly) {
        std::cout << "[FAIL] Anomaly Detected! Score: " << result.anomaly_score << std::endl;

        // Save Heatmap for debugging
        // Heatmap is a Float tensor [0..1], scale to 255 for saving
        core::Tensor& map = result.heatmap;
        int h = map.shape()[2];
        int w = map.shape()[3];

        cv::Mat heatmap(h, w, CV_32F, map.data());
        cv::Mat heatmap_8u;
        heatmap.convertTo(heatmap_8u, CV_8U, 255.0);
        cv::applyColorMap(heatmap_8u, heatmap_8u, cv::COLORMAP_JET);

        cv::imwrite("anomaly_heatmap.jpg", heatmap_8u);
    } else {
        std::cout << "[PASS] Part is OK. Score: " << result.anomaly_score << std::endl;
    }

    return 0;
}