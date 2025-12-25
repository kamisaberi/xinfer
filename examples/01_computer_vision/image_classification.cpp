#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

// xInfer Core Headers
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>

using namespace xinfer;

int main(int argc, char** argv) {
    // -------------------------------------------------------------------------
    // 1. CONFIGURATION
    // -------------------------------------------------------------------------
    Target target = Target::INTEL_OV; // e.g., Run on CPU/iGPU
    std::string model_path = "resnet50.xml";
    std::string image_path = "dog.jpg";

    // -------------------------------------------------------------------------
    // 2. SETUP PIPELINE
    // -------------------------------------------------------------------------

    // A. Backend
    auto engine = backends::BackendFactory::create(target);
    if (!engine->load_model(model_path)) {
        std::cerr << "Failed to load model: " << model_path << std::endl;
        return -1;
    }

    // B. Preprocessor
    // ResNet expects: 224x224, RGB, Mean=[0.485, ...], Std=[0.229, ...]
    auto preproc = preproc::create_image_preprocessor(target);

    preproc::ImagePreprocConfig pre_cfg;
    pre_cfg.target_width = 224;
    pre_cfg.target_height = 224;
    pre_cfg.target_format = preproc::ImageFormat::RGB;
    pre_cfg.layout_nchw = true; // Standard for PyTorch/ONNX models

    // ImageNet Normalization Constants
    pre_cfg.norm_params.mean = {0.485f * 255, 0.456f * 255, 0.406f * 255};
    pre_cfg.norm_params.std  = {0.229f * 255, 0.224f * 255, 0.225f * 255};
    pre_cfg.norm_params.scale_factor = 1.0f; // Input is 0-255, mean/std handles scaling

    preproc->init(pre_cfg);

    // C. Postprocessor
    auto postproc = postproc::create_classification(target);

    postproc::ClassificationConfig post_cfg;
    post_cfg.top_k = 5;
    post_cfg.apply_softmax = true; // If model outputs raw logits
    // Optional: Load labels from file in real app
    post_cfg.labels = {"Tench", "Goldfish", "Great White Shark", "..." /* 1000 classes */};

    postproc->init(post_cfg);

    // -------------------------------------------------------------------------
    // 3. INFERENCE LOOP
    // -------------------------------------------------------------------------
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Could not read image." << std::endl;
        return -1;
    }

    core::Tensor input_tensor, output_tensor;

    // Preprocess (CPU or GPU based on target)
    preproc::ImageFrame frame{img.data, img.cols, img.rows, preproc::ImageFormat::BGR};
    preproc->process(frame, input_tensor);

    // Inference
    engine->predict({input_tensor}, {output_tensor});

    // Postprocess
    // Returns batch of results (we take batch index 0)
    auto results = postproc->process(output_tensor);
    auto& top_k = results[0];

    // -------------------------------------------------------------------------
    // 4. DISPLAY RESULTS
    // -------------------------------------------------------------------------
    std::cout << "--- Classification Results ---" << std::endl;
    for (const auto& res : top_k) {
        std::cout << "ID: " << res.id
                  << " | Score: " << (res.score * 100.0f) << "%"
                  << " | Label: " << res.label << std::endl;
    }

    return 0;
}