#include <iostream>
#include <opencv2/opencv.hpp>

// xInfer Core
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>

using namespace xinfer;

int main(int argc, char** argv) {
    // ---------------------------------------------------------------
    // 1. CONFIGURATION
    // Change this line to switch hardware!
    // Options: NVIDIA_TRT, ROCKCHIP_RKNN, AMD_VITIS, INTEL_OV...
    Target target = Target::NVIDIA_TRT;
    std::string model_path = "yolov8m.engine"; // or .rknn, .xmodel
    // ---------------------------------------------------------------

    // 2. Load Engine
    auto engine = backends::BackendFactory::create(target);
    if (!engine->load_model(model_path)) {
        std::cerr << "Failed to load model!" << std::endl;
        return -1;
    }

    // 3. Setup Preprocessor (Auto-selects CUDA/RGA/NEON)
    auto preproc = preproc::create_image_preprocessor(target);
    preproc::ImagePreprocConfig pre_cfg;
    pre_cfg.target_width = 640;
    pre_cfg.target_height = 640;
    pre_cfg.target_format = preproc::ImageFormat::RGB;
    pre_cfg.layout_nchw = true; // YOLO standard
    preproc->init(pre_cfg);

    // 4. Setup Postprocessor (Auto-selects CPU/CUDA NMS)
    auto postproc = postproc::create_detection(target);
    postproc::DetectionConfig post_cfg;
    post_cfg.conf_threshold = 0.45f;
    post_cfg.nms_threshold = 0.50f;
    postproc->init(post_cfg);

    // 5. Allocate Tensors
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    // 6. Run Loop
    cv::Mat frame = cv::imread("street.jpg");

    // A. Preprocess
    preproc::ImageFrame img_frame{frame.data, frame.cols, frame.rows, preproc::ImageFormat::BGR};
    preproc->process(img_frame, input_tensor);

    // B. Inference
    engine->predict({input_tensor}, {output_tensor});

    // C. Postprocess
    auto detections = postproc->process({output_tensor});

    // D. Visualization
    for (const auto& det : detections) {
        cv::rectangle(frame, cv::Point(det.x1, det.y1), cv::Point(det.x2, det.y2), cv::Scalar(0, 255, 0), 2);
        std::string label = "Class " + std::to_string(det.class_id) + ": " + std::to_string(det.confidence);
        cv::putText(frame, label, cv::Point(det.x1, det.y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }

    cv::imwrite("result.jpg", frame);
    std::cout << "Saved result.jpg with " << detections.size() << " detections." << std::endl;

    return 0;
}