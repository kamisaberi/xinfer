#include <iostream>
#include <opencv2/opencv.hpp>
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/video/video_preprocessor.h> // Frame Stacker
#include <xinfer/postproc/factory.h>

using namespace xinfer;

int main() {
    // Target: Intel iGPU (good for 3D CNNs like ResNet3D or R(2+1)D)
    Target target = Target::INTEL_OV;

    // 1. Setup Inference Engine
    auto engine = backends::BackendFactory::create(target);
    engine->load_model("action_resnet3d.xml");

    // 2. Setup Video Preprocessor (Ring Buffer)
    // We implemented 'CpuFrameStacker' in src/preproc/video/cpu/frame_stacker.cpp
    auto video_prep = std::make_unique<preproc::CpuFrameStacker>(); // Direct instantiation or via factory

    preproc::VideoConfig v_cfg;
    v_cfg.time_steps = 16;  // Model needs last 16 frames
    v_cfg.height = 112;
    v_cfg.width = 112;
    v_cfg.channels = 3;
    video_prep->init(v_cfg);

    // 3. Setup Classification Postproc
    auto postproc = postproc::create_classification(target);
    postproc::ClassificationConfig cls_cfg;
    cls_cfg.top_k = 3;
    cls_cfg.labels = {"Walking", "Running", "Fighting", "Falling", "Standing"};
    postproc->init(cls_cfg);

    // 4. Capture Loop
    cv::VideoCapture cap("security_cam.mp4");
    cv::Mat frame;
    core::Tensor input_stack; // 5D Tensor
    core::Tensor output_logits;

    while (cap.read(frame)) {
        // A. Push frame into Ring Buffer
        // The preprocessor resizes the frame and shifts the buffer
        preproc::ImageFrame img{frame.data, frame.cols, frame.rows, preproc::ImageFormat::BGR};

        video_prep->push_and_get(img, input_stack);

        // B. Inference (Run every frame or skip frames for speed)
        engine->predict({input_stack}, {output_logits});

        // C. Post-process
        auto results = postproc->process(output_logits);
        auto top1 = results[0][0];

        // D. Display
        if (top1.score > 0.7f) {
            std::string text = top1.label + " (" + std::to_string(top1.score) + ")";
            cv::putText(frame, text, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

            if (top1.label == "Fighting") {
                std::cout << "[ALERT] Violence detected!" << std::endl;
            }
        }

        cv::imshow("Action Rec", frame);
        if (cv::waitKey(30) == 27) break;
    }
    return 0;
}