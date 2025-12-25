#include <opencv2/opencv.hpp>
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>

using namespace xinfer;

int main() {
    Target target = Target::ROCKCHIP_RKNN; // Edge Camera

    // --- ENGINE 1: Plate Detector (YOLO) ---
    auto det_engine = backends::BackendFactory::create(target);
    det_engine->load_model("plate_detect.rknn");

    auto det_pre = preproc::create_image_preprocessor(target);
    det_pre->init({640, 640, preproc::ImageFormat::RGB});

    auto det_post = postproc::create_detection(target);
    det_post->init({0.5f, 0.45f}); // High threshold

    // --- ENGINE 2: Text Recognizer (LPRNet / CRNN) ---
    auto ocr_engine = backends::BackendFactory::create(target);
    ocr_engine->load_model("lprnet.rknn");

    auto ocr_pre = preproc::create_image_preprocessor(target);
    ocr_pre->init({94, 24, preproc::ImageFormat::RGB}); // LPRNet input size

    auto ocr_post = postproc::create_ocr(target);
    postproc::OcrConfig ocr_cfg;
    ocr_cfg.vocabulary = "-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"; // '-' is blank
    ocr_cfg.blank_index = 0;
    ocr_post->init(ocr_cfg);

    // --- LOOP ---
    cv::Mat frame = cv::imread("car.jpg");
    core::Tensor t_det_in, t_det_out, t_ocr_in, t_ocr_out;

    // 1. Detect
    det_pre->process({frame.data, frame.cols, frame.rows, preproc::ImageFormat::BGR}, t_det_in);
    det_engine->predict({t_det_in}, {t_det_out});
    auto plates = det_post->process({t_det_out});

    // 2. Recognize each plate
    for (const auto& plate : plates) {
        // Crop
        cv::Rect roi(plate.x1, plate.y1, plate.x2 - plate.x1, plate.y2 - plate.y1);
        // Safety check bounds...
        cv::Mat plate_img = frame(roi);

        // Preprocess Crop
        ocr_pre->process({plate_img.data, plate_img.cols, plate_img.rows, preproc::ImageFormat::BGR}, t_ocr_in);

        // Infer
        ocr_engine->predict({t_ocr_in}, {t_ocr_out});

        // Decode (CTC)
        auto text = ocr_post->process(t_ocr_out);

        std::cout << "Plate: " << text[0] << " (Conf: " << plate.confidence << ")" << std::endl;
        cv::rectangle(frame, roi, cv::Scalar(0, 0, 255), 2);
        cv::putText(frame, text[0], roi.tl(), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,255,0), 2);
    }

    cv::imwrite("lpr_result.jpg", frame);
    return 0;
}