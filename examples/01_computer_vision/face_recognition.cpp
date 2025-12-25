#include <opencv2/opencv.hpp>
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>

using namespace xinfer;

int main() {
    // Target: Rockchip RK3588 (NPU handles both models easily)
    Target target = Target::ROCKCHIP_RKNN;

    // --- Model 1: Face Detection (RetinaFace/YOLO-Face) ---
    auto detect_eng = backends::BackendFactory::create(target);
    detect_eng->load_model("retinaface.rknn");

    auto detect_pre = preproc::create_image_preprocessor(target);
    detect_pre->init({640, 640, preproc::ImageFormat::RGB});

    auto detect_post = postproc::create_detection(target); // Standard box decoder
    detect_post->init({0.6f, 0.4f}); // Strict confidence

    // --- Model 2: Face Recognition (ArcFace) ---
    auto rec_eng = backends::BackendFactory::create(target);
    rec_eng->load_model("arcface_mobile.rknn");

    auto rec_pre = preproc::create_image_preprocessor(target);
    // ArcFace usually takes 112x112
    rec_pre->init({112, 112, preproc::ImageFormat::RGB});

    // --- Database (Mock) ---
    std::vector<float> known_embedding = {/*... 512 floats ...*/};

    // --- Run ---
    cv::Mat frame = cv::imread("office.jpg");
    core::Tensor t_det_in, t_det_out;
    core::Tensor t_rec_in, t_rec_out;

    // 1. Detect
    detect_pre->process({frame.data, frame.cols, frame.rows, preproc::ImageFormat::BGR}, t_det_in);
    detect_eng->predict({t_det_in}, {t_det_out});
    auto faces = detect_post->process({t_det_out});

    // 2. Recognize Loop
    for (const auto& face : faces) {
        // Crop face
        cv::Rect roi(face.x1, face.y1, face.x2-face.x1, face.y2-face.y1);
        cv::Mat face_img = frame(roi);

        // Preprocess crop (Resize 112x112)
        rec_pre->process({face_img.data, face_img.cols, face_img.rows, preproc::ImageFormat::BGR}, t_rec_in);

        // Get Embedding
        rec_eng->predict({t_rec_in}, {t_rec_out});

        // Compare (Cosine Similarity)
        const float* curr_emb = (float*)t_rec_out.data();
        float score = 0.0f; // calc_cosine(curr_emb, known_embedding);

        std::string name = (score > 0.6f) ? "Admin" : "Unknown";

        cv::rectangle(frame, roi, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, name, roi.tl(), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0));
    }

    cv::imwrite("faces.jpg", frame);
    return 0;
}