#include <include/zoo/vision/face_detector.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>
#include <include/postproc/detection.h>
#include <include/postproc/yolo_decoder.h>

namespace xinfer::zoo::vision {

const int MAX_DECODED_FACES = 1024;

struct FaceDetector::Impl {
    FaceDetectorConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;

    core::Tensor decoded_boxes_gpu;
    core::Tensor decoded_scores_gpu;
    core::Tensor decoded_classes_gpu;

    std::vector<float> h_boxes;
    std::vector<float> h_scores;

    Impl(const FaceDetectorConfig& config) : config_(config) {
        decoded_boxes_gpu = core::Tensor({MAX_DECODED_FACES, 4}, core::DataType::kFLOAT);
        decoded_scores_gpu = core::Tensor({MAX_DECODED_FACES}, core::DataType::kFLOAT);
        decoded_classes_gpu = core::Tensor({MAX_DECODED_FACES}, core::DataType::kINT32);
        h_boxes.resize(MAX_DECODED_FACES * 4);
        h_scores.resize(MAX_DECODED_FACES);
    }
};

FaceDetector::FaceDetector(const FaceDetectorConfig& config) : pimpl_(new Impl(config)) {
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Face detection engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(
        pimpl_->config_.input_width,
        pimpl_->config_.input_height,
        true
    );
}

FaceDetector::~FaceDetector() = default;
FaceDetector::FaceDetector(FaceDetector&&) noexcept = default;
FaceDetector& FaceDetector::operator=(FaceDetector&&) noexcept = default;

std::vector<Face> FaceDetector::predict(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("FaceDetector is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(image, input_tensor);

    auto output_tensors = pimpl_->engine_->infer({input_tensor});
    const core::Tensor& raw_output = output_tensors[0];

    postproc::yolo::decode(raw_output, pimpl_->config_.confidence_threshold,
                           pimpl_->decoded_boxes_gpu, pimpl_->decoded_scores_gpu, pimpl_->decoded_classes_gpu);

    std::vector<int> nms_indices = postproc::detection::nms(
        pimpl_->decoded_boxes_gpu, pimpl_->decoded_scores_gpu, pimpl_->config_.nms_iou_threshold);

    std::vector<Face> final_faces;
    if (nms_indices.empty()) {
        return final_faces;
    }

    pimpl_->decoded_boxes_gpu.copy_to_host(pimpl_->h_boxes.data());
    pimpl_->decoded_scores_gpu.copy_to_host(pimpl_->h_scores.data());

    float scale_x = (float)image.cols / pimpl_->config_.input_width;
    float scale_y = (float)image.rows / pimpl_->config_.input_height;

    for (int idx : nms_indices) {
        Face face;
        face.x1 = pimpl_->h_boxes[idx * 4 + 0] * scale_x;
        face.y1 = pimpl_->h_boxes[idx * 4 + 1] * scale_y;
        face.x2 = pimpl_->h_boxes[idx * 4 + 2] * scale_x;
        face.y2 = pimpl_->h_boxes[idx * 4 + 3] * scale_y;
        face.confidence = pimpl_->h_scores[idx];

        // Landmark decoding would require a model that outputs them
        // and a corresponding decoder kernel. This is a placeholder.
        // For a landmark model, raw_output would have more channels.

        final_faces.push_back(face);
    }

    return final_faces;
}

} // namespace xinfer::zoo::vision