#include <include/zoo/vision/license_plate_recognizer.h>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <algorithm> // For std::min

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>
#include <include/postproc/detection.h>
#include <include/postproc/yolo_decoder.h> // For detecting plate bounding boxes
#include <include/postproc/ocr_decoder.h>  // For warping plate patches and CTC decoding

namespace xinfer::zoo::vision {

const int MAX_DETECTED_PLATES = 10; // Max number of plates to detect per image

struct LicensePlateRecognizer::Impl {
    LicensePlateRecognizerConfig config_;

    std::unique_ptr<core::InferenceEngine> engine_detector_;
    std::unique_ptr<core::InferenceEngine> engine_recognizer_;

    std::unique_ptr<preproc::ImageProcessor> preprocessor_detector_;
    // No specific preprocessor for recognizer, as patches are handled per-plate

    std::vector<char> character_map_; // For CTC decoding

    // Buffers for detection stage output
    core::Tensor decoded_boxes_gpu;
    core::Tensor decoded_scores_gpu;
    core::Tensor decoded_classes_gpu;
    std::vector<float> h_boxes_decoded; // Host buffer for detected boxes
    std::vector<float> h_scores_decoded;
    std::vector<int> h_classes_decoded;
};

LicensePlateRecognizer::LicensePlateRecognizer(const LicensePlateRecognizerConfig& config)
    : pimpl_(new Impl{config})
{
    // Validate detection engine
    if (!std::ifstream(pimpl_->config_.detection_engine_path).good()) {
        throw std::runtime_error("License plate detection engine file not found: " + pimpl_->config_.detection_engine_path);
    }
    // Validate recognition engine
    if (!std::ifstream(pimpl_->config_.recognition_engine_path).good()) {
        throw std::runtime_error("License plate recognition engine file not found: " + pimpl_->config_.recognition_engine_path);
    }

    // Load engines
    pimpl_->engine_detector_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.detection_engine_path);
    pimpl_->engine_recognizer_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.recognition_engine_path);

    // Initialize preprocessor for the detection stage (YOLO-style)
    pimpl_->preprocessor_detector_ = std::make_unique<preproc::ImageProcessor>(
        pimpl_->config_.detection_input_width,
        pimpl_->config_.detection_input_height,
        true // Enable letterbox padding
    );

    // Initialize buffers for detection results
    pimpl_->decoded_boxes_gpu = core::Tensor({MAX_DETECTED_PLATES, 4}, core::DataType::kFLOAT);
    pimpl_->decoded_scores_gpu = core::Tensor({MAX_DETECTED_PLATES}, core::DataType::kFLOAT);
    pimpl_->decoded_classes_gpu = core::Tensor({MAX_DETECTED_PLATES}, core::DataType::kINT32);
    pimpl_->h_boxes_decoded.resize(MAX_DETECTED_PLATES * 4);
    pimpl_->h_scores_decoded.resize(MAX_DETECTED_PLATES);
    pimpl_->h_classes_decoded.resize(MAX_DETECTED_PLATES);

    // Prepare character map for CTC decoding
    pimpl_->character_map_.push_back('-'); // CTC blank character at index 0
    for (char c : pimpl_->config_.character_set) {
        pimpl_->character_map_.push_back(c);
    }
}

LicensePlateRecognizer::~LicensePlateRecognizer() = default;
LicensePlateRecognizer::LicensePlateRecognizer(LicensePlateRecognizer&&) noexcept = default;
LicensePlateRecognizer& LicensePlateRecognizer::operator=(LicensePlateRecognizer&&) noexcept = default;

std::vector<LPResult> LicensePlateRecognizer::predict(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("LicensePlateRecognizer is in a moved-from state.");

    // --- STAGE 1: LICENSE PLATE DETECTION ---

    // 1a. Pre-process the full image for the detection model
    auto input_shape_det = pimpl_->engine_detector_->get_input_shape(0);
    core::Tensor input_tensor_det(input_shape_det, core::DataType::kFLOAT);
    pimpl_->preprocessor_detector_->process(image, input_tensor_det);

    // 1b. Run the detection model
    auto output_tensors_det = pimpl_->engine_detector_->infer({input_tensor_det});
    const core::Tensor& raw_output_det = output_tensors_det[0];

    // 1c. Decode detection output and run NMS on GPU
    // Assumes YOLO-style output from detector model
    postproc::yolo::decode(raw_output_det, pimpl_->config_.detection_confidence_threshold,
                           pimpl_->decoded_boxes_gpu, pimpl_->decoded_scores_gpu, pimpl_->decoded_classes_gpu);

    std::vector<int> nms_indices = postproc::detection::nms(
        pimpl_->decoded_boxes_gpu, pimpl_->decoded_scores_gpu, pimpl_->config_.detection_nms_iou_threshold);

    // 1d. Copy results back to CPU for iteration and further processing
    if (nms_indices.empty()) {
        return {}; // No plates detected
    }
    pimpl_->decoded_boxes_gpu.copy_to_host(pimpl_->h_boxes_decoded.data());
    pimpl_->decoded_scores_gpu.copy_to_host(pimpl_->h_scores_decoded.data());
    pimpl_->decoded_classes_gpu.copy_to_host(pimpl_->h_classes_decoded.data());

    // Calculate scaling factors for original image coordinates
    float scale_x = (float)image.cols / pimpl_->config_.detection_input_width;
    float scale_y = (float)image.rows / pimpl_->config_.detection_input_height;

    // --- STAGE 2: LICENSE PLATE RECOGNITION (per detected plate) ---

    std::vector<LPResult> final_results;
    for (int idx : nms_indices) {
        // Get detected box coordinates
        float x1_norm = pimpl_->h_boxes_decoded[idx * 4 + 0];
        float y1_norm = pimpl_->h_boxes_decoded[idx * 4 + 1];
        float x2_norm = pimpl_->h_boxes_decoded[idx * 4 + 2];
        float y2_norm = pimpl_->h_boxes_decoded[idx * 4 + 3];

        // Convert normalized coordinates back to original image pixel space
        std::vector<cv::Point2f> lp_box_points = {
            {x1_norm * scale_x, y1_norm * scale_y},
            {x2_norm * scale_x, y1_norm * scale_y},
            {x2_norm * scale_x, y2_norm * scale_y},
            {x1_norm * scale_x, y2_norm * scale_y}
        };

        // 2a. Crop and warp the license plate region
        cv::Mat plate_patch = postproc::ocr::get_warped_text_patch(image, lp_box_points, pimpl_->config_.recognition_input_height);

        // 2b. Pre-process the plate patch for the recognition model (typically grayscale and normalize)
        cv::Mat gray_patch;
        cv::cvtColor(plate_patch, gray_patch, cv::COLOR_BGR2GRAY);
        gray_patch.convertTo(gray_patch, CV_32F, 1.0 / 127.5, -1.0); // Normalize to [-1, 1]

        // Reshape to NCHW for the recognizer engine (Batch=1, Channels=1, H, W)
        auto rec_input_shape = pimpl_->engine_recognizer_->get_input_shape(0);
        rec_input_shape[0] = 1; // Always batch size 1 for individual plate patches
        rec_input_shape[1] = 1; // Grayscale has 1 channel
        rec_input_shape[2] = gray_patch.rows;
        rec_input_shape[3] = gray_patch.cols;

        core::Tensor input_tensor_rec(rec_input_shape, core::DataType::kFLOAT);
        input_tensor_rec.copy_from_host(gray_patch.data);

        // 2c. Run the recognition model
        // We need to set the input shape on the context as recognizers often have dynamic width
        pimpl_->engine_recognizer_->setInputShape("input", rec_input_shape);
        auto output_tensors_rec = pimpl_->engine_recognizer_->infer({input_tensor_rec});

        // 2d. Decode the recognition output (CTC greedy decode)
        LPResult result;
        std::tie(result.text, result.confidence) = postproc::ocr::decode_recognition_output_ctc(
            output_tensors_rec[0], pimpl_->character_map_);

        result.box_points = lp_box_points; // Assign the original box points

        // Add to final results if confidence is above a threshold
        if (result.confidence > pimpl_->config_.detection_confidence_threshold) { // Re-use detection threshold
            final_results.push_back(result);
        }
    }

    return final_results;
}

} // namespace xinfer::zoo::vision