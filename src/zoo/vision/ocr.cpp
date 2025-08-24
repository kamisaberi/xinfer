// src/zoo/vision/ocr.cpp

#include <include/zoo/vision/ocr.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>
#include <include/postproc/ocr_decoder.h> // The key helper functions

namespace xinfer::zoo::vision {

struct OCR::Impl {
    OCRConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_detector_;
    std::unique_ptr<core::InferenceEngine> engine_recognizer_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_detector_;
    std::vector<char> character_map_;
};

OCR::OCR(const OCRConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.detection_engine_path).good()) {
        throw std::runtime_error("Text detection engine file not found: " + pimpl_->config_.detection_engine_path);
    }
    if (!std::ifstream(pimpl_->config_.recognition_engine_path).good()) {
        throw std::runtime_error("Text recognition engine file not found: " + pimpl_->config_.recognition_engine_path);
    }

    pimpl_->engine_detector_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.detection_engine_path);
    pimpl_->engine_recognizer_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.recognition_engine_path);

    auto det_shape = pimpl_->engine_detector_->get_input_shape(0);
    pimpl_->preprocessor_detector_ = std::make_unique<preproc::ImageProcessor>(
        det_shape[3], // width from engine
        det_shape[2], // height from engine
        std::vector<float>{0.485f, 0.456f, 0.406f},
        std::vector<float>{0.229f, 0.224f, 0.225f}
    );

    pimpl_->character_map_.push_back('-'); // CTC blank character
    for (char c : pimpl_->config_.character_set) {
        pimpl_->character_map_.push_back(c);
    }
}

OCR::~OCR() = default;
OCR::OCR(OCR&&) noexcept = default;
OCR& OCR::operator=(OCR&&) noexcept = default;

std::vector<OCRResult> OCR::predict(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("OCR object is in a moved-from state.");

    // --- STAGE 1: TEXT DETECTION ---
    auto input_shape_det = pimpl_->engine_detector_->get_input_shape(0);
    core::Tensor input_tensor_det(input_shape_det, core::DataType::kFLOAT);
    pimpl_->preprocessor_detector_->process(image, input_tensor_det);

    auto output_tensors_det = pimpl_->engine_detector_->infer({input_tensor_det});

    // The CRAFT model has two outputs: region scores and affinity scores
    std::vector<std::vector<cv::Point2f>> detected_boxes = postproc::ocr::decode_detection_output_craft(
        output_tensors_det[0], // region scores
        output_tensors_det[1], // affinity scores
        pimpl_->config_.text_threshold,
        pimpl_->config_.box_threshold,
        (float)image.cols / input_shape_det[3],
        (float)image.rows / input_shape_det[2]
    );

    // --- STAGE 2: TEXT RECOGNITION (BATCHED) ---
    std::vector<OCRResult> final_results;
    if (detected_boxes.empty()) return final_results;

    // Create a batch of text patches for efficient inference
    std::vector<cv::Mat> patches;
    for (const auto& box_points : detected_boxes) {
        patches.push_back(postproc::ocr::get_warped_text_patch(image, box_points, pimpl_->config_.recognition_input_height));
    }

    // This is a simplified batching. A real implementation would group by width.
    for (const auto& patch : patches) {
        cv::Mat gray_patch;
        cv::cvtColor(patch, gray_patch, cv::COLOR_BGR2GRAY);
        gray_patch.convertTo(gray_patch, CV_32F, 1.0 / 127.5, -1.0);

        auto rec_shape = pimpl_->engine_recognizer_->get_input_shape(0);
        rec_shape[0] = 1; rec_shape[2] = gray_patch.rows; rec_shape[3] = gray_patch.cols;

        core::Tensor input_tensor_rec(rec_shape, core::DataType::kFLOAT);
        input_tensor_rec.copy_from_host(gray_patch.data);

        pimpl_->engine_recognizer_->setInputShape("input", rec_shape);
        auto output_tensors_rec = pimpl_->engine_recognizer_->infer({input_tensor_rec});

        OCRResult result;
        std::tie(result.text, result.confidence) = postproc::ocr::decode_recognition_output_ctc(
            output_tensors_rec[0], pimpl_->character_map_);

        if (result.confidence > 0.5) { // Filter low-confidence results
            // Find the corresponding box (this is inefficient, batching would fix this)
            // result.box_points = ...
            final_results.push_back(result);
        }
    }

    return final_results;
}

} // namespace xinfer::zoo::vision