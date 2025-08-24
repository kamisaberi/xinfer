#pragma once

#include <include/core/tensor.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <tuple>

namespace xinfer::postproc::ocr {

    // Decodes the two-part output of a CRAFT text detection model
    std::vector<std::vector<cv::Point2f>> decode_detection_output_craft(
        const core::Tensor& region_scores, const core::Tensor& affinity_scores,
        float text_threshold, float box_threshold,
        float scale_w, float scale_h);

    // Warps a detected polygonal text region into a straight rectangular patch
    cv::Mat get_warped_text_patch(const cv::Mat& image, const std::vector<cv::Point2f>& box, int target_height);

    // Decodes the output of a CRNN-style recognition model using CTC greedy decoding on the GPU
    std::tuple<std::string, float> decode_recognition_output_ctc(
        const core::Tensor& logits, const std::vector<char>& character_map);

} // namespace xinfer::postproc::ocr

