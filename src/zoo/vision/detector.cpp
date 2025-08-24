// src/zoo/vision/detector.cpp

#include <include/zoo/vision/detector.h>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <numeric>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>
#include <include/postproc/detection.h>
#include <include/postproc/yolo_decoder.h> // Include your new decoder

namespace xinfer::zoo::vision {

// Maximum number of boxes to consider after the decode step
// This affects how large we size our intermediate GPU buffers.
const int MAX_DECODED_BOXES = 4096;

struct ObjectDetector::Impl {
    DetectorConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;
    std::vector<std::string> class_labels_;

    // Intermediate GPU buffers for the post-processing pipeline
    core::Tensor decoded_boxes_gpu;
    core::Tensor decoded_scores_gpu;
    core::Tensor decoded_classes_gpu;

    // Host-side buffers for the final results after NMS
    std::vector<float> h_boxes;
    std::vector<float> h_scores;
    std::vector<int> h_classes;

    Impl(const DetectorConfig& config) : config_(config) {
        // Pre-allocate the intermediate buffers once during initialization
        decoded_boxes_gpu = core::Tensor({MAX_DECODED_BOXES, 4}, core::DataType::kFLOAT);
        decoded_scores_gpu = core::Tensor({MAX_DECODED_BOXES}, core::DataType::kFLOAT);
        decoded_classes_gpu = core::Tensor({MAX_DECODED_BOXES}, core::DataType::kINT32);

        h_boxes.resize(MAX_DECODED_BOXES * 4);
        h_scores.resize(MAX_DECODED_BOXES);
        h_classes.resize(MAX_DECODED_BOXES);
    }
};

ObjectDetector::ObjectDetector(const DetectorConfig& config) : pimpl_(new Impl(config)) {
    // ... (Constructor logic to load engine, preprocessor, labels is the same) ...
}
// ... (Destructor and move semantics are the same) ...

std::vector<BoundingBox> ObjectDetector::predict(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("ObjectDetector is in a moved-from state.");

    // --- STEP 1: Pre-process ---
    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(image, input_tensor);

    // --- STEP 2: Inference ---
    auto output_tensors = pimpl_->engine_->infer({input_tensor});
    const core::Tensor& raw_output = output_tensors[0];

    // --- STEP 3: Decode on GPU ---
    postproc::yolo::decode(raw_output, pimpl_->config_.confidence_threshold,
                           pimpl_->decoded_boxes_gpu, pimpl_->decoded_scores_gpu, pimpl_->decoded_classes_gpu);

    // --- STEP 4: Non-Maximum Suppression on GPU ---
    // The NMS function returns a small vector of indices of the winning boxes.
    std::vector<int> final_indices = postproc::detection::nms(
        pimpl_->decoded_boxes_gpu, pimpl_->decoded_scores_gpu,
        pimpl_->config_.nms_iou_threshold);

    // --- STEP 5: Copy final results to CPU and format ---
    std::vector<BoundingBox> final_boxes;
    if (final_indices.empty()) return final_boxes;

    // We can't directly index the GPU tensors, so we download the full decoded lists
    // and index them on the CPU. A more advanced version might have a custom "gather" kernel.
    pimpl_->decoded_boxes_gpu.copy_to_host(pimpl_->h_boxes.data());
    pimpl_->decoded_scores_gpu.copy_to_host(pimpl_->h_scores.data());
    pimpl_->decoded_classes_gpu.copy_to_host(pimpl_->h_classes.data());

    float scale_x = (float)image.cols / pimpl_->config_.input_width;
    float scale_y = (float)image.rows / pimpl_->config_.input_height;

    for (int idx : final_indices) {
        BoundingBox box;
        box.x1 = pimpl_->h_boxes[idx * 4 + 0] * scale_x;
        box.y1 = pimpl_->h_boxes[idx * 4 + 1] * scale_y;
        box.x2 = pimpl_->h_boxes[idx * 4 + 2] * scale_x;
        box.y2 = pimpl_->h_boxes[idx * 4 + 3] * scale_y;
        box.confidence = pimpl_->h_scores[idx];
        box.class_id = pimpl_->h_classes[idx];
        if (box.class_id < pimpl_->class_labels_.size()) {
            box.label = pimpl_->class_labels_[box.class_id];
        }
        final_boxes.push_back(box);
    }

    return final_boxes;
}

} // namespace xinfer::zoo::vision