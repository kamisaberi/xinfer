
#include <include/zoo/vision/detector.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>
// This is the CRITICAL include that provides your F1-car NMS kernel
#include <include/postproc/detection.h>
// This is a NEW, REQUIRED helper for decoding the model's output
#include <include/postproc/yolo_decoder.h> // You will need to create this file

namespace xinfer::zoo::vision {

// --- PIMPL Idiom Implementation ---
struct ObjectDetector::Impl {
    DetectorConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;
    std::vector<std::string> class_labels_;

    // NEW: Buffers to hold the intermediate decoded results on the GPU
    core::Tensor decoded_boxes_gpu;
    core::Tensor decoded_scores_gpu;
    core::Tensor decoded_classes_gpu;
};

// --- Constructor Implementation ---
ObjectDetector::ObjectDetector(const DetectorConfig& config)
    : pimpl_(new Impl{config})
{
    std::ifstream f(pimpl_->config_.engine_path.c_str());
    if (!f.good()) {
        throw std::runtime_error("TensorRT engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    // The ImageProcessor for YOLO needs to support letterbox padding.
    // We assume it has a constructor for this.
    pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(
        pimpl_->config_.input_width,
        pimpl_->config_.input_height,
        true // Enable letterbox padding
    );

    if (!pimpl_->config_.labels_path.empty()) {
        std::ifstream labels_file(pimpl_->config_.labels_path);
        if (!labels_file) throw std::runtime_error("Could not open labels file: " + pimpl_->config_.labels_path);
        std::string line;
        while (std::getline(labels_file, line)) {
            pimpl_->class_labels_.push_back(line);
        }
    }
}

ObjectDetector::~ObjectDetector() = default;
ObjectDetector::ObjectDetector(ObjectDetector&&) noexcept = default;
ObjectDetector& ObjectDetector::operator=(ObjectDetector&&) noexcept = default;


// --- Public Method Implementation ---
std::vector<BoundingBox> ObjectDetector::predict(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("ObjectDetector is in a moved-from state.");

    // --- STEP 1: Pre-process ---
    // The preprocessor handles letterbox padding and normalization in a single fused kernel.
    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(image, input_tensor);

    // --- STEP 2: Inference ---
    // This returns a single raw output tensor for a YOLO-style model
    auto output_tensors = pimpl_->engine_->infer({input_tensor});
    const core::Tensor& raw_output = output_tensors[0];

    // --- STEP 3: Decode (on GPU) ---
    // This is the CRITICAL new step. We call a dedicated CUDA kernel to parse the
    // raw YOLO output tensor into separate tensors for boxes, scores, and class IDs.
    // This avoids a slow GPU->CPU download of a huge tensor.
    postproc::yolo::decode(
        raw_output,
        pimpl_->config_.confidence_threshold,
        pimpl_->decoded_boxes_gpu,
        pimpl_->decoded_scores_gpu,
        pimpl_->decoded_classes_gpu
    );

    // --- STEP 4: Non-Maximum Suppression (on GPU) ---
    // Now, we run our hyper-optimized NMS kernel on the clean, decoded tensors.
    std::vector<int> final_indices = postproc::detection::nms(
        pimpl_->decoded_boxes_gpu,
        pimpl_->decoded_scores_gpu,
        pimpl_->config_.nms_iou_threshold
    );

    // --- STEP 5: Post-process and return results ---
    // The final indices are a small list, so this CPU download is very fast.
    std::vector<BoundingBox> final_boxes;
    if (final_indices.empty()) {
        return final_boxes;
    }

    // Download the data only for the boxes that survived NMS
    std::vector<float> h_boxes = pimpl_->decoded_boxes_gpu.get_items(final_indices);
    std::vector<float> h_scores = pimpl_->decoded_scores_gpu.get_items(final_indices);
    std::vector<int> h_classes = pimpl_->decoded_classes_gpu.get_items(final_indices);

    // Scale coordinates back to original image size
    float scale = std::min(
        (float)pimpl_->config_.input_width / image.cols,
        (float)pimpl_->config_.input_height / image.rows
    );

    for (size_t i = 0; i < final_indices.size(); ++i) {
        BoundingBox box;
        box.x1 = h_boxes[i*4 + 0] / scale;
        box.y1 = h_boxes[i*4 + 1] / scale;
        box.x2 = h_boxes[i*4 + 2] / scale;
        box.y2 = h_boxes[i*4 + 3] / scale;
        box.confidence = h_scores[i];
        box.class_id = h_classes[i];
        if (box.class_id < pimpl_->class_labels_.size()) {
            box.label = pimpl_->class_labels_[box.class_id];
        }
        final_boxes.push_back(box);
    }

    return final_boxes;
}

} // namespace xinfer::zoo::vision