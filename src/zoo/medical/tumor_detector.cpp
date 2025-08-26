#include <include/zoo/medical/tumor_detector.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>
// #include <xinfer/postproc/detection3d.h> // Assumed to exist for 3D NMS

namespace xinfer::zoo::medical {

struct TumorDetector::Impl {
    TumorDetectorConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::vector<std::string> class_labels_;
};

TumorDetector::TumorDetector(const TumorDetectorConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Tumor detector engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    if (!pimpl_->config_.labels_path.empty()) {
        std::ifstream labels_file(pimpl_->config_.labels_path);
        if (!labels_file) throw std::runtime_error("Could not open labels file: " + pimpl_->config_.labels_path);
        std::string line;
        while (std::getline(labels_file, line)) {
            pimpl_->class_labels_.push_back(line);
        }
    }
}

TumorDetector::~TumorDetector() = default;
TumorDetector::TumorDetector(TumorDetector&&) noexcept = default;
TumorDetector& TumorDetector::operator=(TumorDetector&&) noexcept = default;

std::vector<Tumor> TumorDetector::predict(const std::vector<cv::Mat>& ct_scan_slices) {
    if (!pimpl_) throw std::runtime_error("TumorDetector is in a moved-from state.");

    if (ct_scan_slices.empty()) {
        return {};
    }

    // Pre-process: stack 2D slices into a 3D volume
    cv::Mat volume;
    cv::vconcat(ct_scan_slices, volume);

    cv::Mat resized_volume;
    cv::resize(volume, resized_volume, cv::Size(pimpl_->config_.input_width, pimpl_->config_.input_height * pimpl_->config_.input_depth));

    cv::Mat float_volume;
    resized_volume.reshape(0, pimpl_->config_.input_depth).convertTo(float_volume, CV_32F, 1.0/255.0);

    // Create input tensor
    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    input_tensor.copy_from_host(float_volume.data);

    // Run inference
    auto output_tensors = pimpl_->engine_->infer({input_tensor});

    // Post-process
    // This assumes the engine outputs clean boxes, scores, and labels.
    // A real implementation would call a dedicated 3D NMS post-processor.
    const core::Tensor& boxes_tensor = output_tensors[0];
    const core::Tensor& scores_tensor = output_tensors[1];
    const core::Tensor& labels_tensor = output_tensors[2];

    std::vector<float> h_boxes(boxes_tensor.num_elements());
    std::vector<float> h_scores(scores_tensor.num_elements());
    std::vector<int> h_labels(labels_tensor.num_elements());

    boxes_tensor.copy_to_host(h_boxes.data());
    scores_tensor.copy_to_host(h_scores.data());
    labels_tensor.copy_to_host(h_labels.data());

    std::vector<Tumor> results;
    int num_detections = scores_tensor.shape()[0];

    for (int i = 0; i < num_detections; ++i) {
        if (h_scores[i] < pimpl_->config_.confidence_threshold) {
            continue;
        }

        Tumor tumor;
        tumor.confidence = h_scores[i];
        tumor.class_id = h_labels[i];

        tumor.cx = h_boxes[i * 6 + 0];
        tumor.cy = h_boxes[i * 6 + 1];
        tumor.cz = h_boxes[i * 6 + 2];
        tumor.w = h_boxes[i * 6 + 3];
        tumor.h = h_boxes[i * 6 + 4];
        tumor.d = h_boxes[i * 6 + 5];

        if (!pimpl_->class_labels_.empty() && tumor.class_id < pimpl_->class_labels_.size()) {
            tumor.label = pimpl_->class_labels_[tumor.class_id];
        } else {
            tumor.label = "Tumor " + std::to_string(tumor.class_id);
        }
        results.push_back(tumor);
    }

    return results;
}

} // namespace xinfer::zoo::medical