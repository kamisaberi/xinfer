#include <include/zoo/vision/instance_segmenter.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>
#include <include/postproc/instance_segmentation.h>

namespace xinfer::zoo::vision {

struct InstanceSegmenter::Impl {
    InstanceSegmenterConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;
    std::vector<std::string> class_labels_;
};

InstanceSegmenter::InstanceSegmenter(const InstanceSegmenterConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Instance segmentation engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(
        pimpl_->config_.input_width,
        pimpl_->config_.input_height,
        true
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

InstanceSegmenter::~InstanceSegmenter() = default;
InstanceSegmenter::InstanceSegmenter(InstanceSegmenter&&) noexcept = default;
InstanceSegmenter& InstanceSegmenter::operator=(InstanceSegmenter&&) noexcept = default;

std::vector<InstanceSegmentationResult> InstanceSegmenter::predict(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("InstanceSegmenter is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(image, input_tensor);

    auto output_tensors = pimpl_->engine_->infer({input_tensor});

    // YOLACT/MaskRCNN models typically have multiple outputs.
    // We assume the first is detections [boxes, scores, classes]
    // and the second is mask prototypes.
    const core::Tensor& raw_detections = output_tensors[0];
    const core::Tensor& raw_masks = output_tensors[1];

    std::vector<postproc::InstanceSegmentationResult> processed_results = postproc::instance_segmentation::process(
        raw_detections,
        raw_masks,
        pimpl_->config_.confidence_threshold,
        pimpl_->config_.nms_iou_threshold,
        pimpl_->config_.mask_threshold,
        pimpl_->config_.input_width,
        pimpl_->config_.input_height,
        image.cols,
        image.rows
    );

    std::vector<InstanceSegmentationResult> final_results;
    for (const auto& res : processed_results) {
        InstanceSegmentationResult final_res;
        final_res.class_id = res.class_id;
        final_res.confidence = res.confidence;
        final_res.bounding_box = res.bounding_box;
        final_res.mask = res.mask;
        if (!pimpl_->class_labels_.empty() && res.class_id < pimpl_->class_labels_.size()) {
            final_res.label = pimpl_->class_labels_[res.class_id];
        } else {
            final_res.label = "Class " + std::to_string(res.class_id);
        }
        final_results.push_back(final_res);
    }

    return final_results;
}

} // namespace xinfer::zoo::vision