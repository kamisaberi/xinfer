#include <include/zoo/vision/vehicle_identifier.h>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>

namespace xinfer::zoo::vision {

void load_labels(const std::string& path, std::vector<std::string>& labels) {
    if (path.empty()) return;
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Could not open labels file: " + path);
    std::string line;
    while (std::getline(file, line)) {
        labels.push_back(line);
    }
}

std::pair<int, float> get_top1(const float* logits, int num_classes) {
    auto max_it = std::max_element(logits, logits + num_classes);
    int max_idx = std::distance(logits, max_it);
    float max_val = *max_it;

    float sum_exp = 0.0f;
    for (int i = 0; i < num_classes; ++i) {
        sum_exp += expf(logits[i] - max_val);
    }
    float confidence = 1.0f / sum_exp;

    return {max_idx, confidence};
}

struct VehicleIdentifier::Impl {
    VehicleIdentifierConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;
    std::vector<std::string> type_labels_;
    std::vector<std::string> color_labels_;
    std::vector<std::string> make_labels_;
    std::vector<std::string> model_labels_;
};

VehicleIdentifier::VehicleIdentifier(const VehicleIdentifierConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Vehicle identification engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    if (pimpl_->engine_->get_num_outputs() != 4) {
        throw std::runtime_error("Vehicle identification engine must have exactly four outputs (type, color, make, model).");
    }

    pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(
        pimpl_->config_.input_width,
        pimpl_->config_.input_height,
        std::vector<float>{0.485f, 0.456f, 0.406f},
        std::vector<float>{0.229f, 0.224f, 0.225f}
    );

    load_labels(pimpl_->config_.type_labels_path, pimpl_->type_labels_);
    load_labels(pimpl_->config_.color_labels_path, pimpl_->color_labels_);
    load_labels(pimpl_->config_.make_labels_path, pimpl_->make_labels_);
    load_labels(pimpl_->config_.model_labels_path, pimpl_->model_labels_);
}

VehicleIdentifier::~VehicleIdentifier() = default;
VehicleIdentifier::VehicleIdentifier(VehicleIdentifier&&) noexcept = default;
VehicleIdentifier& VehicleIdentifier::operator=(VehicleIdentifier&&) noexcept = default;

VehicleAttributes VehicleIdentifier::predict(const cv::Mat& vehicle_image) {
    if (!pimpl_) throw std::runtime_error("VehicleIdentifier is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(vehicle_image, input_tensor);

    auto output_tensors = pimpl_->engine_->infer({input_tensor});

    const core::Tensor& type_logits_tensor = output_tensors[0];
    const core::Tensor& color_logits_tensor = output_tensors[1];
    const core::Tensor& make_logits_tensor = output_tensors[2];
    const core::Tensor& model_logits_tensor = output_tensors[3];

    std::vector<float> h_type(type_logits_tensor.num_elements());
    type_logits_tensor.copy_to_host(h_type.data());

    std::vector<float> h_color(color_logits_tensor.num_elements());
    color_logits_tensor.copy_to_host(h_color.data());

    std::vector<float> h_make(make_logits_tensor.num_elements());
    make_logits_tensor.copy_to_host(h_make.data());

    std::vector<float> h_model(model_logits_tensor.num_elements());
    model_logits_tensor.copy_to_host(h_model.data());

    VehicleAttributes attrs;

    auto top1_type = get_top1(h_type.data(), h_type.size());
    attrs.type_confidence = top1_type.second;
    attrs.type = (top1_type.first < pimpl_->type_labels_.size()) ? pimpl_->type_labels_[top1_type.first] : "Unknown";

    auto top1_color = get_top1(h_color.data(), h_color.size());
    attrs.color_confidence = top1_color.second;
    attrs.color = (top1_color.first < pimpl_->color_labels_.size()) ? pimpl_->color_labels_[top1_color.first] : "Unknown";

    auto top1_make = get_top1(h_make.data(), h_make.size());
    attrs.make_confidence = top1_make.second;
    attrs.make = (top1_make.first < pimpl_->make_labels_.size()) ? pimpl_->make_labels_[top1_make.first] : "Unknown";

    auto top1_model = get_top1(h_model.data(), h_model.size());
    attrs.model_confidence = top1_model.second;
    attrs.model = (top1_model.first < pimpl_->model_labels_.size()) ? pimpl_->model_labels_[top1_model.first] : "Unknown";

    return attrs;
}

} // namespace xinfer::zoo::vision