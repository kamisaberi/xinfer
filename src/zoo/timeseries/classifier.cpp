#include <xinfer/zoo/timeseries/classifier.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
// We reuse the vision classification postproc because Softmax/TopK is generic
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/classification_interface.h>

#include <iostream>
#include <deque>
#include <fstream>
#include <algorithm>

namespace xinfer::zoo::timeseries {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct Classifier::Impl {
    TSClassifierConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<postproc::IClassificationPostprocessor> postproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    // Sliding Window Buffer
    std::deque<std::vector<float>> window_buffer_;

    // Labels
    std::vector<std::string> labels_;

    Impl(const TSClassifierConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("TSClassifier: Failed to load model " + config_.model_path);
        }

        // Validate Normalization Config
        if (config_.mean.size() != config_.num_features || config_.std.size() != config_.num_features) {
            XINFER_LOG_WARN("TSClassifier: Norm params missing or size mismatch. Defaulting to raw values.");
        }

        // 2. Setup Post-processor
        // We reuse the generic classification postproc (Softmax + TopK)
        postproc_ = postproc::create_classification(config_.target);

        postproc::ClassificationConfig cls_cfg;
        cls_cfg.top_k = config_.top_k;
        cls_cfg.apply_softmax = true;

        // Load labels
        if (!config_.labels_path.empty()) {
            load_labels(config_.labels_path);
            cls_cfg.labels = labels_;
        }

        postproc_->init(cls_cfg);

        // 3. Pre-allocate Input Tensor
        // Shape depends on layout: [1, T, F] or [1, F, T]
        if (config_.layout_time_first) {
            input_tensor.resize({1, (int64_t)config_.window_size, (int64_t)config_.num_features}, core::DataType::kFLOAT);
        } else {
            input_tensor.resize({1, (int64_t)config_.num_features, (int64_t)config_.window_size}, core::DataType::kFLOAT);
        }
    }

    void load_labels(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) return;
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            labels_.push_back(line);
        }
    }

    void add_point(const std::vector<float>& features) {
        if (features.size() != config_.num_features) {
            XINFER_LOG_ERROR("Input feature size mismatch.");
            return;
        }

        window_buffer_.push_back(features);
        if (window_buffer_.size() > config_.window_size) {
            window_buffer_.pop_front();
        }
    }

    // Convert Deque -> Tensor with Normalization
    void prepare_input() {
        float* ptr = static_cast<float*>(input_tensor.data());
        int idx = 0;
        bool use_norm = (!config_.mean.empty() && !config_.std.empty());

        if (config_.layout_time_first) {
            // Layout: [Time 0: Feat0, Feat1...], [Time 1...]
            for (const auto& step : window_buffer_) {
                for (int f = 0; f < config_.num_features; ++f) {
                    float val = step[f];
                    if (use_norm) val = (val - config_.mean[f]) / (config_.std[f] + 1e-6f);
                    ptr[idx++] = val;
                }
            }
        } else {
            // Layout: [Feat 0: T0, T1...], [Feat 1: T0, T1...]
            // We need to iterate features first, then time
            for (int f = 0; f < config_.num_features; ++f) {
                for (const auto& step : window_buffer_) {
                    float val = step[f];
                    if (use_norm) val = (val - config_.mean[f]) / (config_.std[f] + 1e-6f);
                    ptr[idx++] = val;
                }
            }
        }
    }
};

// =================================================================================
// Public API
// =================================================================================

Classifier::Classifier(const TSClassifierConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

Classifier::~Classifier() = default;
Classifier::Classifier(Classifier&&) noexcept = default;
Classifier& Classifier::operator=(Classifier&&) noexcept = default;

void Classifier::reset() {
    if (pimpl_) pimpl_->window_buffer_.clear();
}

bool Classifier::push(const std::vector<float>& features) {
    if (!pimpl_) return false;
    pimpl_->add_point(features);
    return (pimpl_->window_buffer_.size() >= pimpl_->config_.window_size);
}

std::vector<TSClassResult> Classifier::classify() {
    if (!pimpl_) throw std::runtime_error("TSClassifier is null.");

    // Check data availability
    if (pimpl_->window_buffer_.size() < pimpl_->config_.window_size) {
        // Pad with zeros or return empty?
        // Returning empty to signal "not ready"
        return {};
    }

    // 1. Prepare Input
    pimpl_->prepare_input();

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess
    // Reuse the Vision classification postprocessor logic
    auto batch_results = pimpl_->postproc_->process(pimpl_->output_tensor);

    std::vector<TSClassResult> results;
    if (!batch_results.empty()) {
        for (const auto& res : batch_results[0]) {
            TSClassResult ts_res;
            ts_res.id = res.id;
            ts_res.confidence = res.score;
            ts_res.label = res.label;
            results.push_back(ts_res);
        }
    }

    return results;
}

} // namespace xinfer::zoo::timeseries