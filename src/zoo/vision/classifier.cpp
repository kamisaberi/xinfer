#include <xinfer/zoo/vision/classifier.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>

#include <fstream>
#include <iostream>

namespace xinfer::zoo::vision {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct ImageClassifier::Impl {
    ClassifierConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::IClassificationPostprocessor> postproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;
    std::vector<std::string> labels_;

    Impl(const ClassifierConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("ImageClassifier: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::RGB;
        pre_cfg.layout_nchw = true;

        // Configure Normalization (Crucial for Classification accuracy)
        pre_cfg.norm_params.mean = config_.mean;
        pre_cfg.norm_params.std = config_.std;
        // If user provided 0-255 mean values, we usually use scale=1.0.
        // If user wants 0-1 range first, they might set scale=1/255.
        // Here we assume standard ImageNet approach.
        pre_cfg.norm_params.scale_factor = 1.0f;

        preproc_->init(pre_cfg);

        // 3. Setup Post-processing
        postproc_ = postproc::create_classification(config_.target);

        postproc::ClassificationConfig post_cfg;
        post_cfg.top_k = config_.top_k;
        post_cfg.apply_softmax = config_.apply_softmax;
        postproc_->init(post_cfg);

        // 4. Load Labels
        if (!config_.labels_path.empty()) {
            load_labels(config_.labels_path);
        }
    }

    void load_labels(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            XINFER_LOG_WARN("Could not open labels file: " + path);
            return;
        }
        std::string line;
        while (std::getline(file, line)) {
            // Trim CR/LF
            if (!line.empty() && line.back() == '\r') line.pop_back();
            labels_.push_back(line);
        }
    }
};

// =================================================================================
// Public API
// =================================================================================

ImageClassifier::ImageClassifier(const ClassifierConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

ImageClassifier::~ImageClassifier() = default;
ImageClassifier::ImageClassifier(ImageClassifier&&) noexcept = default;
ImageClassifier& ImageClassifier::operator=(ImageClassifier&&) noexcept = default;

std::vector<ClassificationResult> ImageClassifier::classify(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("ImageClassifier is null.");

    // --- 1. Preprocess ---
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    frame.format = preproc::ImageFormat::BGR; // OpenCV default

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // --- 2. Inference ---
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // --- 3. Postprocess (Softmax + TopK) ---
    // Returns batch of results, we take index 0
    auto raw_results = pimpl_->postproc_->process(pimpl_->output_tensor);

    if (raw_results.empty()) return {};
    auto& top_k_raw = raw_results[0];

    // --- 4. Format Output ---
    std::vector<ClassificationResult> final_results;
    final_results.reserve(top_k_raw.size());

    for (const auto& res : top_k_raw) {
        ClassificationResult r;
        r.id = res.id;
        r.confidence = res.score;

        // Map ID to Label String
        if (res.id >= 0 && res.id < (int)pimpl_->labels_.size()) {
            r.label = pimpl_->labels_[res.id];
        } else {
            r.label = "Class " + std::to_string(res.id);
        }

        final_results.push_back(r);
    }

    return final_results;
}

} // namespace xinfer::zoo::vision