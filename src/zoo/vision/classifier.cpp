
#include <include/zoo/vision/classifier.h>
#include <stdexcept>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <functional>

#include <xinfer/core/engine.h>
#include <xinfer/preproc/image_processor.h>

namespace xinfer::zoo::vision {

// --- PIMPL Idiom Implementation ---
// The private implementation now only contains what it needs to RUN the model.
struct ImageClassifier::Impl {
    ClassifierConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;
    std::vector<std::string> class_labels_;

    // The post-processing logic is a perfect fit for a private helper method.
    std::vector<ClassificationResult> postprocess(const core::Tensor& logits_tensor, int top_k);
};

// --- Constructor Implementation ---
ImageClassifier::ImageClassifier(const ClassifierConfig& config)
    : pimpl_(new Impl{config})
{
    // Check if the engine file exists
    std::ifstream f(pimpl_->config_.engine_path.c_str());
    if (!f.good()) {
        throw std::runtime_error("TensorRT engine file not found: " + pimpl_->config_.engine_path +
            "\nPlease build the engine first using xinfer-cli or xinfer::builders::EngineBuilder.");
    }

    // 1. Load the pre-built, optimized engine. This is fast.
    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    // 2. Initialize the pre-processor with the specified parameters.
    pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(
        pimpl_->config_.input_width,
        pimpl_->config_.input_height,
        pimpl_->config_.mean,
        pimpl_->config_.std
    );

    // 3. Load class labels if a path is provided.
    if (!pimpl_->config_.labels_path.empty()) {
        std::ifstream labels_file(pimpl_->config_.labels_path);
        if (!labels_file) {
            throw std::runtime_error("Could not open labels file: " + pimpl_->config_.labels_path);
        }
        std::string line;
        while (std::getline(labels_file, line)) {
            pimpl_->class_labels_.push_back(line);
        }
    }
}

// Destructor and Move semantics must be defined after Impl is fully defined.
ImageClassifier::~ImageClassifier() = default;
ImageClassifier::ImageClassifier(ImageClassifier&&) noexcept = default;
ImageClassifier& ImageClassifier::operator=(ImageClassifier&&) noexcept = default;

// --- Public Method Implementation ---
std::vector<ClassificationResult> ImageClassifier::predict(const cv::Mat& image, int top_k) {
    if (!pimpl_ || !pimpl_->engine_ || !pimpl_->preprocessor_) {
        throw std::runtime_error("Classifier is not initialized or has been moved.");
    }

    // 1. Create a GPU tensor to hold the pre-processed input image
    auto input_shape = pimpl_->engine_->get_input_shape(0); // Get shape from the engine
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);

    // 2. Pre-process the image using the fused CUDA kernel
    pimpl_->preprocessor_->process(image, input_tensor);

    // 3. Run inference using the TensorRT engine
    // The engine's infer method returns a vector of output tensors.
    // For classification, we only expect one.
    auto output_tensors = pimpl_->engine_->infer({input_tensor});

    // 4. Post-process the raw output logits
    return pimpl_->postprocess(output_tensors[0], top_k);
}

// --- Post-processing Implementation ---
std::vector<ClassificationResult> ImageClassifier::Impl::postprocess(const core::Tensor& logits_tensor, int top_k) {
    // 1. Copy the raw logits from GPU to a CPU vector
    std::vector<float> logits(logits_tensor.num_elements());
    logits_tensor.copy_to_host(logits.data());

    // 2. Find the top_k results by sorting indices
    std::vector<int> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, 2, ...

    // Sort the indices based on the corresponding logit values in descending order
    std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });

    // 3. Apply Softmax only to the top_k values for efficiency
    float max_logit = logits[indices[0]];
    float sum_exp = 0.0f;
    std::vector<float> top_k_probs;
    for (int i = 0; i < top_k; ++i) {
        float prob = std::exp(logits[indices[i]] - max_logit);
        top_k_probs.push_back(prob);
        sum_exp += prob;
    }

    // 4. Populate the result vector
    std::vector<ClassificationResult> results;
    for (int i = 0; i < top_k; ++i) {
        int class_id = indices[i];
        float confidence = top_k_probs[i] / sum_exp;
        std::string label = class_labels_.empty() || class_id >= class_labels_.size() ?
                            "Class " + std::to_string(class_id) :
                            class_labels_[class_id];
        results.push_back({class_id, confidence, label});
    }

    return results;
}

} // namespace xinfer::zoo::vision