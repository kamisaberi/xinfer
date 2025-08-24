#include <include/zoo/vision/image_similarity.h>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <cmath>
#include <numeric>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>

namespace xinfer::zoo::vision {

struct ImageSimilarity::Impl {
    ImageSimilarityConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;
};

ImageSimilarity::ImageSimilarity(const ImageSimilarityConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Image similarity engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(
        pimpl_->config_.input_width,
        pimpl_->config_.input_height,
        std::vector<float>{0.485f, 0.456f, 0.406f},
        std::vector<float>{0.229f, 0.224f, 0.225f}
    );
}

ImageSimilarity::~ImageSimilarity() = default;
ImageSimilarity::ImageSimilarity(ImageSimilarity&&) noexcept = default;
ImageSimilarity& ImageSimilarity::operator=(ImageSimilarity&&) noexcept = default;

ImageEmbedding ImageSimilarity::predict(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("ImageSimilarity is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(image, input_tensor);

    auto output_tensors = pimpl_->engine_->infer({input_tensor});
    const core::Tensor& embedding_tensor = output_tensors[0];

    ImageEmbedding embedding(embedding_tensor.num_elements());
    embedding_tensor.copy_to_host(embedding.data());

    float norm = 0.0f;
    for (float val : embedding) {
        norm += val * val;
    }
    norm = std::sqrt(norm);

    if (norm > 1e-6) {
        for (float& val : embedding) {
            val /= norm;
        }
    }

    return embedding;
}

float ImageSimilarity::compare(const ImageEmbedding& emb1, const ImageEmbedding& emb2) {
    if (emb1.size() != emb2.size() || emb1.empty()) {
        return 0.0f;
    }
    float dot_product = std::inner_product(emb1.begin(), emb1.end(), emb2.begin(), 0.0f);

    return std::max(0.0f, std::min(1.0f, dot_product));
}

} // namespace xinfer::zoo::vision