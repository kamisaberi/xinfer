#include <include/zoo/nlp/keyword_extractor.h>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <algorithm>

#include <include/core/engine.h>
// #include <xinfer/preproc/tokenizer.h>

namespace xinfer::zoo::nlp {

struct KeywordExtractor::Impl {
    KeywordExtractorConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    // std::unique_ptr<preproc::Tokenizer> tokenizer_;
};

KeywordExtractor::KeywordExtractor(const KeywordExtractorConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Keyword extractor engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);
    // pimpl_->tokenizer_ = std::make_unique<preproc::Tokenizer>(pimpl_->config_.vocab_path);
}

KeywordExtractor::~KeywordExtractor() = default;
KeywordExtractor::KeywordExtractor(KeywordExtractor&&) noexcept = default;
KeywordExtractor& KeywordExtractor::operator=(KeywordExtractor&&) noexcept = default;

std::vector<Keyword> KeywordExtractor::predict(const std::string& text, int top_k) {
    if (!pimpl_) throw std::runtime_error("KeywordExtractor is in a moved-from state.");

    // auto token_ids = pimpl_->tokenizer_->encode(text);
    // token_ids.resize(pimpl_->config_.max_sequence_length, 0);

    // auto input_shape = pimpl_->engine_->get_input_shape(0);
    // input_shape[0] = 1;
    // input_shape[1] = pimpl_->config_.max_sequence_length;

    // core::Tensor input_tensor(input_shape, core::DataType::kINT32);
    // input_tensor.copy_from_host(token_ids.data());

    // auto output_tensors = pimpl_->engine_->infer({input_tensor});
    // const core::Tensor& logits_tensor = output_tensors[0];

    // std::vector<float> logits(logits_tensor.num_elements());
    // logits_tensor.copy_to_host(logits.data());

    std::vector<Keyword> results;

    // A full implementation would involve complex logic to group subword tokens
    // and map their aggregated scores back to words in the original text.
    // This is a placeholder for that complex post-processing.

    return results;
}

} // namespace xinfer::zoo::nlp