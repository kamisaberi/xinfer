#include <include/zoo/nlp/question_answering.h>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

#include <include/core/engine.h>
// #include <xinfer/preproc/tokenizer.h>

namespace xinfer::zoo::nlp {

struct QuestionAnswering::Impl {
    QAConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    // std::unique_ptr<preproc::Tokenizer> tokenizer_;
};

QuestionAnswering::QuestionAnswering(const QAConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Question Answering engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);
    // pimpl_->tokenizer_ = std::make_unique<preproc::Tokenizer>(pimpl_->config_.vocab_path);
}

QuestionAnswering::~QuestionAnswering() = default;
QuestionAnswering::QuestionAnswering(QuestionAnswering&&) noexcept = default;
QuestionAnswering& QuestionAnswering::operator=(QuestionAnswering&&) noexcept = default;

QAResult QuestionAnswering::predict(const std::string& question, const std::string& context) {
    if (!pimpl_) throw std::runtime_error("QuestionAnswering is in a moved-from state.");

    // auto tokenization_result = pimpl_->tokenizer_->encode_qa(question, context);
    // auto& token_ids = tokenization_result.ids;
    // token_ids.resize(pimpl_->config_.max_sequence_length, 0);

    // auto input_shape = pimpl_->engine_->get_input_shape(0);
    // input_shape[0] = 1;
    // input_shape[1] = pimpl_->config_.max_sequence_length;

    // core::Tensor input_tensor(input_shape, core::DataType::kINT32);
    // input_tensor.copy_from_host(token_ids.data());

    // auto output_tensors = pimpl_->engine_->infer({input_tensor});
    // const core::Tensor& start_logits_tensor = output_tensors[0];
    // const core::Tensor& end_logits_tensor = output_tensors[1];

    // std::vector<float> start_logits(start_logits_tensor.num_elements());
    // start_logits_tensor.copy_to_host(start_logits.data());

    // std::vector<float> end_logits(end_logits_tensor.num_elements());
    // end_logits_tensor.copy_to_host(end_logits.data());

    // auto start_it = std::max_element(start_logits.begin(), start_logits.end());
    // int start_idx = std::distance(start_logits.begin(), start_it);

    // auto end_it = std::max_element(end_logits.begin(), end_logits.end());
    // int end_idx = std::distance(end_logits.begin(), end_it);

    QAResult result;
    // if (start_idx <= end_idx) {
    //     auto answer_tokens = std::vector<int64_t>(token_ids.begin() + start_idx, token_ids.begin() + end_idx + 1);
    //     result.answer = pimpl_->tokenizer_->decode(answer_tokens);

    //     // Simplified confidence score
    //     float start_max = *start_it;
    //     float end_max = *end_it;
    //     float start_sum_exp = 0.0f, end_sum_exp = 0.0f;
    //     for(float logit : start_logits) start_sum_exp += expf(logit - start_max);
    //     for(float logit : end_logits) end_sum_exp += expf(logit - end_max);
    //     result.score = (1.0f / start_sum_exp) * (1.0f / end_sum_exp);

    //     result.start_pos = 0; // Placeholder for character-level offsets
    //     result.end_pos = 0;   // Placeholder
    // } else {
    //     result.answer = "";
    //     result.score = 0.0f;
    //     result.start_pos = -1;
    //     result.end_pos = -1;
    // }

    return result;
}

} // namespace xinfer::zoo::nlp