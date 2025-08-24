#include <include/zoo/nlp/text_generator.h>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <string>

#include <include/core/engine.h>
// #include <xinfer/preproc/tokenizer.h>

namespace xinfer::zoo::nlp {

struct TextGenerator::Impl {
    TextGeneratorConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    // std::unique_ptr<preproc::Tokenizer> tokenizer_;
};

TextGenerator::TextGenerator(const TextGeneratorConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Text generator engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);
    // pimpl_->tokenizer_ = std::make_unique<preproc::Tokenizer>(pimpl_->config_.vocab_path);
}

TextGenerator::~TextGenerator() = default;
TextGenerator::TextGenerator(TextGenerator&&) noexcept = default;
TextGenerator& TextGenerator::operator=(TextGenerator&&) noexcept = default;

std::string TextGenerator::predict(const std::string& prompt) {
    std::string full_text = "";
    predict_stream(prompt, [&](const std::string& token_str) {
        full_text += token_str;
    });
    return full_text;
}

void TextGenerator::predict_stream(const std::string& prompt,
                                     std::function<void(const std::string&)> stream_callback) {
    if (!pimpl_) throw std::runtime_error("TextGenerator is in a moved-from state.");

    // auto token_ids = pimpl_->tokenizer_->encode(prompt);

    // for (int i = 0; i < pimpl_->config_.max_new_tokens; ++i) {
    //     core::Tensor input_tensor({1, (int64_t)token_ids.size()}, core::DataType::kINT32);
    //     input_tensor.copy_from_host(token_ids.data());

    //     auto output_tensors = pimpl_->engine_->infer({input_tensor});
    //     const core::Tensor& logits_tensor = output_tensors[0];

    //     std::vector<float> logits;
    //     logits.resize(logits_tensor.num_elements());
    //     logits_tensor.copy_to_host(logits.data());

    //     // --- Add sampling logic here (top-p, temperature) ---
    //     int next_token_id = 0; // Placeholder for sampling result

    //     // if (next_token_id == pimpl_->tokenizer_->eos_token_id()) {
    //     //     break;
    //     // }

    //     // token_ids.push_back(next_token_id);
    //     // std::string token_str = pimpl_->tokenizer_->decode({next_token_id});
    //     // stream_callback(token_str);
    // }
}

} // namespace xinfer::zoo::nlp