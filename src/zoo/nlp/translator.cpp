#include <include/zoo/nlp/translator.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
// #include <xinfer/preproc/tokenizer.h>

namespace xinfer::zoo::nlp {

struct Translator::Impl {
    TranslatorConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    // std::unique_ptr<preproc::Tokenizer> tokenizer_;
};

Translator::Translator(const TranslatorConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Translator engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);
    // pimpl_->tokenizer_ = std::make_unique<preproc::Tokenizer>(pimpl_->config_.vocab_path);
}

Translator::~Translator() = default;
Translator::Translator(Translator&&) noexcept = default;
Translator& Translator::operator=(Translator&&) noexcept = default;

std::string Translator::predict(const std::string& text) {
    std::string full_translation = "";
    predict_stream(text, [&](const std::string& token_str) {
        full_translation += token_str;
    });
    return full_translation;
}

void Translator::predict_stream(const std::string& text,
                                std::function<void(const std::string&)> stream_callback) {
    if (!pimpl_) throw std::runtime_error("Translator is in a moved-from state.");

    // auto input_token_ids = pimpl_->tokenizer_->encode(text);
    // input_token_ids.resize(pimpl_->config_.max_input_length, 0);

    // auto decoder_token_ids = std::vector<int64_t>{pimpl_->tokenizer_->bos_token_id()};

    // for (int i = 0; i < pimpl_->config_.max_output_length; ++i) {
    //     core::Tensor encoder_input({1, (int64_t)input_token_ids.size()}, core::DataType::kINT32);
    //     encoder_input.copy_from_host(input_token_ids.data());

    //     core::Tensor decoder_input({1, (int64_t)decoder_token_ids.size()}, core::DataType::kINT32);
    //     decoder_input.copy_from_host(decoder_token_ids.data());

    //     auto output_tensors = pimpl_->engine_->infer({encoder_input, decoder_input});
    //     const core::Tensor& logits_tensor = output_tensors[0];

    //     std::vector<float> logits;
    //     logits.resize(logits_tensor.num_elements());
    //     logits_tensor.copy_to_host(logits.data());

    //     int next_token_id = 0; // Placeholder for sampling result

    //     // if (next_token_id == pimpl_->tokenizer_->eos_token_id()) {
    //     //     break;
    //     // }

    //     // decoder_token_ids.push_back(next_token_id);
    //     // std::string token_str = pimpl_->tokenizer_->decode({next_token_id});
    //     // stream_callback(token_str);
    // }
}

} // namespace xinfer::zoo::nlp