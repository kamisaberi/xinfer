#include <include/zoo/nlp/ner.h>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>

#include <include/core/engine.h>
// #include <xinfer/preproc/tokenizer.h>

namespace xinfer::zoo::nlp {

struct NER::Impl {
    NERConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    // std::unique_ptr<preproc::Tokenizer> tokenizer_;
    std::vector<std::string> entity_labels_;
};

NER::NER(const NERConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("NER engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);
    // pimpl_->tokenizer_ = std::make_unique<preproc::Tokenizer>(pimpl_->config_.vocab_path);

    if (!pimpl_->config_.labels_path.empty()) {
        std::ifstream labels_file(pimpl_->config_.labels_path);
        if (!labels_file) throw std::runtime_error("Could not open labels file: " + pimpl_->config_.labels_path);
        std::string line;
        while (std::getline(labels_file, line)) {
            pimpl_->entity_labels_.push_back(line);
        }
    }
}

NER::~NER() = default;
NER::NER(NER&&) noexcept = default;
NER& NER::operator=(NER&&) noexcept = default;

std::vector<NamedEntity> NER::predict(const std::string& text) {
    if (!pimpl_) throw std::runtime_error("NER is in a moved-from state.");

    // auto tokenization_result = pimpl_->tokenizer_->encode_with_offsets(text);
    // auto& token_ids = tokenization_result.ids;
    // auto& offsets = tokenization_result.offsets;
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

    // auto logits_shape = logits_tensor.shape();
    // int seq_len = logits_shape[1];
    // int num_classes = logits_shape[2];

    std::vector<NamedEntity> results;

    // A full implementation requires complex logic to find the argmax for each token,
    // apply softmax, and then group consecutive B- (begin) and I- (inside) tokens
    // into single entities. This is a placeholder for that logic.

    return results;
}

} // namespace xinfer::zoo::nlp