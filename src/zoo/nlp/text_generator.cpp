#include <xinfer/zoo/nlp/text_generator.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/text/llm_sampler_interface.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>

namespace xinfer::zoo::nlp {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct TextGenerator::Impl {
    TextGenConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::ITextPreprocessor> tokenizer_;
    std::unique_ptr<postproc::ILlmSampler> sampler_;

    // Data Containers
    core::Tensor input_ids;      // [1, SeqLen]
    core::Tensor attention_mask; // [1, SeqLen]
    core::Tensor output_logits;  // [1, VocabSize]

    // State
    std::vector<int> history_ids_;
    int eos_token_id_ = 2; // Default Llama EOS
    int bos_token_id_ = 1; // Default Llama BOS

    Impl(const TextGenConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("TextGenerator: Failed to load model " + config_.model_path);
        }

        // 2. Setup Tokenizer
        // Llama/GPT uses SentencePiece or BPE
        auto type = config_.is_llama ? preproc::text::TokenizerType::SENTENCEPIECE
                                     : preproc::text::TokenizerType::GPT_BPE;

        tokenizer_ = preproc::create_text_preprocessor(type, config_.target);
        preproc::text::TokenizerConfig tok_cfg;
        tok_cfg.vocab_path = config_.tokenizer_path;
        tok_cfg.max_length = config_.context_window;
        tokenizer_->init(tok_cfg);

        // 3. Setup Sampler
        sampler_ = postproc::create_llm_sampler(config_.target);
        postproc::LlmSampleConfig samp_cfg;
        samp_cfg.temperature = config_.temperature;
        samp_cfg.top_k = config_.top_k;
        samp_cfg.top_p = config_.top_p;
        samp_cfg.repetition_penalty = config_.repetition_penalty;
        // Determine vocab size (heuristic or assume standard)
        // GPT-2: 50257, Llama: 32000, Llama3: 128256
        // Ideally queried from model metadata.
        samp_cfg.vocab_size = 32000;
        samp_cfg.eos_token_id = eos_token_id_;
        sampler_->init(samp_cfg);
    }

    void run_generation(const std::string& prompt, TextStreamCallback callback) {
        // 1. Prepare Input Text
        std::string full_prompt = prompt;
        if (!config_.system_prompt.empty()) {
            // Simple concatenation.
            // Real chat models need template: "<|system|>\n...<|user|>\n..."
            full_prompt = config_.system_prompt + "\n" + prompt;
        }

        // 2. Tokenize
        tokenizer_->process(full_prompt, input_ids, attention_mask);

        // Copy tensor data to history vector
        const int* ptr = static_cast<const int*>(input_ids.data());
        int seq_len = input_ids.shape()[1]; // Assuming batch 1

        // Filter padding (0) if present at end
        history_ids_.clear();
        if (config_.is_llama) history_ids_.push_back(bos_token_id_); // Prepend BOS

        for (int i = 0; i < seq_len; ++i) {
            if (ptr[i] != 0) history_ids_.push_back(ptr[i]);
        }

        // 3. Generation Loop
        for (int i = 0; i < config_.max_new_tokens; ++i) {
            if (history_ids_.size() >= (size_t)config_.context_window) break;

            // Prepare Input Tensor
            // (Naive re-feeding of full history. Stateful backends optimize this internally)
            size_t curr_len = history_ids_.size();
            input_ids.resize({1, (int64_t)curr_len}, core::DataType::kINT32);
            std::memcpy(input_ids.data(), history_ids_.data(), curr_len * sizeof(int));

            // Inference
            // Output logits usually [1, SeqLen, Vocab] or [1, Vocab] (last token only)
            // We assume backend handles extracting the last token logits if seq output.
            engine_->predict({input_ids}, {output_logits});

            // Sample
            std::vector<int> next_ids = sampler_->sample(output_logits, input_ids);
            int next_token = next_ids[0];

            // Decode for streaming
            core::Tensor single_tok({1}, core::DataType::kINT32);
            ((int*)single_tok.data())[0] = next_token;
            std::string text_chunk = tokenizer_->decode(single_tok);

            // Stream Callback
            if (callback) {
                if (!callback(text_chunk)) break; // User aborted
            }

            // Update State
            history_ids_.push_back(next_token);

            if (next_token == eos_token_id_) break;
        }
    }
};

// =================================================================================
// Public API
// =================================================================================

TextGenerator::TextGenerator(const TextGenConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

TextGenerator::~TextGenerator() = default;
TextGenerator::TextGenerator(TextGenerator&&) noexcept = default;
TextGenerator& TextGenerator::operator=(TextGenerator&&) noexcept = default;

void TextGenerator::reset() {
    if (pimpl_) pimpl_->history_ids_.clear();
}

std::string TextGenerator::generate(const std::string& prompt) {
    std::string result;
    if (pimpl_) {
        pimpl_->run_generation(prompt, [&](const std::string& chunk) {
            result += chunk;
            return true;
        });
    }
    return result;
}

void TextGenerator::generate_stream(const std::string& prompt, TextStreamCallback callback) {
    if (pimpl_) {
        pimpl_->run_generation(prompt, callback);
    }
}

} // namespace xinfer::zoo::nlp