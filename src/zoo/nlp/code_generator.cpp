#include <xinfer/zoo/nlp/code_generator.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/text/llm_sampler_interface.h>

#include <iostream>
#include <algorithm>

namespace xinfer::zoo::nlp {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct CodeGenerator::Impl {
    CodeGenConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::ITextPreprocessor> tokenizer_;
    std::unique_ptr<postproc::ILlmSampler> sampler_;

    // Data Containers
    core::Tensor input_ids;      // [1, SeqLen]
    core::Tensor attention_mask; // [1, SeqLen]
    core::Tensor output_logits;  // [1, VocabSize] (for last token)

    // Internal State
    std::vector<int> history_ids_; // Keeps track of full conversation for rep penalty

    Impl(const CodeGenConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("CodeGenerator: Failed to load model " + config_.model_path);
        }

        // 2. Setup Tokenizer
        // CodeLlama/StarCoder usually use BPE / SentencePiece
        auto type = config_.is_bpe ? preproc::text::TokenizerType::SENTENCEPIECE
                                   : preproc::text::TokenizerType::BERT_WORDPIECE;

        tokenizer_ = preproc::create_text_preprocessor(type, config_.target);

        preproc::text::TokenizerConfig tok_cfg;
        tok_cfg.vocab_path = config_.tokenizer_path;
        // The tokenizer needs to know max length to handle padding if necessary,
        // though for generation we usually dynamic shape.
        tok_cfg.max_length = config_.context_window;
        tokenizer_->init(tok_cfg);

        // 3. Setup Sampler
        sampler_ = postproc::create_llm_sampler(config_.target);

        postproc::LlmSampleConfig samp_cfg;
        samp_cfg.temperature = config_.temperature;
        samp_cfg.top_p = config_.top_p;
        samp_cfg.repetition_penalty = config_.repetition_penalty;
        // Note: Vocab size needs to be known.
        // We assume 32000 (Llama standard) or inspect model output shape after first run.
        samp_cfg.vocab_size = 32000;

        sampler_->init(samp_cfg);
    }

    void run_generation(const std::string& prompt, StreamCallback callback) {
        // 1. Tokenize Prompt
        // Note: We reuse input_ids tensor.
        tokenizer_->process(prompt, input_ids, attention_mask);

        // Extract initial IDs from tensor to history
        const int* ptr = static_cast<const int*>(input_ids.data());
        int seq_len = input_ids.shape()[1]; // Assuming batch 1

        // Find actual length (ignore padding zeros at end if tokenizer padded)
        // This logic depends on tokenizer impl, let's assume it fills & pads.
        int real_len = 0;
        for(; real_len < seq_len; ++real_len) {
            if (ptr[real_len] == 0) break; // Assuming 0 is PAD/EOS for simplicity
        }

        history_ids_.assign(ptr, ptr + real_len);

        // Core Generation Loop
        for (int i = 0; i < config_.max_new_tokens; ++i) {
            // Check Context Window
            if (history_ids_.size() >= (size_t)config_.context_window) {
                break;
            }

            // Update Input Tensor
            // For stateful models (KV cache enabled in backend), we might only send the *last* token
            // after the first pass. However, standard ONNX execution usually requires full sequence
            // unless the backend handles the KV caching explicitly.
            //
            // Here, we assume the Naive approach (send full history) for generic compatibility,
            // or the Backend is smart enough to detect incremental updates.

            size_t current_len = history_ids_.size();
            input_ids.resize({1, (int64_t)current_len}, core::DataType::kINT32);
            std::memcpy(input_ids.data(), history_ids_.data(), current_len * sizeof(int));

            // Inference
            engine_->predict({input_ids}, {output_logits});

            // Sampling
            // We pass the full history tensor for repetition penalty calculation
            std::vector<int> next_token_vec = sampler_->sample(output_logits, input_ids);
            int next_token = next_token_vec[0];

            // Decode Token to String
            core::Tensor single_tok({1}, core::DataType::kINT32);
            ((int*)single_tok.data())[0] = next_token;
            std::string token_str = tokenizer_->decode(single_tok);

            // Stream Output
            bool continue_gen = true;
            if (callback) {
                continue_gen = callback(token_str);
            }

            // Update History
            history_ids_.push_back(next_token);

            // Stop Conditions
            if (!continue_gen) break; // Callback stop
            if (next_token == 2) break; // EOS (Llama standard)

            // Check Stop Words
            // (Simplified: checks if current token matches a stop word)
            // Real implementation needs a sliding window on string buffer
            bool stop_hit = false;
            for (const auto& sw : config_.stop_words) {
                if (token_str == sw) { stop_hit = true; break; }
            }
            if (stop_hit) break;
        }
    }
};

// =================================================================================
// Public API
// =================================================================================

CodeGenerator::CodeGenerator(const CodeGenConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

CodeGenerator::~CodeGenerator() = default;
CodeGenerator::CodeGenerator(CodeGenerator&&) noexcept = default;
CodeGenerator& CodeGenerator::operator=(CodeGenerator&&) noexcept = default;

void CodeGenerator::reset_context() {
    if (pimpl_) pimpl_->history_ids_.clear();
    // Also notify backend to clear KV cache if API supports it
}

std::string CodeGenerator::generate(const std::string& prompt) {
    std::string full_output = "";
    if (pimpl_) {
        pimpl_->run_generation(prompt, [&](const std::string& token) {
            full_output += token;
            return true; // Continue
        });
    }
    return full_output;
}

void CodeGenerator::generate_stream(const std::string& prompt, StreamCallback callback) {
    if (pimpl_) {
        pimpl_->run_generation(prompt, callback);
    }
}

} // namespace xinfer::zoo::nlp