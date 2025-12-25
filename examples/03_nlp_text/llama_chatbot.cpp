#include <iostream>
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>

using namespace xinfer;

int main() {
    Target target = Target::NVIDIA_TRT; // LLMs usually need GPU

    // 1. Setup Components
    auto engine = backends::BackendFactory::create(target);
    engine->load_model("llama3_8b_int4.engine");

    auto tokenizer = preproc::create_text_preprocessor(preproc::text::TokenizerType::SENTENCEPIECE);
    preproc::text::TokenizerConfig tok_cfg;
    tok_cfg.vocab_path = "tokenizer.model";
    tokenizer->init(tok_cfg);

    auto sampler = postproc::create_llm_sampler(target);
    postproc::LlmSampleConfig samp_cfg;
    samp_cfg.temperature = 0.7f;
    samp_cfg.top_p = 0.9f;
    sampler->init(samp_cfg);

    // 2. Chat Loop
    std::string prompt = "User: How do FPGAs accelerate AI?\nAI:";

    // Tokenize Prompt
    core::Tensor input_ids, attention_mask;
    tokenizer->process(prompt, input_ids, attention_mask);

    std::cout << prompt;

    // Autoregressive Generation
    for (int i = 0; i < 100; ++i) { // Gen 100 tokens
        core::Tensor logits;
        engine->predict({input_ids}, {logits}); // In real LLMs, use KV-Cache here

        // Sample next token
        std::vector<int> next_token = sampler->sample(logits, input_ids);

        // Decode and Print
        core::Tensor token_tensor({1}, core::DataType::kINT32);
        ((int*)token_tensor.data())[0] = next_token[0];

        std::string word = tokenizer->decode(token_tensor);
        std::cout << word << std::flush;

        // Append to history (Simulated)
        // input_ids.append(next_token[0]);
        if (next_token[0] == 2) break; // EOS
    }
    std::cout << std::endl;

    return 0;
}