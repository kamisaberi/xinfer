#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>

// #include <xinfer/preproc/tokenizer.h>

namespace xinfer::zoo::nlp {

    struct TextClassificationResult {
        int class_id;
        float confidence;
        std::string label;
    };

    struct ClassifierConfig {
        std::string engine_path;
        std::string labels_path = "";
        std::string vocab_path = "";
        int max_sequence_length = 128;
    };

    class Classifier {
    public:
        explicit Classifier(const ClassifierConfig& config);
        ~Classifier();

        Classifier(const Classifier&) = delete;
        Classifier& operator=(const Classifier&) = delete;
        Classifier(Classifier&&) noexcept;
        Classifier& operator=(Classifier&&) noexcept;

        std::vector<TextClassificationResult> predict(const std::string& text, int top_k = 5);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::nlp

