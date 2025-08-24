#pragma once

#include <string>
#include <vector>
#include <memory>

namespace xinfer::zoo::nlp {

    struct NamedEntity {
        std::string text;
        std::string label;
        float score;
        int start_pos;
        int end_pos;
    };

    struct NERConfig {
        std::string engine_path;
        std::string labels_path = "";
        std::string vocab_path = "";
        int max_sequence_length = 256;
    };

    class NER {
    public:
        explicit NER(const NERConfig& config);
        ~NER();

        NER(const NER&) = delete;
        NER& operator=(const NER&) = delete;
        NER(NER&&) noexcept;
        NER& operator=(NER&&) noexcept;

        std::vector<NamedEntity> predict(const std::string& text);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::nlp

