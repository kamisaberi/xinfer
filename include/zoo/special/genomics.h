#pragma once

#include <string>
#include <vector>
#include <memory>

namespace xinfer::zoo::special {

    struct GenomicVariant {
        long long position;
        std::string reference_base;
        std::string alternate_base;
        float confidence;
    };

    struct VariantCallerConfig {
        std::string engine_path;
        std::string vocab_path; // For the DNA tokenizer
    };

    class VariantCaller {
    public:
        explicit VariantCaller(const VariantCallerConfig& config);
        ~VariantCaller();

        VariantCaller(const VariantCaller&) = delete;
        VariantCaller& operator=(const VariantCaller&) = delete;
        VariantCaller(VariantCaller&&) noexcept;
        VariantCaller& operator=(VariantCaller&&) noexcept;

        std::vector<GenomicVariant> predict(const std::string& dna_sequence);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::special

