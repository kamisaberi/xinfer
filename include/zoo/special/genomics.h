#pragma once

#include <string>
#include <vector>
#include <memory>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::special {

    /**
     * @brief Result of Genomic Analysis.
     */
    struct GenomicResult {
        std::string sequence;      // The decoded DNA sequence (A, C, G, T)
        std::vector<float> q_scores; // Quality scores (Phred scale) per base
        float average_quality;
    };

    struct GenomicsConfig {
        // Hardware Target (FPGAs are very popular for Genomics e.g., Dragen)
        xinfer::Target target = xinfer::Target::INTEL_FPGA;

        // Model Path (e.g., bonito_dna.onnx, guppy_lite.rknn)
        std::string model_path;

        // Input Specs
        // Models typically take a fixed window of raw signal (float)
        int window_size = 4096;
        int overlap = 512;      // Overlap between chunks to prevent edge data loss

        // Normalization (Med-MAD or Standard)
        bool use_robust_scaling = true; // (x - median) / MAD

        // Decoding
        // Vocabulary is usually "N A C G T" (N=Blank)
        std::string vocabulary = "NACGT";
        int blank_index = 0;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class GenomicAnalyzer {
    public:
        explicit GenomicAnalyzer(const GenomicsConfig& config);
        ~GenomicAnalyzer();

        // Move semantics
        GenomicAnalyzer(GenomicAnalyzer&&) noexcept;
        GenomicAnalyzer& operator=(GenomicAnalyzer&&) noexcept;
        GenomicAnalyzer(const GenomicAnalyzer&) = delete;
        GenomicAnalyzer& operator=(const GenomicAnalyzer&) = delete;

        /**
         * @brief Analyze a raw signal trace (Basecalling).
         *
         * Pipeline:
         * 1. Normalize Signal (pico-Amperes handling).
         * 2. Chunking (Split long read into model-sized windows).
         * 3. Inference (RNN/Transformer).
         * 4. CTC Decoding (Logits -> ACGT).
         * 5. Stitching.
         *
         * @param raw_signal Raw float data from sequencer.
         * @return Decoded DNA sequence.
         */
        GenomicResult analyze_signal(const std::vector<float>& raw_signal);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::special