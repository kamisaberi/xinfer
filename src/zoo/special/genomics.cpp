#include <xinfer/zoo/special/genomics.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
// We don't use image/audio preproc factories.
// DNA signal processing is unique (1D array math), implemented internally.
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/text/ocr_interface.h> // Reuse CTC Decoder

#include <iostream>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace xinfer::zoo::special {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct GenomicAnalyzer::Impl {
    GenomicsConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<postproc::IOcrPostprocessor> ctc_decoder_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    Impl(const GenomicsConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("GenomicAnalyzer: Failed to load model " + config_.model_path);
        }

        // 2. Setup Post-processor (Reuse OCR/CTC logic)
        // Basecalling is essentially "Text Recognition" on a 1D signal.
        ctc_decoder_ = postproc::create_ocr(config_.target);

        postproc::OcrConfig ctc_cfg;
        ctc_cfg.vocabulary = config_.vocabulary;
        ctc_cfg.blank_index = config_.blank_index;
        ctc_cfg.min_confidence = 0.0f; // Keep all bases
        ctc_decoder_->init(ctc_cfg);

        // 3. Pre-allocate Input [1, 1, WindowSize] or [1, WindowSize, 1] depending on model
        // Assuming [1, 1, WindowSize] (Channel-first 1D Convolution)
        input_tensor.resize({1, 1, (int64_t)config_.window_size}, core::DataType::kFLOAT);
    }

    // --- Signal Processing: Robust Scaling (Med-MAD) ---
    void normalize_signal(const float* src, float* dst, size_t size) {
        if (!config_.use_robust_scaling) {
            std::memcpy(dst, src, size * sizeof(float));
            return;
        }

        // 1. Calculate Median
        std::vector<float> buffer(src, src + size);
        size_t n = buffer.size();
        auto mid_iter = buffer.begin() + n / 2;
        std::nth_element(buffer.begin(), mid_iter, buffer.end());
        float median = *mid_iter;

        // 2. Calculate MAD (Median Absolute Deviation)
        for (auto& val : buffer) val = std::abs(val - median);
        std::nth_element(buffer.begin(), mid_iter, buffer.end());
        float mad = *mid_iter;

        // Avoid div by zero
        if (mad < 1e-6) mad = 1.0f;

        // 3. Scale
        // Formula: (x - median) / (mad * 1.4826)
        // 1.4826 scales MAD to approximate Standard Deviation for normal distributions
        float scale = 1.0f / (mad * 1.4826f);

        for (size_t i = 0; i < size; ++i) {
            dst[i] = (src[i] - median) * scale;
        }
    }
};

// =================================================================================
// Public API
// =================================================================================

GenomicAnalyzer::GenomicAnalyzer(const GenomicsConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

GenomicAnalyzer::~GenomicAnalyzer() = default;
GenomicAnalyzer::GenomicAnalyzer(GenomicAnalyzer&&) noexcept = default;
GenomicAnalyzer& GenomicAnalyzer::operator=(GenomicAnalyzer&&) noexcept = default;

GenomicResult GenomicAnalyzer::analyze_signal(const std::vector<float>& raw_signal) {
    if (!pimpl_) throw std::runtime_error("GenomicAnalyzer is null.");

    GenomicResult result;
    std::string full_sequence = "";

    // Chunking Logic
    // DNA reads are often 100k+ samples. Models take ~4096.
    size_t signal_len = raw_signal.size();
    size_t window = pimpl_->config_.window_size;
    size_t stride = window - pimpl_->config_.overlap;

    // Tensor pointer
    float* tensor_ptr = static_cast<float*>(pimpl_->input_tensor.data());

    for (size_t i = 0; i < signal_len; i += stride) {
        // Handle last chunk padding
        size_t current_chunk_size = std::min(window, signal_len - i);

        // 1. Normalize directly into Tensor memory
        // Copy chunk to tensor, pad with zeros if needed
        pimpl_->normalize_signal(raw_signal.data() + i, tensor_ptr, current_chunk_size);

        if (current_chunk_size < window) {
            std::fill(tensor_ptr + current_chunk_size, tensor_ptr + window, 0.0f);
        }

        // 2. Inference
        pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

        // 3. Decode (CTC)
        // Returns vector<string> (batch size 1)
        auto decoded_batch = pimpl_->ctc_decoder_->process(pimpl_->output_tensor);

        if (!decoded_batch.empty()) {
            std::string chunk_seq = decoded_batch[0];

            // 4. Stitching
            // Naive stitching: Just append.
            // Advanced stitching requires Smith-Waterman alignment on the overlap region.
            // For this example, we assume the overlap is handled by ignoring the first/last N bases,
            // or we just append (which is common for non-overlapping strides).
            full_sequence += chunk_seq;
        }

        if (current_chunk_size < window) break; // End of signal
    }

    result.sequence = full_sequence;

    // Placeholder quality scores (Real Q-Scores require probability analysis from CTC output)
    result.average_quality = 20.0f; // Q20 is standard OK quality

    return result;
}

} // namespace xinfer::zoo::special