#pragma once

#include <xinfer/postproc/text/ocr_interface.h>

namespace xinfer::postproc {

/**
 * @brief CPU CTC Greedy Decoder
 * 
 * Algorithm:
 * 1. ArgMax: Find the class with highest probability at each time step.
 * 2. Merge: Collapse repeated characters (e.g. "AA" -> "A").
 * 3. Filter: Remove Blank tokens.
 * 4. Map: Convert remaining indices to characters using vocabulary.
 */
class CtcDecoder : public IOcrPostprocessor {
public:
    CtcDecoder();
    ~CtcDecoder() override;

    void init(const OcrConfig& config) override;

    std::vector<std::string> process(const core::Tensor& logits) override;

private:
    OcrConfig m_config;

    /**
     * @brief Helper to decode a single sequence (row).
     */
    std::string decode_sequence(const float* seq_data, 
                                int time_steps, 
                                int num_classes, 
                                int stride);
};

} // namespace xinfer::postproc