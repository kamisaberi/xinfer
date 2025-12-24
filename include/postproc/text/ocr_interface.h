#pragma once

#include <xinfer/core/tensor.h>
#include <string>
#include <vector>

namespace xinfer::postproc {

/**
 * @brief Configuration for OCR/CTC Decoding
 */
struct OcrConfig {
    // The character set mapping (Index -> Char).
    // e.g. "0123456789abcdefghijklmnopqrstuvwxyz"
    std::string vocabulary; 
    
    // The index representing the "Blank" token in CTC.
    // Usually 0 (for PyTorch) or NumClasses-1 (for Keras).
    int blank_index = 0;

    // Minimum confidence probability to consider a character.
    float min_confidence = 0.0f;
};

/**
 * @brief Interface for Text Post-processors (OCR/ASR).
 */
class IOcrPostprocessor {
public:
    virtual ~IOcrPostprocessor() = default;

    virtual void init(const OcrConfig& config) = 0;

    /**
     * @brief Decode raw probabilities into strings.
     * 
     * @param logits Tensor of shape [TimeSteps, Batch, NumClasses] 
     *               or [Batch, TimeSteps, NumClasses].
     * @return Vector of decoded strings (one per batch item).
     */
    virtual std::vector<std::string> process(const core::Tensor& logits) = 0;
};

} // namespace xinfer::postproc