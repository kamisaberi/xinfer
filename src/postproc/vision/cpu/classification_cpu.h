#pragma once

#include <xinfer/postproc/vision/classification_interface.h>
#include <vector>

namespace xinfer::postproc {

    /**
     * @brief CPU Implementation of Classification Post-processing.
     *
     * Handles:
     * 1. Softmax (converting logits to probabilities).
     * 2. Top-K selection (finding best matches).
     * 3. Label mapping (Index -> String).
     */
    class CpuClassificationPostproc : public IClassificationPostprocessor {
    public:
        CpuClassificationPostproc();
        ~CpuClassificationPostproc() override;

        void init(const ClassificationConfig& config) override;

        /**
         * @brief Process logits.
         *
         * @param logits Tensor of shape [Batch, NumClasses].
         * @return A vector containing a list of ClassResults for each batch item.
         */
        std::vector<std::vector<ClassResult>> process(const core::Tensor& logits) override;

    private:
        ClassificationConfig m_config;

        /**
         * @brief Helper to compute Softmax in-place on a vector.
         * Uses the "Max-Trick" for numerical stability to prevent float overflow.
         */
        void apply_softmax(std::vector<float>& values);
    };

} // namespace xinfer::postproc