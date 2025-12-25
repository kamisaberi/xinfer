#pragma once
#include <xinfer/core/tensor.h>
#include <vector>
#include <string>

namespace xinfer::postproc {

    struct ClassResult {
        int id;
        float score;
        std::string label; // Optional, populated if labels provided
    };

    struct ClassificationConfig {
        int top_k = 5;       // Return top 5 results
        bool apply_softmax = true; // If model returns raw logits
        std::vector<std::string> labels; // Label map
    };

    class IClassificationPostprocessor {
    public:
        virtual ~IClassificationPostprocessor() = default;
        virtual void init(const ClassificationConfig& config) = 0;

        /**
         * @brief Process classification logits.
         * @param logits Tensor [Batch, NumClasses]
         * @return Batch of vectors (one vector of Top-K results per image)
         */
        virtual std::vector<std::vector<ClassResult>> process(const core::Tensor& logits) = 0;
    };

}