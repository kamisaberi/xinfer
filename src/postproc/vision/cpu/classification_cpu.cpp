#include "classification_cpu.h"
#include <xinfer/core/logging.h>

#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace xinfer::postproc {

CpuClassificationPostproc::CpuClassificationPostproc() {}
CpuClassificationPostproc::~CpuClassificationPostproc() {}

void CpuClassificationPostproc::init(const ClassificationConfig& config) {
    m_config = config;
}

void CpuClassificationPostproc::apply_softmax(std::vector<float>& values) {
    if (values.empty()) return;

    // 1. Find Max (for stability)
    float max_val = *std::max_element(values.begin(), values.end());

    // 2. Exponentiate and Sum
    float sum = 0.0f;
    for (float& v : values) {
        v = std::exp(v - max_val);
        sum += v;
    }

    // 3. Normalize
    float inv_sum = 1.0f / sum;
    for (float& v : values) {
        v *= inv_sum;
    }
}

std::vector<std::vector<ClassResult>> CpuClassificationPostproc::process(const core::Tensor& logits) {
    auto shape = logits.shape();

    // Validation
    if (shape.size() != 2) {
        XINFER_LOG_ERROR("Classification input must be 2D [Batch, Classes].");
        return {};
    }

    int batch_size = (int)shape[0];
    int num_classes = (int)shape[1];

    // Prepare Output
    std::vector<std::vector<ClassResult>> batch_results(batch_size);

    // Raw Data Pointer
    const float* data_ptr = static_cast<const float*>(logits.data());

    // Loop over batch
    for (int b = 0; b < batch_size; ++b) {
        // Points to start of this image's class scores
        const float* row_ptr = data_ptr + (b * num_classes);

        // Copy to local vector to perform Softmax (don't modify input Tensor)
        std::vector<float> probs(row_ptr, row_ptr + num_classes);

        // Apply Softmax if requested
        if (m_config.apply_softmax) {
            apply_softmax(probs);
        }

        // Top-K Selection
        int k = std::min(m_config.top_k, num_classes);

        // Create an index array: [0, 1, 2, ... num_classes-1]
        std::vector<int> indices(num_classes);
        std::iota(indices.begin(), indices.end(), 0);

        // Partial Sort: O(N log K) - Much faster than sorting the whole array
        // Moves the top K elements to the beginning of the vector
        std::partial_sort(indices.begin(),
                          indices.begin() + k,
                          indices.end(),
                          [&probs](int i1, int i2) {
                              return probs[i1] > probs[i2];
                          });

        // Pack Results
        auto& results = batch_results[b];
        results.reserve(k);

        for (int i = 0; i < k; ++i) {
            int idx = indices[i];

            ClassResult res;
            res.id = idx;
            res.score = probs[idx];

            // Map label string if available
            if (idx < (int)m_config.labels.size()) {
                res.label = m_config.labels[idx];
            } else {
                res.label = "Class " + std::to_string(idx);
            }

            results.push_back(res);
        }
    }

    return batch_results;
}

} // namespace xinfer::postproc