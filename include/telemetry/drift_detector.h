#pragma once

#include "types.h"
#include <xinfer/core/tensor.h>
#include <vector>
#include <memory>

namespace xinfer::telemetry {

    /**
     * @brief Concept Drift Detector.
     *
     * Monitors input tensors to detect if live data deviates from training data.
     * Uses Welford's Online Algorithm to compute running Mean/Std
     * with O(1) memory complexity.
     */
    class DriftDetector {
    public:
        /**
         * @brief Initialize with baseline statistics from training set.
         * @param baseline_mean Expected mean.
         * @param baseline_std Expected std dev.
         * @param sensitivity Z-Score threshold (default 3.0 = 3 sigma).
         */
        DriftDetector(const std::vector<float>& baseline_mean,
                      const std::vector<float>& baseline_std,
                      float sensitivity = 3.0f);
        ~DriftDetector();

        /**
         * @brief Update stats with new batch and check for drift.
         *
         * @param batch Input tensor.
         * @return DriftResult indicating if an alert should be raised.
         */
        DriftResult update(const core::Tensor& batch);

        /**
         * @brief Reset internal running stats.
         */
        void reset();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::telemetry