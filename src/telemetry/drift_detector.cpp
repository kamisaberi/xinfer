#include <xinfer/telemetry/drift_detector.h>
#include <xinfer/core/logging.h>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace xinfer::telemetry {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct DriftDetector::Impl {
    std::vector<float> base_mean_;
    std::vector<float> base_std_;
    float threshold_;

    // Running stats (Welford's Algorithm)
    std::vector<float> running_mean_;
    std::vector<float> running_M2_; // Sum of squares of differences
    uint64_t count_ = 0;

    Impl(const std::vector<float>& mean, const std::vector<float>& std, float thresh)
        : base_mean_(mean), base_std_(std), threshold_(thresh)
    {
        running_mean_.resize(mean.size(), 0.0f);
        running_M2_.resize(mean.size(), 0.0f);
    }

    void reset() {
        std::fill(running_mean_.begin(), running_mean_.end(), 0.0f);
        std::fill(running_M2_.begin(), running_M2_.end(), 0.0f);
        count_ = 0;
    }

    DriftResult check_batch(const core::Tensor& batch) {
        DriftResult result = {false, 0.0f, ""};

        // Assume batch is [Batch, Features] or flattened features
        // Simplified: Calculate mean of this batch and update running stats

        const float* data = static_cast<const float*>(batch.data());
        size_t total_elements = batch.size();
        size_t num_features = base_mean_.size();

        // Basic check
        if (total_elements % num_features != 0) {
            // Dimension mismatch, can't check per-feature drift
            return result;
        }

        int batch_size = total_elements / num_features;

        // 1. Update Running Statistics (Iterate over batch)
        for (int b = 0; b < batch_size; ++b) {
            count_++;
            for (int f = 0; f < num_features; ++f) {
                float val = data[b * num_features + f];
                float delta = val - running_mean_[f];
                running_mean_[f] += delta / count_;
                float delta2 = val - running_mean_[f];
                running_M2_[f] += delta * delta2;
            }
        }

        // 2. Check for Drift (Compare Running Mean vs Baseline Mean)
        // Z = (RunningMean - BaseMean) / BaseStd
        // (Note: This is a simplified Z-test. Real drift detection is more complex)

        float max_z_score = 0.0f;
        int drifted_feature_idx = -1;

        for (int f = 0; f < num_features; ++f) {
            if (base_std_[f] < 1e-6) continue; // Avoid div/0 for constant features

            float z_score = std::abs(running_mean_[f] - base_mean_[f]) / base_std_[f];

            if (z_score > max_z_score) {
                max_z_score = z_score;
                drifted_feature_idx = f;
            }
        }

        result.drift_score = max_z_score;
        if (max_z_score > threshold_) {
            result.has_drift = true;
            result.feature_name = "Feature_" + std::to_string(drifted_feature_idx);

            // Auto-reset if drift is massive? Or just alerting.
            // Keeping state allows tracking "return to normal".
        }

        return result;
    }
};

// =================================================================================
// Public API
// =================================================================================

DriftDetector::DriftDetector(const std::vector<float>& m, const std::vector<float>& s, float t)
    : pimpl_(std::make_unique<Impl>(m, s, t)) {}

DriftDetector::~DriftDetector() = default;

void DriftDetector::reset() {
    pimpl_->reset();
}

DriftResult DriftDetector::update(const core::Tensor& batch) {
    return pimpl_->check_batch(batch);
}

} // namespace xinfer::telemetry