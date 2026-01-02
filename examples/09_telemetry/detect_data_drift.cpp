#include <iostream>
#include <vector>
#include <random>

#include <xinfer/telemetry/drift_detector.h>
#include <xinfer/core/tensor.h>

using namespace xinfer::telemetry;
using namespace xinfer::core;

// Helper to generate a batch of random data
Tensor generate_batch(int batch_size, int num_features, float mean, float std) {
    Tensor t({1, (int64_t)batch_size, (int64_t)num_features}, DataType::kFLOAT);
    float* ptr = static_cast<float*>(t.data());

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, std);

    for(size_t i=0; i<t.size(); ++i) {
        ptr[i] = dist(gen);
    }
    return t;
}

int main() {
    std::cout << "--- xInfer Data Drift Detector ---" << std::endl;

    // 1. Define Training Baseline
    // Assume we have 4 features (e.g., Packet Size, Frequency, TTL, Flags)
    // These are the stats from your Training Dataset.
    int num_features = 4;
    std::vector<float> train_mean = {100.0f, 50.0f, 0.5f, 10.0f};
    std::vector<float> train_std  = { 20.0f, 10.0f, 0.1f,  2.0f};

    // Initialize Detector (Sensitivity = 3 Sigma)
    DriftDetector detector(train_mean, train_std, 3.0f);

    // 2. Simulate Normal Production Traffic
    // Data matches training distribution
    std::cout << "\n[Phase 1] Processing Normal Traffic..." << std::endl;
    for(int i=0; i<3; ++i) {
        // Generate batch with mean=100, std=20 (Same as training)
        Tensor batch = generate_batch(32, num_features, 100.0f, 20.0f);

        DriftResult res = detector.update(batch);

        if (res.has_drift) {
            std::cout << "  WARNING: Drift detected! (Score: " << res.drift_score << ")" << std::endl;
        } else {
            std::cout << "  Status: OK" << std::endl;
        }
    }

    // 3. Simulate Attack / Anomaly
    // SUDDEN CHANGE: Feature 0 (Packet Size) jumps to Mean=500
    std::cout << "\n[Phase 2] Simulating DDoS Attack (Distribution Shift)..." << std::endl;

    // Batch with mean=500 (Way outside 3-sigma of base=100)
    Tensor attack_batch = generate_batch(32, num_features, 500.0f, 20.0f);

    DriftResult res = detector.update(attack_batch);

    if (res.has_drift) {
        std::cout << "  !!! DRIFT ALERT !!!" << std::endl;
        std::cout << "  Feature: " << res.feature_name << std::endl;
        std::cout << "  Z-Score: " << res.drift_score << " (Threshold: 3.0)" << std::endl;
        std::cout << "  Action: Triggering Retraining Pipeline or Fallback Mode." << std::endl;
    } else {
        std::cout << "  Status: OK (Missed detection?)" << std::endl;
    }

    return 0;
}