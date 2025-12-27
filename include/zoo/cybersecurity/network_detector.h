#pragma once

#include <string>
#include <vector>
#include <memory>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::cybersecurity {

    /**
     * @brief Represents the features of a single network flow.
     * Based on common IDS dataset features.
     */
    struct NetworkFlow {
        // Example features (simplified from 80+ in real datasets)
        std::string protocol;    // "TCP", "UDP"
        int src_port;
        int dst_port;
        long long flow_duration; // microseconds
        long long total_fwd_packets;
        long long total_bwd_packets;
        long long total_fwd_bytes;
        long long total_bwd_bytes;
    };

    struct IntrusionResult {
        bool is_attack;
        std::string attack_type; // "Benign", "DDoS", "PortScan", "Botnet"
        float confidence;
    };

    struct NetworkDetectorConfig {
        // Hardware Target (CPU or lightweight NPU like Rockchip)
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., ids_mlp.onnx)
        std::string model_path;

        // Label Map (Class ID -> Attack Name)
        std::vector<std::string> labels;

        // Normalization (Mean/Std for each feature)
        // Must match the training data statistics.
        std::vector<float> mean;
        std::vector<float> std;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class NetworkDetector {
    public:
        explicit NetworkDetector(const NetworkDetectorConfig& config);
        ~NetworkDetector();

        // Move semantics
        NetworkDetector(NetworkDetector&&) noexcept;
        NetworkDetector& operator=(NetworkDetector&&) noexcept;
        NetworkDetector(const NetworkDetector&) = delete;
        NetworkDetector& operator=(const NetworkDetector&) = delete;

        /**
         * @brief Analyze a network flow.
         *
         * @param flow The features of the network connection.
         * @return The classification of the flow.
         */
        IntrusionResult analyze(const NetworkFlow& flow);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::cybersecurity