#pragma once

#include <string>
#include <vector>
#include <memory>
#include <xinfer/core/backend_interface.h>
#include <xinfer/core/tensor.h>
#include "config.h"

namespace xinfer::backends::qnn {

/**
 * @brief Qualcomm QNN (AI Engine Direct) Backend
 * 
 * High-performance inference engine for Snapdragon processors.
 * Primary target is the Hexagon Tensor Processor (HTP).
 * 
 * Workflow:
 * 1. Loads specific Backend Library (libQnnHtp.so).
 * 2. Loads System Library (libQnnSystem.so).
 * 3. Deserializes a compiled Context Binary.
 * 4. Executes inference graphs.
 */
class QnnBackend : public xinfer::IBackend {
public:
    explicit QnnBackend(const QnnConfig& config);
    ~QnnBackend() override;

    // --- Implementation of IBackend ---

    /**
     * @brief Loads the QNN Context Binary.
     * 
     * @param model_path Path to the .bin file.
     * @return true if context created and graph retrieved.
     */
    bool load_model(const std::string& model_path) override;

    /**
     * @brief Executes inference on the Hexagon NPU.
     * 
     * Handles quantization of inputs if the model expects quantized data
     * but floats are provided (though pre-quantized input is recommended).
     */
    void predict(const std::vector<core::Tensor>& inputs, 
                 std::vector<core::Tensor>& outputs) override;

    /**
     * @brief Returns device name (e.g., "Qualcomm Hexagon HTP")
     */
    std::string device_name() const override;

    // --- QNN Specific API ---

    /**
     * @brief Update the performance vote.
     * Use this to throttle down when idle or boost up when processing a burst.
     */
    void set_performance_mode(HtpPerformanceMode mode);

private:
    // PImpl idiom to hide QNN SDK headers
    struct Impl;
    std::unique_ptr<Impl> m_impl;

    QnnConfig m_config;
};

} // namespace xinfer::backends::qnn