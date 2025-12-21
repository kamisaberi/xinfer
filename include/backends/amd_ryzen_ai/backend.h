#pragma once

#include <string>
#include <vector>
#include <memory>
#include <xinfer/core/backend_interface.h>
#include <xinfer/core/tensor.h>
#include "config.h"

namespace xinfer::backends::ryzen_ai {

    /**
     * @brief AMD Ryzen AI Backend
     *
     * Supports inference on AMD XDNA NPUs (Neural Processing Units).
     * Can utilize either the Vitis AI Execution Provider or Native XRT/VART.
     */
    class RyzenAIBackend : public xinfer::IBackend {
    public:
        explicit RyzenAIBackend(const RyzenAIConfig& config);
        ~RyzenAIBackend() override;

        // --- Implementation of IBackend ---

        /**
         * @brief Loads the model artifact.
         * @param model_path Path to .onnx (if EP) or .xmodel (if Native)
         */
        bool load_model(const std::string& model_path) override;

        /**
         * @brief Executes inference on the NPU.
         *
         * Handles memory movement between System RAM and NPU Private Memory.
         * If 'inputs' use CmaContiguous memory, zero-copy optimization is attempted
         * (supported on Native XRT mode).
         */
        void predict(const std::vector<core::Tensor>& inputs,
                     std::vector<core::Tensor>& outputs) override;

        /**
         * @brief Returns device name (e.g., "AMD Ryzen AI - Phoenix NPU")
         */
        std::string device_name() const override;

        /**
         * @brief Specific query for NPU utilization (if supported by driver)
         */
        double get_npu_load() const;

    private:
        // PImpl idiom to hide <xrt/xrt_device.h> and <onnxruntime_cxx_api.h>
        struct Impl;
        std::unique_ptr<Impl> m_impl;

        RyzenAIConfig m_config;
    };

} // namespace xinfer::backends::ryzen_ai