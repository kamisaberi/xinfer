#pragma once

#include <string>
#include <vector>
#include <memory>
#include <xinfer/core/backend_interface.h>
#include <xinfer/core/tensor.h>
#include "config.h"

namespace xinfer::backends::google_tpu {

    /**
     * @brief Google Edge TPU Backend
     *
     * Executes inference using the libedgetpu delegate for TensorFlow Lite.
     * Requires models to be quantized (INT8) and compiled via the edgetpu_compiler.
     */
    class EdgeTpuBackend : public xinfer::IBackend {
    public:
        explicit EdgeTpuBackend(const EdgeTpuConfig& config);
        ~EdgeTpuBackend() override;

        // --- Implementation of IBackend ---

        /**
         * @brief Loads the .tflite model and initializes the Edge TPU Delegate.
         *
         * @param model_path Path to the _edgetpu.tflite file.
         * @return true if the TPU context was created and model loaded.
         */
        bool load_model(const std::string& model_path) override;

        /**
         * @brief Executes inference.
         *
         * Copies data into the TFLite input tensors, invokes the interpreter,
         * and reads from output tensors.
         */
        void predict(const std::vector<core::Tensor>& inputs,
                     std::vector<core::Tensor>& outputs) override;

        /**
         * @brief Returns device name (e.g., "Google Edge TPU (USB)")
         */
        std::string device_name() const override;

        /**
         * @brief Get the quantization parameters for a specific input tensor.
         * Useful for pre-processing scaling.
         */
        float get_input_scale(int index) const;
        int32_t get_input_zero_point(int index) const;

    private:
        // PImpl idiom to hide TFLite and libedgetpu headers
        struct Impl;
        std::unique_ptr<Impl> m_impl;

        EdgeTpuConfig m_config;
    };

} // namespace xinfer::backends::google_tpu