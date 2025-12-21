#pragma once

#include <string>
#include <vector>
#include <memory>
#include <xinfer/core/backend_interface.h>
#include <xinfer/core/tensor.h>
#include "config.h"

namespace xinfer::backends::coreml {

    /**
     * @brief Apple Core ML Backend
     *
     * Executes inference using the Apple Neural Engine (ANE) or Metal GPU.
     * Requires the input model to be compiled into a .mlmodelc bundle.
     */
    class CoreMLBackend : public xinfer::IBackend {
    public:
        explicit CoreMLBackend(const CoreMLConfig& config);
        ~CoreMLBackend() override;

        // --- Implementation of IBackend ---

        /**
         * @brief Loads the .mlmodelc bundle.
         *
         * @param model_path Path to the compiled .mlmodelc directory.
         * @return true if the model loaded and input/output descriptions were parsed.
         */
        bool load_model(const std::string& model_path) override;

        /**
         * @brief Executes inference.
         *
         * Wraps C++ Tensors into MLMultiArray objects and dispatches to the ANE.
         */
        void predict(const std::vector<core::Tensor>& inputs,
                     std::vector<core::Tensor>& outputs) override;

        /**
         * @brief Returns device name (e.g., "Apple Neural Engine / Metal")
         */
        std::string device_name() const override;

    private:
        // PImpl idiom to hide Objective-C types (id<MLFeatureProvider>, MLModel*)
        struct Impl;
        std::unique_ptr<Impl> m_impl;

        CoreMLConfig m_config;
    };

} // namespace xinfer::backends::coreml