#pragma once

#include <string>
#include <vector>
#include <memory>
#include <xinfer/core/backend_interface.h> // Your global interface
#include <xinfer/core/tensor.h>
#include "config.h" // Platform-specific config

namespace xinfer::backends::ambarella {

    /**
     * @brief Ambarella CVFlow Backend Implementation
     * Loads .cavalry binary DAGs and executes them on the VP (Vector Processor).
     */
    class AmbarellaBackend : public xinfer::IBackend {
    public:
        explicit AmbarellaBackend(const AmbarellaConfig& config);
        ~AmbarellaBackend() override;

        // --- Implementation of IBackend ---

        // Load the .cavalry file
        bool load_model(const std::string& model_path) override;

        // The actual hardware execution
        // Note: Ambarella requires physical memory addresses, handled inside implementation
        void predict(const std::vector<core::Tensor>& inputs,
                     std::vector<core::Tensor>& outputs) override;

        // Returns "Ambarella CV2x/CV5x"
        std::string device_name() const override;

    private:
        // PImpl idiom to hide the proprietary "cavalry_gen.h" headers
        // from the public API (NDA protection).
        struct Impl;
        std::unique_ptr<Impl> m_impl;

        AmbarellaConfig m_config;
    };

} // namespace xinfer::backends::ambarella