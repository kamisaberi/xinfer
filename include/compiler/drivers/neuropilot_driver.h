#pragma once

#include <xinfer/compiler/base_compiler.h>

namespace xinfer::compiler {

    /**
     * @brief MediaTek NeuroPilot Compiler Driver
     *
     * Wraps the 'ncc' (Neuron Compiler) tool to convert models into
     * MediaTek DLA binaries.
     *
     * Target: MediaTek Genio (IoT), Dimensity (Mobile).
     */
    class NeuroPilotDriver : public ICompiler {
    public:
        bool compile(const CompileConfig& config) override;
        bool validate_environment() override;

        std::string get_name() const override {
            return "MediaTek NeuroPilot Compiler (ncc)";
        }
    };

} // namespace xinfer::compiler