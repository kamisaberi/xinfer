#pragma once

#include <xinfer/compiler/base_compiler.h>

namespace xinfer::compiler {

    /**
     * @brief Intel OpenVINO Compiler Driver
     *
     * Wraps the 'ovc' (formerly Model Optimizer) tool to convert
     * standard models into OpenVINO IR format (.xml + .bin).
     *
     * Targets: Intel CPU, iGPU (Iris Xe), Discrete GPU (Arc), NPU (Core Ultra).
     */
    class OpenVINODriver : public ICompiler {
    public:
        bool compile(const CompileConfig& config) override;
        bool validate_environment() override;

        std::string get_name() const override {
            return "Intel OpenVINO Converter (ovc)";
        }
    };

} // namespace xinfer::compiler