#pragma once

#include <xinfer/compiler/base_compiler.h>

namespace xinfer::compiler {

    /**
     * @brief AMD/Xilinx Vitis AI Compiler Driver
     *
     * Wraps the 'vai_c_onnx' (Vitis AI Compiler) tool.
     *
     * Capability:
     * - Can run natively (if installed) or automatically via Docker.
     * - Converts Quantized ONNX -> .xmodel (DPU Instructions).
     *
     * Required Params:
     * - DPU_ARCH: Path to the arch.json file describing the FPGA hardware.
     */
    class VitisDriver : public ICompiler {
    public:
        bool compile(const CompileConfig& config) override;
        bool validate_environment() override;

        std::string get_name() const override {
            return "AMD Vitis AI Compiler (vai_c)";
        }

    private:
        /**
         * @brief Checks if we are currently running inside the Vitis Docker container.
         */
        bool is_native_tool_available();
    };

} // namespace xinfer::compiler