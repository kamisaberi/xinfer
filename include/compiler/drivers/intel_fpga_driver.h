#pragma once

#include <xinfer/compiler/base_compiler.h>

namespace xinfer::compiler {

    /**
     * @brief Intel FPGA AI Suite Compiler Driver
     *
     * Orchestrates the compilation for the Intel DLA (Deep Learning Accelerator) IP.
     *
     * Workflow:
     * 1. Convert ONNX to OpenVINO IR (using 'ovc').
     * 2. Compile IR to DLA Graph Binary (using 'dla_compiler').
     *
     * Requires: Intel OpenVINO + Intel FPGA AI Suite installed.
     */
    class IntelFpgaDriver : public ICompiler {
    public:
        bool compile(const CompileConfig& config) override;
        bool validate_environment() override;

        std::string get_name() const override {
            return "Intel FPGA AI Suite Compiler";
        }

    private:
        /**
         * @brief Helper to convert ONNX to OpenVINO IR first.
         * The DLA compiler consumes IR, not ONNX directly.
         */
        bool convert_onnx_to_ir(const std::string& onnx_path,
                                const std::string& out_dir,
                                const std::string& model_name);
    };

} // namespace xinfer::compiler