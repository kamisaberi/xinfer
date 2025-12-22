#pragma once

#include <xinfer/compiler/base_compiler.h>

namespace xinfer::compiler {

    /**
     * @brief AMD Ryzen AI Compiler Driver
     *
     * Wraps the Vitis AI Quantizer (vai_q_onnx) to prepare models for the
     * Ryzen AI NPU (XDNA Architecture).
     *
     * Process:
     * 1. Takes standard Float32 ONNX.
     * 2. Runs Post-Training Quantization (PTQ) via vai_q_onnx.
     * 3. Outputs a Quantized ONNX file ready for the Vitis AI Execution Provider.
     *
     * Requirements:
     * - Python environment with 'vai_q_onnx' installed.
     */
    class RyzenAIDriver : public ICompiler {
    public:
        bool compile(const CompileConfig& config) override;
        bool validate_environment() override;

        std::string get_name() const override {
            return "AMD Ryzen AI Quantizer (vai_q_onnx)";
        }

    private:
        /**
         * @brief Generates a Python script to run the quantization process.
         */
        std::string generate_quantize_script(const CompileConfig& config,
                                             const std::string& script_path);
    };

} // namespace xinfer::compiler