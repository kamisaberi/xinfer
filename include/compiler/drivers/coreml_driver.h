#pragma once

#include <xinfer/compiler/base_compiler.h>

namespace xinfer::compiler {

    /**
     * @brief Apple Core ML Compiler Driver
     *
     * Orchestrates the conversion of models (ONNX -> CoreML -> .mlmodelc).
     * Uses 'xcrun coremlcompiler' for the final compilation step.
     */
    class CoreMLDriver : public ICompiler {
    public:
        bool compile(const CompileConfig& config) override;
        bool validate_environment() override;

        std::string get_name() const override {
            return "Apple Core ML Compiler (coremlc)";
        }

    private:
        /**
         * @brief Helper to convert ONNX to intermediate .mlmodel using coremltools (Python)
         */
        bool convert_onnx_to_mlmodel(const std::string& onnx_path,
                                     const std::string& mlmodel_path,
                                     Precision precision);
    };

} // namespace xinfer::compiler