#pragma once

#include <xinfer/compiler/base_compiler.h>

namespace xinfer::compiler {

    /**
     * @brief Qualcomm QNN Compiler Driver
     *
     * Orchestrates the multi-stage build process for Snapdragon:
     * 1. ONNX -> QNN C++ (qnn-onnx-converter)
     * 2. C++ -> Shared Object (qnn-model-lib-generator)
     * 3. Shared Object -> Context Binary (qnn-context-binary-generator)
     *
     * Requirements:
     * - QNN SDK installed
     * - Android NDK installed (for compiling the intermediate graph library)
     */
    class QnnDriver : public ICompiler {
    public:
        bool compile(const CompileConfig& config) override;
        bool validate_environment() override;

        std::string get_name() const override {
            return "Qualcomm QNN Toolchain";
        }

    private:
        // Helper to get path to specific QNN binary (converter/generator)
        std::string get_tool_path(const std::string& tool_name);
    };

} // namespace xinfer::compiler