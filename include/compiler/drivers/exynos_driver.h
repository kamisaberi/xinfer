#pragma once

#include <xinfer/compiler/base_compiler.h>

namespace xinfer::compiler {

    /**
     * @brief Samsung Exynos Compiler Driver
     *
     * Wraps the Samsung ENN SDK tools (enn_converter / SCVT) to convert
     * ONNX models into the NNC (Neural Network Container) format.
     *
     * Target: Exynos 9820, 990, 2100, 2200, 2400 (NPU)
     */
    class ExynosDriver : public ICompiler {
    public:
        bool compile(const CompileConfig& config) override;
        bool validate_environment() override;

        std::string get_name() const override {
            return "Samsung ENN Compiler";
        }
    };

} // namespace xinfer::compiler