#pragma once

#include <xinfer/compiler/base_compiler.h>

namespace xinfer::compiler {

    /**
     * @brief Ambarella CVFlow Compiler Driver
     *
     * Wraps the 'cnngen' tool to convert ONNX models into Cavalry binaries.
     * Targets CV2x, CV5x, and CV3 platforms.
     */
    class AmbarellaDriver : public ICompiler {
    public:
        bool compile(const CompileConfig& config) override;
        bool validate_environment() override;

        std::string get_name() const override {
            return "Ambarella CVFlow Compiler (CNNGen)";
        }
    };

} // namespace xinfer::compiler