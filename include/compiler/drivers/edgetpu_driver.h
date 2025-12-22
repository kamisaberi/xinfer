#pragma once

#include <xinfer/compiler/base_compiler.h>

namespace xinfer::compiler {

    /**
     * @brief Google Edge TPU Compiler Driver
     *
     * Wraps the 'edgetpu_compiler' CLI tool.
     *
     * Limitations:
     * 1. Input MUST be a fully INT8 quantized .tflite file.
     * 2. Does not support ONNX directly (user must convert ONNX -> TF -> TFLite first).
     */
    class EdgeTpuDriver : public ICompiler {
    public:
        bool compile(const CompileConfig& config) override;
        bool validate_environment() override;

        std::string get_name() const override {
            return "Google Edge TPU Compiler";
        }
    };

} // namespace xinfer::compiler