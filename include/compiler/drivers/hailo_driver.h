#pragma once

#include <xinfer/compiler/base_compiler.h>

namespace xinfer::compiler {

    /**
     * @brief Hailo Dataflow Compiler Driver
     *
     * Wraps the 'hailo_sdk_client' Python API to convert ONNX models
     * into the Hailo Executable Format (.hef).
     *
     * Targets: Hailo-8, Hailo-8L, Hailo-10, Hailo-15.
     */
    class HailoDriver : public ICompiler {
    public:
        bool compile(const CompileConfig& config) override;
        bool validate_environment() override;

        std::string get_name() const override {
            return "Hailo Dataflow Compiler (DFC)";
        }

    private:
        // Helper to generate the python build script
        std::string generate_python_script(const CompileConfig& config,
                                           const std::string& script_path,
                                           const std::string& arch);
    };

} // namespace xinfer::compiler