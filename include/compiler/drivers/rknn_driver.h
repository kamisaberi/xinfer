#pragma once

#include <xinfer/compiler/base_compiler.h>

namespace xinfer::compiler {

    /**
     * @brief Rockchip RKNN Compiler Driver
     *
     * Wraps the 'rknn-toolkit2' Python API.
     * Generates a temporary Python script to drive the conversion from ONNX to RKNN.
     *
     * Targets: RK3588, RK3568, RK3566, RV1126, etc.
     */
    class RknnDriver : public ICompiler {
    public:
        bool compile(const CompileConfig& config) override;
        bool validate_environment() override;

        std::string get_name() const override {
            return "Rockchip RKNN Toolkit2 (Python Wrapper)";
        }

    private:
        /**
         * @brief Generates the build_rknn.py script content.
         */
        std::string generate_python_script(const CompileConfig& config,
                                           const std::string& script_path,
                                           const std::string& target_platform);
    };

} // namespace xinfer::compiler