#pragma once

#include <xinfer/compiler/base_compiler.h>

namespace xinfer::compiler {

    /**
     * @brief Microchip VectorBlox Compiler Driver
     *
     * Wraps the VectorBlox SDK to convert models into binary blobs
     * compatible with the VectorBlox CNN IP Core on PolarFire FPGAs.
     *
     * Flow:
     * 1. (Optional) Convert ONNX -> VNNX (VectorBlox Intermediate).
     * 2. Compile VNNX -> BLOB (Hardware Instructions).
     *
     * Critical Param: 'CORE' (e.g., V1000) must match the FPGA bitstream.
     */
    class VectorBloxDriver : public ICompiler {
    public:
        bool compile(const CompileConfig& config) override;
        bool validate_environment() override;

        std::string get_name() const override {
            return "Microchip VectorBlox SDK Compiler";
        }
    };

} // namespace xinfer::compiler