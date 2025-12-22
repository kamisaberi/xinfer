#pragma once

#include <xinfer/compiler/base_compiler.h>

namespace xinfer::compiler {

    /**
     * @brief Lattice sensAI Compiler Driver
     *
     * Wraps the Lattice Neural Network Compiler.
     * Converts models (TFLite INT8) into FPGA Command Streams (.bin).
     *
     * Target: iCE40 UltraPlus, CrossLink-NX, ECP5, Certus-NX.
     */
    class LatticeDriver : public ICompiler {
    public:
        bool compile(const CompileConfig& config) override;
        bool validate_environment() override;

        std::string get_name() const override {
            return "Lattice sensAI Compiler";
        }
    };

} // namespace xinfer::compiler