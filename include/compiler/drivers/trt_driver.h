#pragma once
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::compiler {

class TrtDriver : public ICompiler {
public:
    bool compile(const CompileConfig& config) override;
    bool validate_environment() override;
    std::string get_name() const override { return "NVIDIA TensorRT Driver"; }
};

}