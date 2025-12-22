#pragma once

#include <memory>
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::compiler {

class CompilerFactory {
public:
    /**
     * @brief Creates a compiler driver for the requested target.
     * 
     * @param target The hardware platform (e.g., NVIDIA_TRT, ROCKCHIP_RKNN)
     * @return std::unique_ptr<ICompiler> The specific driver instance, or nullptr if invalid.
     */
    static std::unique_ptr<ICompiler> create(Target target);
};

} // namespace xinfer::compiler