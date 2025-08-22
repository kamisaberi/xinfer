#pragma once


#include <string>
#include <vector>
#include <xtorch/nn/module.h> // Dependency on xTorch's base module
#include "engine_builder.h" // For InputSpec

namespace xinfer::builders {

    /**
     * @brief Exports a trained xTorch model to the ONNX format.
     * @param model A reference to the trained xTorch model.
     * @param input_specs A vector describing the shapes and names of the input tensors.
     * @param output_path The path to save the resulting .onnx file.
     * @return True if export was successful, false otherwise.
     */
    bool export_to_onnx(xt::Module& model,
                        const std::vector<InputSpec>& input_specs,
                        const std::string& output_path);

} // namespace xinfer::builders
