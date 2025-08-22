// src/builders/onnx_exporter.cpp

#include <include/builders/onnx_exporter.h>
#include <torch/onnx.h>
#include <iostream>
#include <vector>

namespace xinfer::builders {

bool export_to_onnx(xt::Module& model,
                    const std::vector<InputSpec>& input_specs,
                    const std::string& output_path)
{
    try {
        // --- 1. Set model to evaluation mode ---
        model.eval();

        // --- 2. Create example dummy inputs from the specifications ---
        // The exporter needs example tensors to trace the model's execution.
        std::vector<torch::jit::IValue> example_inputs;
        for (const auto& spec : input_specs) {
            // Create a list of c10::IntArrayRef from the shape vector
            c10::IntArrayRef shape(spec.shape.data(), spec.shape.size());
            // Create a random tensor with the specified shape on the CPU
            // (The device doesn't matter for tracing the graph structure)
            example_inputs.push_back(torch::randn(shape, torch::kCPU));
        }

        // --- 3. Define dynamic axes (optional but highly recommended) ---
        // This tells TensorRT that the batch size (dimension 0) can change.
        c10::Dict<c10::string, c10::Dict<int64_t, c10::string>> dynamic_axes;

        std::vector<std::string> input_names;
        std::vector<std::string> output_names;

        for (size_t i = 0; i < input_specs.size(); ++i) {
            c10::Dict<int64_t, c10::string> axis_info;
            axis_info.insert(0, "batch_size"); // Mark dimension 0 as dynamic
            dynamic_axes.insert(input_specs[i].name, axis_info);
            input_names.push_back(input_specs[i].name);
        }

        // Assume output names are "output_0", "output_1", etc. for now
        // A more advanced version could get these from the model spec.
        auto example_outputs = model.forward(example_inputs);
        if (example_outputs.isTensor()) {
             c10::Dict<int64_t, c10::string> axis_info;
             axis_info.insert(0, "batch_size");
             dynamic_axes.insert("output_0", axis_info);
             output_names.push_back("output_0");
        } else if (example_outputs.isTuple()) {
            auto tuple = example_outputs.toTuple();
            for(size_t i=0; i < tuple->elements().size(); ++i) {
                 c10::Dict<int64_t, c10::string> axis_info;
                 axis_info.insert(0, "batch_size");
                 std::string name = "output_" + std::to_string(i);
                 dynamic_axes.insert(name, axis_info);
                 output_names.push_back(name);
            }
        }

        // --- 4. Call the core LibTorch ONNX export function ---
        torch::onnx::export(
            std::make_shared<torch::jit::script::Module>(model._ivalue()), // Get the underlying module
            example_inputs,          // The dummy input tensors
            output_path,             // The file to save to
            torch::onnx::OperatorExportTypes::ONNX, // The export type
            13,                      // Opset version (13 is a good, modern default)
            false,                   // do_constant_folding
            input_names,             // Names for input nodes
            output_names,            // Names for output nodes
            dynamic_axes             // The dynamic axes dictionary
        );

        std::cout << "xTorch model successfully exported to " << output_path << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error during ONNX export: " << e.what() << std::endl;
        return false;
    }
}

} // namespace xinfer::builders