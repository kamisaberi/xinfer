#pragma once

#include <string>
#include <vector>
#include <memory>
#include "tensor.h"

// Forward declare CUDA stream type to avoid including cuda_runtime.h
typedef struct CUstream_st* cudaStream_t;

namespace xinfer::core {

/**
 * @class InferenceEngine
 * @brief Manages a compiled TensorRT engine and provides methods for execution.
 *
 * This class loads a serialized .engine file and handles the full inference
 * lifecycle, including managing the execution context and GPU memory bindings.
 */
class InferenceEngine {
public:
    /**
     * @brief Constructor that loads and deserializes a TensorRT engine file.
     * @param engine_path Path to the .engine file.
     */
    explicit InferenceEngine(const std::string& engine_path);
    ~InferenceEngine();

    // --- Rule of Five ---
    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;
    InferenceEngine(InferenceEngine&&) noexcept;
    InferenceEngine& operator=(InferenceEngine&&) noexcept;

    /**
     * @brief Executes the model synchronously on a vector of input tensors.
     * @param inputs A vector of xinfer::core::Tensor objects.
     * @return A vector of xinfer::core::Tensor objects holding the outputs.
     */
    std::vector<Tensor> infer(const std::vector<Tensor>& inputs);

    /**
     * @brief Executes the model asynchronously on a given CUDA stream.
     * @param inputs A vector of input tensors.
     * @param outputs A vector of pre-allocated output tensors.
     * @param stream The CUDA stream on which to enqueue the inference.
     */
    void infer_async(const std::vector<Tensor>& inputs,
                     std::vector<Tensor>& outputs,
                     cudaStream_t stream);

    // --- Introspection API ---
    int get_num_inputs() const;
    int get_num_outputs() const;
    std::vector<int64_t> get_input_shape(int index = 0) const;
    std::vector<int64_t> get_output_shape(int index = 0) const;

private:
    struct Impl; // PIMPL idiom to hide TensorRT headers
    std::unique_ptr<Impl> pimpl_;
};

} // namespace xinfer::core
