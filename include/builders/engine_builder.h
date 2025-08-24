#pragma once

#include <string>
#include <memory>
#include <vector>
#include "calibrator.h" // Include the calibrator interface

namespace xinfer::builders {



    // Specification for an input tensor, used for parsing and optimization
    struct InputSpec {
        std::string name;
        std::vector<int64_t> shape;
    };

    /**
     * @class EngineBuilder
     * @brief A high-level fluent API for creating optimized TensorRT engines.
     *
     * This class abstracts the entire TensorRT build pipeline, from parsing
     * an ONNX file to applying optimizations and serializing the final engine.
     */
    class EngineBuilder {
    public:
        EngineBuilder();
        ~EngineBuilder();

        /**
         * @brief Specifies the ONNX model to be optimized.
         * @param onnx_path Path to the .onnx model file.
         * @return A reference to this builder for fluent chaining.
         */
        EngineBuilder& from_onnx(const std::string& onnx_path);



        /**
         * @brief Enables FP16 precision mode for 2x performance on supported GPUs.
         */
        EngineBuilder& with_fp16();

        /**
         * @brief Enables INT8 precision mode for 4x+ performance.
         * @param calibrator A shared pointer to a user-provided calibrator object.
         */
        EngineBuilder& with_int8(std::shared_ptr<INT8Calibrator> calibrator);

        /**
         * @brief Sets the maximum batch size the engine will support.
         */
        EngineBuilder& with_max_batch_size(int batch_size);

        /**
         * @brief Builds the engine and saves it to a file.
         * @param output_engine_path The path to save the final .engine file.
         */
        void build_and_save(const std::string& output_engine_path);

    private:
        struct Impl; // PIMPL idiom
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::builders
