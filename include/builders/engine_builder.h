#pragma once

#include <string>
#include <memory>
#include <vector>

// Forward declaration to avoid including the full header here
namespace xinfer::builders
{
    class INT8Calibrator;
}

namespace xinfer::builders
{
    struct InputSpec
    {
        std::string name;
        std::vector<int64_t> shape;
    };

    class EngineBuilder
    {
    public:
        EngineBuilder();
        ~EngineBuilder();

        EngineBuilder(const EngineBuilder&) = delete;
        EngineBuilder& operator=(const EngineBuilder&) = delete;
        EngineBuilder(EngineBuilder&&) noexcept;
        EngineBuilder& operator=(EngineBuilder&&) noexcept;

        EngineBuilder& from_onnx(const std::string& onnx_path);
        EngineBuilder& with_fp16();
        EngineBuilder& with_int8(std::shared_ptr<INT8Calibrator> calibrator);
        EngineBuilder& with_max_batch_size(int batch_size);

        void build_and_save(const std::string& output_engine_path);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };


    // --- NEW SECTION ---

    /**
     * @struct BuildFromUrlConfig
     * @brief Configuration for the download-and-build workflow.
     */
    struct BuildFromUrlConfig
    {
        std::string onnx_url;
        std::string output_engine_path;
        bool use_fp16 = false;
        bool use_int8 = false;
        int max_batch_size = 1;
        // ... add other builder options like workspace_mb if needed
    };

    /**
     * @brief Downloads an ONNX model from a URL and builds a TensorRT engine.
     *
     * This is a high-level convenience function that automates the entire process.
     *
     * @param config The configuration for the build process.
     * @return True on success, false on failure.
     */
    bool build_engine_from_url(const BuildFromUrlConfig& config);
}
