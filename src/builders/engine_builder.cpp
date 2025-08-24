// src/builders/engine_builder.cpp
#include <include/builders/engine_builder.h>
#include <include/hub/downloader.h> // We need this to download the file
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <memory>
#include <fstream>
#include <stdexcept>
#include <iostream>

// Logger for TensorRT (can be shared)
class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

namespace xinfer::builders
{
    struct EngineBuilder::Impl
    {
        Logger logger_;
        std::unique_ptr<nvinfer1::IBuilder> builder_{nvinfer1::createInferBuilder(logger_)};
        std::unique_ptr<nvinfer1::IBuilderConfig> config_{builder_->createBuilderConfig()};

        std::string onnx_path_;
        bool use_fp16_ = false;
        std::shared_ptr<INT8Calibrator> calibrator_ = nullptr;
        int max_batch_size_ = 1;
    };

    EngineBuilder::EngineBuilder() : pimpl_(new Impl())
    {
    }

    EngineBuilder::~EngineBuilder() = default;
    EngineBuilder::EngineBuilder(EngineBuilder&&) noexcept = default;
    EngineBuilder& EngineBuilder::operator=(EngineBuilder&&) noexcept = default;


    EngineBuilder& EngineBuilder::from_onnx(const std::string& onnx_path)
    {
        pimpl_->onnx_path_ = onnx_path;
        return *this;
    }

    EngineBuilder& EngineBuilder::with_fp16()
    {
        pimpl_->use_fp16_ = true;
        return *this;
    }

    EngineBuilder& EngineBuilder::with_int8(std::shared_ptr<INT8Calibrator> calibrator)
    {
        pimpl_->calibrator_ = calibrator;
        return *this;
    }

    EngineBuilder& EngineBuilder::with_max_batch_size(int batch_size)
    {
        pimpl_->max_batch_size_ = batch_size;
        return *this;
    }

    void EngineBuilder::build_and_save(const std::string& output_engine_path)
    {
        const auto explicitBatch = 1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        std::unique_ptr<nvinfer1::INetworkDefinition> network{pimpl_->builder_->createNetworkV2(explicitBatch)};
        std::unique_ptr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, pimpl_->logger_)};

        if (!parser->parseFromFile(pimpl_->onnx_path_.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
        {
            throw std::runtime_error("Failed to parse ONNX file: " + pimpl_->onnx_path_);
        }

        pimpl_->builder_->setMaxBatchSize(pimpl_->max_batch_size_);
        pimpl_->config_->setMaxWorkspaceSize(1ULL << 30); // 1 GB Workspace

        if (pimpl_->use_fp16_ && pimpl_->builder_->platformHasFastFp16())
        {
            pimpl_->config_->setFlag(nvinfer1::BuilderFlag::kFP16);
        }

        if (pimpl_->calibrator_ != nullptr && pimpl_->builder_->platformHasFastInt8())
        {
            pimpl_->config_->setFlag(nvinfer1::BuilderFlag::kINT8);
            // pimpl_->config_->setInt8Calibrator(pimpl_->calibrator_.get()); // This requires a TensorRT-compatible calibrator
        }

        std::unique_ptr<nvinfer1::IHostMemory> serialized_engine{
            pimpl_->builder_->buildSerializedNetwork(*network, *pimpl_->config_)
        };
        if (!serialized_engine)
        {
            throw std::runtime_error("Failed to build TensorRT engine.");
        }

        std::ofstream engine_file(output_engine_path, std::ios::binary);
        engine_file.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());
    }


    // --- NEW FUNCTION IMPLEMENTATION ---
    bool build_engine_from_url(const BuildFromUrlConfig& config)
    {
        try
        {
            // 1. Download the ONNX file to a temporary location
            std::string temp_onnx_path = "temp_downloaded_model.onnx";
            std::cout << "Downloading ONNX model from: " << config.onnx_url << std::endl;

            // Use the hub downloader. It will handle caching.
            // We create a dummy HardwareTarget as it's not needed for a generic file download.
            // The filename will be derived from the URL, which is not ideal but works.
            // A dedicated download utility would be better in the long run.
            std::string downloaded_path = hub::download_engine_asset(config.onnx_url, temp_onnx_path);

            std::cout << "Download complete. Starting engine build..." << std::endl;

            // 2. Use the EngineBuilder to perform the optimization
            EngineBuilder builder;
            builder.from_onnx(downloaded_path)
                   .with_max_batch_size(config.max_batch_size);

            if (config.use_fp16)
            {
                builder.with_fp16();
            }
            if (config.use_int8)
            {
                // builder.with_int8(...);
            }

            builder.build_and_save(config.output_engine_path);

            // 3. Optional: Clean up the temporary ONNX file
            // remove(downloaded_path.c_str());

            return true;
        }
        catch (const std::exception& e)
        {
            std::cerr << "Error during build_engine_from_url: " << e.what() << std::endl;
            return false;
        }
    }
} // namespace xinfer::builders
