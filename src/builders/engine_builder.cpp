// src/builders/engine_builder.cpp
#include <include/builders/engine_builder.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <memory>
#include <fstream>

// Re-use the same Logger class from engine.cpp
class Logger : public nvinfer1::ILogger { /* ... */ };

namespace xinfer::builders {

struct EngineBuilder::Impl {
    Logger logger_;
    std::unique_ptr<nvinfer1::IBuilder> builder_{nvinfer1::createInferBuilder(logger_)};
    std::unique_ptr<nvinfer1::IBuilderConfig> config_{builder_->createBuilderConfig()};

    std::string onnx_path_;
    bool use_fp16_ = false;
    // ... other config members
};

EngineBuilder::EngineBuilder() : pimpl_(new Impl()) {}
EngineBuilder::~EngineBuilder() = default;

EngineBuilder& EngineBuilder::from_onnx(const std::string& onnx_path) {
    pimpl_->onnx_path_ = onnx_path;
    return *this;
}

EngineBuilder& EngineBuilder::with_fp16() {
    pimpl_->use_fp16_ = true;
    return *this;
}

// ... other config setters ...

void EngineBuilder::build_and_save(const std::string& output_engine_path) {
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::INetworkDefinition> network{pimpl_->builder_->createNetworkV2(explicitBatch)};
    std::unique_ptr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, pimpl_->logger_)};

    if (!parser->parseFromFile(pimpl_->onnx_path_.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        throw std::runtime_error("Failed to parse ONNX file: " + pimpl_->onnx_path_);
    }

    if (pimpl_->use_fp16_ && pimpl_->builder_->platformHasFastFp16()) {
        pimpl_->config_->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    // ... Set other properties on the config ...

    std::unique_ptr<nvinfer1::IHostMemory> serialized_engine{pimpl_->builder_->buildSerializedNetwork(*network, *pimpl_->config_)};
    if (!serialized_engine) {
        throw std::runtime_error("Failed to build TensorRT engine.");
    }

    std::ofstream engine_file(output_engine_path, std::ios::binary);
    engine_file.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());
}

} // namespace xinfer::builders