#include <xinfer/backends/apple_coreml/backend.h>
#include <xinfer/core/logging.h>
#include <xinfer/backends/backend_factory.h>

#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>

#include <iostream>
#include <vector>
#include <map>

namespace xinfer::backends::coreml {

// =================================================================================
// 1. PImpl Implementation (Objective-C++ Hybrid)
// =================================================================================

struct CoreMLBackend::Impl {
    CoreMLConfig config;
    MLModel* model = nil; // The actual Core ML Engine instance

    // Metadata caching for fast tensor mapping
    NSMutableDictionary<NSString*, MLFeatureDescription*>* inputDescriptions;
    NSMutableDictionary<NSString*, MLFeatureDescription*>* outputDescriptions;

    explicit Impl(const CoreMLConfig& cfg) : config(cfg) {}

    ~Impl() {
        // ARC (Automatic Reference Counting) handles release of MLModel*
        model = nil;
    }

    // Helper: Convert C++ std::string to NSString
    NSString* to_nsstring(const std::string& str) {
        return [NSString stringWithUTF8String:str.c_str()];
    }

    // Helper: Create MLMultiArray from xInfer Tensor
    MLMultiArray* tensor_to_mlarray(const core::Tensor& tensor, NSString* featureName) {
        NSError* error = nil;

        // 1. Get Shape
        std::vector<NSNumber*> shape;
        for (auto dim : tensor.shape()) {
            shape.push_back(@(dim));
        }
        NSArray<NSNumber*>* shapeArray = [NSArray arrayWithObjects:shape.data() count:shape.size()];

        // 2. Get Strides (assuming contiguous C-order for simplicity)
        std::vector<NSNumber*> strides;
        int64_t stride = 1;
        for (int i = tensor.shape().size() - 1; i >= 0; --i) {
            strides.insert(strides.begin(), @(stride));
            stride *= tensor.shape()[i];
        }
        NSArray<NSNumber*>* strideArray = [NSArray arrayWithObjects:strides.data() count:strides.size()];

        // 3. Determine DataType
        MLMultiArrayDataType type = MLMultiArrayDataTypeFloat32;
        if (tensor.dtype() == core::DataType::kFLOAT16) type = MLMultiArrayDataTypeFloat16;
        else if (tensor.dtype() == core::DataType::kINT32) type = MLMultiArrayDataTypeInt32;

        // 4. Create MLMultiArray (Ideally zero-copy, but starting with safe copy)
        // Note: For absolute zero-copy, we need to wrap the existing buffer ptr,
        // but that requires managing the lifetime of the C++ Tensor vs Obj-C object.
        MLMultiArray* mlArray = [[MLMultiArray alloc] initWithShape:shapeArray
                                                           dataType:type
                                                              error:&error];

        if (error) {
            XINFER_LOG_ERROR("Failed to allocate MLMultiArray: " + std::string([[error localizedDescription] UTF8String]));
            return nil;
        }

        // 5. Copy Data
        // Core ML expects data in its own managed buffer usually
        size_t byte_size = tensor.size() * core::element_size(tensor.dtype());
        memcpy(mlArray.dataPointer, tensor.data(), byte_size);

        return mlArray;
    }

    // Helper: Convert MLMultiArray back to xInfer Tensor
    void mlarray_to_tensor(MLMultiArray* mlArray, core::Tensor& tensor) {
        // Resize tensor if needed
        std::vector<int64_t> shape;
        for (NSNumber* dim in mlArray.shape) {
            shape.push_back([dim longLongValue]);
        }

        // Core ML mostly returns Float32 or Float16
        // We assume Float32 for this implementation
        if (tensor.empty()) {
            tensor.resize(shape, core::DataType::kFLOAT);
        }

        // Copy Data
        size_t byte_size = tensor.size() * sizeof(float);
        memcpy(tensor.data(), mlArray.dataPointer, byte_size);
    }
};

// =================================================================================
// 2. Public API Implementation
// =================================================================================

CoreMLBackend::CoreMLBackend(const CoreMLConfig& config)
    : m_config(config), m_impl(std::make_unique<Impl>(config)) {
}

CoreMLBackend::~CoreMLBackend() = default;

bool CoreMLBackend::load_model(const std::string& model_path) {
    @autoreleasepool {
        NSError* error = nil;
        NSURL* modelURL = [NSURL fileURLWithPath:m_impl->to_nsstring(model_path)];

        // 1. Configure Hardware Strategy
        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];

        switch (m_config.compute_unit) {
            case ComputeUnit::ALL:
                config.computeUnits = MLComputeUnitsAll;
                break;
            case ComputeUnit::CPU_AND_GPU:
                config.computeUnits = MLComputeUnitsCPUAndGPU;
                break;
            case ComputeUnit::CPU_ONLY:
                config.computeUnits = MLComputeUnitsCPUOnly;
                break;
            case ComputeUnit::CPU_AND_NEURAL_ENGINE:
                config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
                break;
        }

        config.allowLowPrecisionAccumulationOnGPU = m_config.allow_low_precision;

        // 2. Load Model
        // Note: Core ML loads synchronously here
        m_impl->model = [MLModel modelWithContentsOfURL:modelURL
                                          configuration:config
                                                  error:&error];

        if (error || !m_impl->model) {
            std::string errStr = error ? [[error localizedDescription] UTF8String] : "Unknown";
            XINFER_LOG_ERROR("Core ML Load Failed: " + errStr);
            return false;
        }

        // 3. Cache Input/Output Descriptions
        MLModelDescription* desc = m_impl->model.modelDescription;
        m_impl->inputDescriptions = [NSMutableDictionary dictionaryWithDictionary:desc.inputDescriptionsByName];
        m_impl->outputDescriptions = [NSMutableDictionary dictionaryWithDictionary:desc.outputDescriptionsByName];

        XINFER_LOG_INFO("Loaded Core ML Model: " + model_path);
        return true;
    }
}

void CoreMLBackend::predict(const std::vector<core::Tensor>& inputs,
                            std::vector<core::Tensor>& outputs) {
    @autoreleasepool {
        NSError* error = nil;

        // 1. Prepare Inputs Dictionary
        // Core ML requires named inputs. We map inputs[0] -> First Feature Name, etc.
        // NOTE: This assumes inputs are passed in the same order as model metadata.

        NSMutableDictionary<NSString*, MLFeatureValue*>* featureDict = [NSMutableDictionary dictionary];
        NSArray<NSString*>* inputNames = [m_impl->inputDescriptions allKeys]; // Order might be random, need strict ordering logic ideally.

        // Better strategy: Sort names to ensure deterministic mapping if the user didn't provide names.
        // For xInfer standard, we might assume strict ordering index 0 -> model input 0.
        inputNames = [inputNames sortedArrayUsingSelector:@selector(compare:)];

        if (inputs.size() != inputNames.count) {
            XINFER_LOG_ERROR("Input count mismatch. Model expects " + std::to_string(inputNames.count));
            return;
        }

        for (size_t i = 0; i < inputs.size(); ++i) {
            NSString* name = inputNames[i];
            MLMultiArray* mlArray = m_impl->tensor_to_mlarray(inputs[i], name);
            if (!mlArray) return;

            featureDict[name] = [MLFeatureValue featureValueWithMultiArray:mlArray];
        }

        id<MLFeatureProvider> inputFeatures = [[MLDictionaryFeatureProvider alloc] initWithDictionary:featureDict error:&error];
        if (error) {
            XINFER_LOG_ERROR("Failed to create FeatureProvider: " + std::string([[error localizedDescription] UTF8String]));
            return;
        }

        // 2. Run Prediction
        id<MLFeatureProvider> outputFeatures = [m_impl->model predictionFromFeatures:inputFeatures error:&error];

        if (error) {
            XINFER_LOG_ERROR("Core ML Inference Failed: " + std::string([[error localizedDescription] UTF8String]));
            return;
        }

        // 3. Extract Outputs
        NSArray<NSString*>* outputNames = [m_impl->outputDescriptions allKeys];
        outputNames = [outputNames sortedArrayUsingSelector:@selector(compare:)];

        if (outputs.size() != outputNames.count) {
            outputs.resize(outputNames.count);
        }

        for (size_t i = 0; i < outputNames.count; ++i) {
            NSString* name = outputNames[i];
            MLFeatureValue* value = [outputFeatures featureValueForName:name];

            if (value.type == MLFeatureTypeMultiArray) {
                m_impl->mlarray_to_tensor(value.multiArrayValue, outputs[i]);
            } else {
                XINFER_LOG_WARN("Non-MultiArray output not yet supported (e.g. Image/String).");
            }
        }
    }
}

std::string CoreMLBackend::device_name() const {
    return "Apple Core ML (ANE/GPU/CPU)";
}

// =================================================================================
// 3. Auto-Registration
// =================================================================================

namespace {
    volatile bool registered = xinfer::backends::BackendFactory::register_backend(
        xinfer::Target::APPLE_COREML,
        [](const xinfer::Config& config) -> std::unique_ptr<xinfer::IBackend> {
            CoreMLConfig ml_cfg;
            ml_cfg.model_path = config.model_path;

            // Parse vendor flags
            for(const auto& param : config.vendor_params) {
                if(param == "COMPUTE_UNIT=CPU_ONLY") ml_cfg.compute_unit = ComputeUnit::CPU_ONLY;
                if(param == "COMPUTE_UNIT=ALL") ml_cfg.compute_unit = ComputeUnit::ALL;
            }

            auto backend = std::make_unique<CoreMLBackend>(ml_cfg);
            if(backend->load_model(ml_cfg.model_path)) {
                return backend;
            }
            return nullptr;
        }
    );
}

} // namespace xinfer::backends::coreml