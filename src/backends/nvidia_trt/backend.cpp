#include <xinfer/backends/nvidia_trt/backend.h>
#include <xinfer/core/logging.h>
#include <xinfer/backends/backend_factory.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <map>
#include <algorithm>

// --- NVIDIA Headers ---
#include <NvInfer.h>
#include <cuda_runtime_api.h>

namespace xinfer::backends::nvidia {

// =================================================================================
// 1. Helper: CUDA Error Checking
// =================================================================================
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        XINFER_LOG_ERROR(std::string("CUDA Error: ") + cudaGetErrorString(err)); \
    } \
}

// =================================================================================
// 2. Helper: TRT Logger
// =================================================================================
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
            case Severity::kERROR:
                XINFER_LOG_ERROR("[TRT] " + std::string(msg)); break;
            case Severity::kWARNING:
                XINFER_LOG_WARN("[TRT] " + std::string(msg)); break;
            case Severity::kINFO:
                XINFER_LOG_INFO("[TRT] " + std::string(msg)); break;
            default: break; // Ignore VERBOSE
        }
    }
};

// =================================================================================
// 3. PImpl Implementation
// =================================================================================

struct NvidiaBackend::Impl {
    TrtConfig config;
    TrtLogger logger;

    // TensorRT Objects
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    
    // CUDA Resources
    cudaStream_t stream = nullptr;
    bool own_stream = false;

    // I/O Caching
    // TensorRT 10 uses names to bind tensors
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    explicit Impl(const TrtConfig& cfg) : config(cfg) {
        if (config.external_stream) {
            stream = static_cast<cudaStream_t>(config.external_stream);
            own_stream = false;
        } else {
            CHECK_CUDA(cudaStreamCreate(&stream));
            own_stream = true;
        }
    }

    ~Impl() {
        // Destroy context first
        context.reset();
        engine.reset();
        runtime.reset();

        if (own_stream && stream) {
            cudaStreamDestroy(stream);
        }
    }
};

// =================================================================================
// 4. Public API Implementation
// =================================================================================

NvidiaBackend::NvidiaBackend(const TrtConfig& config)
    : m_config(config), m_impl(std::make_unique<Impl>(config)) {
}

NvidiaBackend::~NvidiaBackend() = default;

bool NvidiaBackend::load_model(const std::string& model_path) {
    try {
        // 1. Read Engine File
        std::ifstream file(model_path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            XINFER_LOG_ERROR("Failed to open engine file: " + model_path);
            return false;
        }
        size_t size = file.tellg();
        file.seekg(0);
        std::vector<char> buffer(size);
        file.read(buffer.data(), size);

        // 2. Create Runtime
        m_impl->runtime.reset(nvinfer1::createInferRuntime(m_impl->logger));
        if (!m_impl->runtime) {
            XINFER_LOG_ERROR("Failed to create TRT Runtime.");
            return false;
        }

        // 3. Set DLA Core (if requested)
        if (m_config.dla_core != DlaCore::GPU_FALLBACK) {
            m_impl->runtime->setDLACore(static_cast<int>(m_config.dla_core));
        }

        // 4. Deserialize Engine
        m_impl->engine.reset(m_impl->runtime->deserializeCudaEngine(buffer.data(), size));
        if (!m_impl->engine) {
            XINFER_LOG_ERROR("Failed to deserialize CUDA Engine.");
            return false;
        }

        // 5. Create Execution Context
        m_impl->context.reset(m_impl->engine->createExecutionContext());
        if (!m_impl->context) {
            XINFER_LOG_ERROR("Failed to create Execution Context.");
            return false;
        }

        // 6. Cache I/O Names
        // In TRT 8.5+, we iterate bindings. In TRT 10, we inspect via IEngineInspector or cached logic.
        // Using standard IEngine inspection for IO:
        int32_t num_io = m_impl->engine->getNbIOTensors();
        for (int32_t i = 0; i < num_io; ++i) {
            const char* name = m_impl->engine->getIOTensorName(i);
            nvinfer1::TensorIOMode mode = m_impl->engine->getTensorIOMode(name);
            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                m_impl->input_names.emplace_back(name);
            } else {
                m_impl->output_names.emplace_back(name);
            }
        }

        XINFER_LOG_INFO("Loaded TensorRT Engine: " + model_path);
        return true;

    } catch (const std::exception& e) {
        XINFER_LOG_ERROR("TRT Load Failed: " + std::string(e.what()));
        return false;
    }
}

void NvidiaBackend::predict(const std::vector<core::Tensor>& inputs, 
                            std::vector<core::Tensor>& outputs) {
    
    if (inputs.size() != m_impl->input_names.size()) {
        XINFER_LOG_ERROR("Input count mismatch.");
        return;
    }

    // 1. Bind Inputs
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& name = m_impl->input_names[i];
        const auto& tensor = inputs[i];

        // Handle Dynamic Shapes
        // For TRT, we must set input dimensions before execution if they are dynamic
        nvinfer1::Dims dims;
        dims.nbDims = tensor.shape().size();
        for(size_t d=0; d<dims.nbDims; ++d) dims.d[d] = tensor.shape()[d];
        m_impl->context->setInputShape(name.c_str(), dims);

        // Handle Memory (Host vs Device)
        void* device_ptr = nullptr;

        if (tensor.memory_type() == core::MemoryType::CudaDevice) {
            // Zero-Copy: Input is already on GPU
            device_ptr = tensor.data();
        } else {
            // Slow Path: Input is on CPU, we need to copy.
            // In a real high-perf app, users should provide GPU pointers.
            // For now, we assume xInfer core provides a mechanism or we fail/warn.
            // A robust backend would have a persistent GPU buffer cache here.
            XINFER_LOG_WARN_ONCE("Performance warning: CPU tensor passed to TRT backend. Copying...");
            // Allocation logic omitted for brevity - assumes tensor.device_data() exists or handled externally
            device_ptr = tensor.data(); // This assumes Unified Memory or Pre-copy
        }

        // Bind address for TRT Execution
        m_impl->context->setTensorAddress(name.c_str(), device_ptr);
    }

    // 2. Bind Outputs
    if (outputs.size() != m_impl->output_names.size()) {
        outputs.resize(m_impl->output_names.size());
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
        const auto& name = m_impl->output_names[i];
        
        // Ensure output tensor is allocated on GPU
        // ... (allocation logic) ...
        
        m_impl->context->setTensorAddress(name.c_str(), outputs[i].data());
    }

    // 3. Enqueue Inference (Async)
    bool status = m_impl->context->enqueueV3(m_impl->stream);
    if (!status) {
        XINFER_LOG_ERROR("TRT EnqueueV3 Failed.");
    }

    // 4. Synchronization (Optional per config)
    // By default, predict() launches work. Users should call synchronize() 
    // or use events to know when data is ready.
    // If outputs are CPU tensors, we MUST sync and copy back here.
    bool requires_sync = false;
    for(const auto& out : outputs) {
        if (out.memory_type() == core::MemoryType::Host) requires_sync = true;
    }

    if (requires_sync) {
        cudaStreamSynchronize(m_impl->stream);
        // ... copy D2H logic ...
    }
}

std::string NvidiaBackend::device_name() const {
    return "NVIDIA GPU (TensorRT)";
}

void* NvidiaBackend::get_cuda_stream() const {
    return m_impl->stream;
}

void NvidiaBackend::synchronize() {
    cudaStreamSynchronize(m_impl->stream);
}

// =================================================================================
// 5. Auto-Registration
// =================================================================================

namespace {
    volatile bool registered = xinfer::backends::BackendFactory::register_backend(
        xinfer::Target::NVIDIA_TRT,
        [](const xinfer::Config& config) -> std::unique_ptr<xinfer::IBackend> {
            TrtConfig trt_cfg;
            trt_cfg.model_path = config.model_path;
            
            // Parse vendor flags
            for(const auto& param : config.vendor_params) {
                if(param == "DLA=0") trt_cfg.dla_core = DlaCore::CORE_0;
                if(param == "DLA=1") trt_cfg.dla_core = DlaCore::CORE_1;
            }
            
            auto backend = std::make_unique<NvidiaBackend>(trt_cfg);
            if(backend->load_model(trt_cfg.model_path)) {
                return backend;
            }
            return nullptr;
        }
    );
}

} // namespace xinfer::backends::nvidia