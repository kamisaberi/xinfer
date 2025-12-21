#include <xinfer/backends/amd_ryzen_ai/backend.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/utils.h>
#include <xinfer/backends/backend_factory.h> // For auto-registration

#include <iostream>
#include <vector>
#include <stdexcept>
#include <mutex>

// --- ONNX Runtime Headers (for Vitis AI EP) ---
#include <onnxruntime_cxx_api.h>

// --- XRT Headers (Optional, for Native mode) ---
// #include <xrt/xrt_device.h>
// #include <xrt/xrt_kernel.h>
// #include <xrt/xrt_bo.h>

namespace xinfer::backends::ryzen_ai {

// =================================================================================
// 1. PImpl Implementation
// =================================================================================

struct RyzenAIBackend::Impl {
    // --- Common Config ---
    RyzenAIConfig config;

    // --- Vitis AI EP (ONNX Runtime) Members ---
    std::unique_ptr<Ort::Env> ort_env;
    std::unique_ptr<Ort::Session> ort_session;
    std::unique_ptr<Ort::AllocatorWithDefaultOptions> allocator;

    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names;
    std::vector<std::string> input_node_names_alloc; // Storage for strings
    std::vector<std::string> output_node_names_alloc;

    // --- Native XRT Members (Placeholder) ---
    // xrt::device device;
    // xrt::kernel kernel;

    explicit Impl(const RyzenAIConfig& cfg) : config(cfg) {
        // Initialize basic ORT environment
        ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "xinfer_ryzen_ai");
        allocator = std::make_unique<Ort::AllocatorWithDefaultOptions>();
    }

    // --------------------------------------------------------------------------
    // Initialization: Vitis AI Execution Provider
    // --------------------------------------------------------------------------
    void init_vitis_ai_ep() {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(config.cpu_threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Configure Vitis AI Provider Options
        // These keys map to the Ryzen AI specific configuration
        std::vector<const char*> keys;
        std::vector<const char*> values;

        // Path to the vaip_config.json which defines DPU architecture
        std::string config_file_arg = config.config_file_path;
        if (!config_file_arg.empty()) {
            keys.push_back("config_file");
            values.push_back(config_file_arg.c_str());
        }

        // Cache directory for compiled models
        keys.push_back("cacheDir");
        values.push_back("./cache/ryzen_ai");
        keys.push_back("cacheKey");
        values.push_back("xinfer_model");

        try {
            // Append Vitis AI Execution Provider
            // Note: The specific function name depends on the ORT version provided by AMD
            // OrtSessionOptionsAppendExecutionProvider_VitisAI(session_options, keys.data(), values.data(), keys.size());

            // Generic fallback for standard ORT builds supporting VitisAI
             session_options.AppendExecutionProvider("VitisAI", keys.data(), values.data(), keys.size());

            XINFER_LOG_INFO("Vitis AI Execution Provider attached.");
        } catch (const std::exception& e) {
            XINFER_LOG_WARN(std::string("Failed to append Vitis AI EP: ") + e.what() + ". Falling back to CPU.");
        }

        // Load the model
#ifdef _WIN32
        std::wstring w_model_path(config.model_path.begin(), config.model_path.end());
        ort_session = std::make_unique<Ort::Session>(*ort_env, w_model_path.c_str(), session_options);
#else
        ort_session = std::make_unique<Ort::Session>(*ort_env, config.model_path.c_str(), session_options);
#endif

        // Resolve IO Names
        size_t num_inputs = ort_session->GetInputCount();
        for (size_t i = 0; i < num_inputs; i++) {
            auto name = ort_session->GetInputNameAllocated(i, *allocator);
            input_node_names_alloc.push_back(name.get());
            input_node_names.push_back(input_node_names_alloc.back().c_str());
        }

        size_t num_outputs = ort_session->GetOutputCount();
        for (size_t i = 0; i < num_outputs; i++) {
            auto name = ort_session->GetOutputNameAllocated(i, *allocator);
            output_node_names_alloc.push_back(name.get());
            output_node_names.push_back(output_node_names_alloc.back().c_str());
        }
    }

    // --------------------------------------------------------------------------
    // Initialization: Native XRT (Embedded/Linux)
    // --------------------------------------------------------------------------
    void init_native_xrt() {
        XINFER_LOG_WARN("Native XRT mode for Ryzen AI is not yet fully implemented in this version.");
        // Logic would involve:
        // device = xrt::device(0);
        // auto uuid = device.load_xclbin(config.xclbin_path);
        // kernel = xrt::kernel(device, uuid, "dpu_kernel_name");
    }

    // --------------------------------------------------------------------------
    // Execution: Vitis AI EP
    // --------------------------------------------------------------------------
    void predict_ort(const std::vector<core::Tensor>& inputs,
                     std::vector<core::Tensor>& outputs) {

        std::vector<Ort::Value> input_tensors;
        std::vector<Ort::Value> output_tensors;

        // Wrap xInfer Tensors into ORT Tensors
        for (const auto& input : inputs) {
            Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            // Assume float for simplicity, in real implementation dispatch based on input.dtype()
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                mem_info,
                const_cast<float*>(static_cast<const float*>(input.data())),
                input.size(),
                input.shape().data(),
                input.shape().size()
            ));
        }

        // Run Inference
        auto ort_outputs = ort_session->Run(
            Ort::RunOptions{nullptr},
            input_node_names.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_node_names.data(),
            output_node_names.size()
        );

        // Copy back to xInfer outputs
        // (If outputs vector is pre-allocated, we copy. If not, we should resize it)
        if (outputs.size() != ort_outputs.size()) {
            outputs.resize(ort_outputs.size());
        }

        for (size_t i = 0; i < ort_outputs.size(); ++i) {
             // Retrieve type info and shape from ort_outputs[i] and populate outputs[i]
             // For this snippet, we assume a raw memcpy to the existing buffer
             float* out_ptr = ort_outputs[i].GetTensorMutableData<float>();

             // In a real implementation: outputs[i].reshape(...)
             size_t byte_size = outputs[i].size() * sizeof(float);
             std::memcpy(outputs[i].data(), out_ptr, byte_size);
        }
    }
};

// =================================================================================
// 2. Public API Implementation
// =================================================================================

RyzenAIBackend::RyzenAIBackend(const RyzenAIConfig& config)
    : m_config(config), m_impl(std::make_unique<Impl>(config)) {
}

RyzenAIBackend::~RyzenAIBackend() = default;

bool RyzenAIBackend::load_model(const std::string& model_path) {
    try {
        if (m_config.runtime_type == RuntimeType::VITIS_AI_EP) {
            // Update path in case it changed from constructor
            m_impl->config.model_path = model_path;
            m_impl->init_vitis_ai_ep();
        } else {
            m_impl->init_native_xrt();
        }
        return true;
    } catch (const std::exception& e) {
        XINFER_LOG_ERROR("Ryzen AI Load Failed: " + std::string(e.what()));
        return false;
    }
}

void RyzenAIBackend::predict(const std::vector<core::Tensor>& inputs,
                             std::vector<core::Tensor>& outputs) {
    if (m_config.runtime_type == RuntimeType::VITIS_AI_EP) {
        m_impl->predict_ort(inputs, outputs);
    } else {
        // m_impl->predict_xrt(inputs, outputs);
    }
}

std::string RyzenAIBackend::device_name() const {
    return "AMD Ryzen AI (XDNA)";
}

double RyzenAIBackend::get_npu_load() const {
    // Placeholder: This typically requires querying the Windows Task Manager API
    // or the XDNA driver sysfs on Linux.
    return 0.0;
}

// =================================================================================
// 3. Auto-Registration
// =================================================================================

namespace {
    volatile bool registered = xinfer::backends::BackendFactory::register_backend(
        xinfer::Target::AMD_RYZEN_AI,
        [](const xinfer::Config& config) -> std::unique_ptr<xinfer::IBackend> {
            RyzenAIConfig ryzen_cfg;
            ryzen_cfg.model_path = config.model_path;

            // Parse common vendor params
            for(const auto& param : config.vendor_params) {
                if(param.find("CONFIG_FILE=") != std::string::npos) {
                     ryzen_cfg.config_file_path = param.substr(12);
                }
            }

            // Default to Vitis EP for broad compatibility
            ryzen_cfg.runtime_type = RuntimeType::VITIS_AI_EP;

            auto backend = std::make_unique<RyzenAIBackend>(ryzen_cfg);
            if(backend->load_model(ryzen_cfg.model_path)) {
                return backend;
            }
            return nullptr;
        }
    );
}

} // namespace xinfer::backends::ryzen_ai