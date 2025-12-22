#include <xinfer/backends/qualcomm_qnn/backend.h>
#include <xinfer/core/logging.h>
#include <xinfer/backends/backend_factory.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <dlfcn.h> // For loading libQnnHtp.so

// --- QNN Headers ---
#include "QnnInterface.h"
#include "QnnBackend.h"
#include "QnnDevice.h"
#include "QnnContext.h"
#include "QnnGraph.h"
#include "QnnTensor.h"
#include "QnnLog.h"

namespace xinfer::backends::qnn {

// =================================================================================
// 1. PImpl Implementation
// =================================================================================

struct QnnBackend::Impl {
    QnnConfig config;

    // Library Handles
    void* backend_lib_handle = nullptr;

    // QNN Interface Function Pointers
    QnnInterface_t qnn_interface;

    // QNN Objects
    Qnn_BackendHandle_t backend_handle = nullptr;
    Qnn_DeviceHandle_t device_handle = nullptr;
    Qnn_ContextHandle_t context_handle = nullptr;
    Qnn_GraphHandle_t graph_handle = nullptr;

    // Tensor Descriptors (Cached for execution)
    std::vector<Qnn_Tensor_t> input_tensors;
    std::vector<Qnn_Tensor_t> output_tensors;

    explicit Impl(const QnnConfig& cfg) : config(cfg) {}

    ~Impl() {
        // Shutdown sequence order matters
        if (graph_handle)   qnn_interface.graphFree(graph_handle);
        if (context_handle) qnn_interface.contextFree(context_handle, nullptr);
        if (device_handle)  qnn_interface.deviceFree(device_handle);
        if (backend_handle) qnn_interface.backendFree(backend_handle);
        
        if (backend_lib_handle) dlclose(backend_lib_handle);
    }

    // --- Helper: Load Dynamic Library ---
    bool load_backend_lib() {
        // Default paths if not provided
        std::string lib_path = config.qnn_backend_lib_path;
        if (lib_path.empty()) {
            if (config.backend_type == QnnBackendType::HTP) lib_path = "libQnnHtp.so";
            else if (config.backend_type == QnnBackendType::GPU) lib_path = "libQnnGpu.so";
            else lib_path = "libQnnCpu.so";
        }

        XINFER_LOG_INFO("Loading QNN Backend Lib: " + lib_path);
        
        backend_lib_handle = dlopen(lib_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (!backend_lib_handle) {
            XINFER_LOG_ERROR("Failed to dlopen QNN lib: " + std::string(dlerror()));
            return false;
        }

        // Get the Provider Function
        using GetProvidersFn = Qnn_ErrorHandle_t (*)(const QnnInterface_t** providerList, uint32_t* numProviders);
        auto get_providers = reinterpret_cast<GetProvidersFn>(dlsym(backend_lib_handle, "QnnInterface_getProviders"));

        if (!get_providers) {
            XINFER_LOG_ERROR("Symbol QnnInterface_getProviders not found.");
            return false;
        }

        const QnnInterface_t* providers = nullptr;
        uint32_t num_providers = 0;
        
        if (get_providers(&providers, &num_providers) != QNN_SUCCESS || num_providers == 0) {
            XINFER_LOG_ERROR("Failed to get QNN Interface providers.");
            return false;
        }

        // We assume the first provider is the one we want (e.g. QNN_HTP_INTERFACE_PROVIDER_NAME)
        qnn_interface = providers[0];
        return true;
    }

    // --- Helper: Read Binary File ---
    std::vector<char> read_binary(const std::string& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) return {};
        size_t size = file.tellg();
        file.seekg(0);
        std::vector<char> buf(size);
        file.read(buf.data(), size);
        return buf;
    }
};

// =================================================================================
// 2. Public API Implementation
// =================================================================================

QnnBackend::QnnBackend(const QnnConfig& config)
    : m_config(config), m_impl(std::make_unique<Impl>(config)) {
}

QnnBackend::~QnnBackend() = default;

bool QnnBackend::load_model(const std::string& model_path) {
    if (!m_impl->load_backend_lib()) return false;

    // 1. Initialize Backend
    // In a real app, you might provide a log callback here
    if (m_impl->qnn_interface.backendCreate(nullptr, (const QnnBackend_Config_t**)nullptr, &m_impl->backend_handle) != QNN_SUCCESS) {
        XINFER_LOG_ERROR("QNN Backend Create failed.");
        return false;
    }

    // 2. Create Device
    // For HTP, this initializes the DSP.
    if (m_impl->qnn_interface.deviceCreate(m_impl->backend_handle, nullptr, &m_impl->device_handle) != QNN_SUCCESS) {
        XINFER_LOG_ERROR("QNN Device Create failed.");
        return false;
    }

    // 3. Create Context from Binary
    std::vector<char> bin_data = m_impl->read_binary(model_path);
    if (bin_data.empty()) {
        XINFER_LOG_ERROR("Failed to read context binary: " + model_path);
        return false;
    }

    QnnContext_Config_t context_cfg;
    context_cfg.option = QNN_CONTEXT_CONFIG_OPTION_UNDEFINED; // Terminate list
    const QnnContext_Config_t* context_configs[] = { &context_cfg, nullptr };

    if (m_impl->qnn_interface.contextCreateFromBinary(
            m_impl->backend_handle,
            m_impl->device_handle,
            (const QnnContext_Config_t**)nullptr, // configs
            reinterpret_cast<const void*>(bin_data.data()),
            bin_data.size(),
            &m_impl->context_handle,
            nullptr // profile handle
        ) != QNN_SUCCESS) {
        XINFER_LOG_ERROR("QNN Context CreateFromBinary failed.");
        return false;
    }

    // 4. Retrieve Graph
    // We assume the binary contains exactly one graph.
    // To be robust, we should query memory info, but 'graphRetrieve' works if we know the graph name 
    // or if we use NULL to retrieve the first one (depends on backend version).
    // QNN APIs often require iterating graphs. 
    // Here we use a simplified assumption that the context has 1 graph.
    
    // NOTE: QNN doesn't have a simple "GetFirstGraph" API. 
    // You usually need to know the graph name passed during compilation.
    // For xInfer standard, we assume the user provides graph_name in config or we try "xinfer_graph".
    
    // Placeholder: We assume graph retrieval logic or re-compilation flows. 
    // In many QNN deployments, 'contextCreateFromBinary' is enough to set up the executor 
    // if you inspect the context.
    
    // Attempt to retrieve graph named "qnn_model" (Default for many converters)
    if (m_impl->qnn_interface.graphRetrieve(
            m_impl->context_handle, 
            "qnn_model", // This MUST match the name used during qnn-context-binary-generator
            &m_impl->graph_handle
        ) != QNN_SUCCESS) {
        
        // Try fallback
        XINFER_LOG_WARN("Could not retrieve graph 'qnn_model'. Trying fallback...");
        // In robust code, use QnnContext_getGraphNames if available in helper libs
        return false;
    }

    // 5. Prepare Tensor Structures
    // In a real implementation, we would call QnnGraph_getInputs/Outputs to populate m_impl->input_tensors.
    // This allows us to bind memory later.
    
    XINFER_LOG_INFO("Loaded QNN Context: " + model_path);
    return true;
}

void QnnBackend::predict(const std::vector<core::Tensor>& inputs, 
                         std::vector<core::Tensor>& outputs) {
    
    // 1. Map Inputs
    // We need to wrap xInfer pointers into Qnn_Tensor_t structs.
    // This is similar to OpenVINO wrapper but manual.
    
    // NOTE: This implementation assumes m_impl->input_tensors was populated during load_model.
    // Since QNN struct setup is verbose, we simulate the execution call here.

    Qnn_Tensor_t* inputs_ptr = nullptr; // Would point to populated structs
    uint32_t num_inputs = inputs.size();
    
    Qnn_Tensor_t* outputs_ptr = nullptr;
    uint32_t num_outputs = outputs.size();

    // 2. Execute
    Qnn_ErrorHandle_t err = m_impl->qnn_interface.graphExecute(
        m_impl->graph_handle,
        inputs_ptr,
        num_inputs,
        outputs_ptr,
        num_outputs,
        nullptr, // profile handle
        nullptr  // signal handle
    );

    if (err != QNN_SUCCESS) {
        XINFER_LOG_ERROR("QNN Graph Execution Failed. Error: " + std::to_string(err));
    }
}

std::string QnnBackend::device_name() const {
    if (m_config.backend_type == QnnBackendType::HTP) return "Qualcomm Hexagon NPU (HTP)";
    if (m_config.backend_type == QnnBackendType::GPU) return "Qualcomm Adreno GPU";
    return "Qualcomm CPU";
}

void QnnBackend::set_performance_mode(HtpPerformanceMode mode) {
    // Requires QnnDevice_setConfig API calls
    // e.g., QNN_DEVICE_PROPERTY_PERF_MODE
}

// =================================================================================
// 3. Auto-Registration
// =================================================================================

namespace {
    volatile bool registered = xinfer::backends::BackendFactory::register_backend(
        xinfer::Target::QUALCOMM_QNN,
        [](const xinfer::Config& config) -> std::unique_ptr<xinfer::IBackend> {
            QnnConfig qnn_cfg;
            qnn_cfg.model_path = config.model_path;
            
            // Parse vendor flags
            for(const auto& param : config.vendor_params) {
                if(param == "BACKEND=HTP") qnn_cfg.backend_type = QnnBackendType::HTP;
                if(param == "BACKEND=GPU") qnn_cfg.backend_type = QnnBackendType::GPU;
                if(param == "PERF=BURST") qnn_cfg.performance_mode = HtpPerformanceMode::BURST;
            }
            
            auto backend = std::make_unique<QnnBackend>(qnn_cfg);
            if(backend->load_model(qnn_cfg.model_path)) {
                return backend;
            }
            return nullptr;
        }
    );
}

} // namespace xinfer::backends::qnn