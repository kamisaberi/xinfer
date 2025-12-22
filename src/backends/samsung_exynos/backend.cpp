#include <xinfer/backends/samsung_exynos/backend.h>
#include <xinfer/core/logging.h>
#include <xinfer/backends/backend_factory.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <mutex>

// --- Samsung ENN Headers ---
// Note: Actual header names vary by SDK version (e.g. EnnApi.h, EnnInterface.h)
#include "EnnApi.h" 

namespace xinfer::backends::exynos {

// =================================================================================
// 1. PImpl Implementation
// =================================================================================

struct ExynosBackend::Impl {
    ExynosConfig config;

    // ENN Handles
    EnnContextId context_id = 0;
    EnnModelId model_id = 0;
    
    // Buffer info (ENN Requires explicit buffer allocation/registration)
    std::vector<EnnBufferInfo> input_buffers;
    std::vector<EnnBufferInfo> output_buffers;
    
    // Model Memory (To keep binary alive)
    std::vector<char> model_data;

    explicit Impl(const ExynosConfig& cfg) : config(cfg) {}

    ~Impl() {
        if (model_id) EnnUnloadModel(context_id, model_id);
        if (context_id) EnnDeinitialize();
    }

    // --- Helper: Convert EnnReturn to string ---
    std::string err_to_str(EnnReturn ret) {
        if (ret == ENN_RET_SUCCESS) return "SUCCESS";
        return "ERR_CODE_" + std::to_string(ret);
    }
    
    // --- Helper: Map Preference ---
    EnnPreference map_pref() {
        switch (config.power_mode) {
            case EnnPowerMode::BOOST: return ENN_PREF_PERFORMANCE_MODE;
            case EnnPowerMode::LOW_POWER: return ENN_PREF_POWER_MODE;
            case EnnPowerMode::SUSTAINED: default: return ENN_PREF_NORMAL_MODE;
        }
    }
};

// =================================================================================
// 2. Public API Implementation
// =================================================================================

ExynosBackend::ExynosBackend(const ExynosConfig& config)
    : m_config(config), m_impl(std::make_unique<Impl>(config)) {
}

ExynosBackend::~ExynosBackend() = default;

bool ExynosBackend::load_model(const std::string& model_path) {
    // 1. Initialize ENN
    EnnReturn ret = EnnInitialize();
    if (ret != ENN_RET_SUCCESS) {
        XINFER_LOG_ERROR("EnnInitialize failed: " + m_impl->err_to_str(ret));
        return false;
    }

    // 2. Open Context
    ret = EnnOpenContext(&m_impl->context_id);
    if (ret != ENN_RET_SUCCESS) {
        XINFER_LOG_ERROR("EnnOpenContext failed.");
        return false;
    }

    // 3. Load Binary File
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        XINFER_LOG_ERROR("Failed to open ENN model: " + model_path);
        return false;
    }
    size_t size = file.tellg();
    file.seekg(0);
    m_impl->model_data.resize(size);
    file.read(m_impl->model_data.data(), size);

    // 4. Load Model into NPU
    // ENN usually takes the buffer pointer
    ret = EnnLoadModel(m_impl->context_id, 
                       reinterpret_cast<void*>(m_impl->model_data.data()), 
                       size, 
                       &m_impl->model_id);
                       
    if (ret != ENN_RET_SUCCESS) {
        XINFER_LOG_ERROR("EnnLoadModel failed: " + m_impl->err_to_str(ret));
        return false;
    }

    // 5. Commit/Prepare
    // This step allocates internal NPU resources and parses graph structure
    ret = EnnCommit(m_impl->model_id);
    if (ret != ENN_RET_SUCCESS) {
        XINFER_LOG_ERROR("EnnCommit failed.");
        return false;
    }

    // 6. Set Performance Mode
    EnnSetPreference(m_impl->context_id, m_impl->map_pref());

    // 7. Query Buffer Requirements (Inputs/Outputs)
    uint32_t num_inputs, num_outputs;
    EnnGetModelInfo(m_impl->model_id, ENN_MODEL_INFO_NUM_INPUTS, &num_inputs);
    EnnGetModelInfo(m_impl->model_id, ENN_MODEL_INFO_NUM_OUTPUTS, &num_outputs);

    m_impl->input_buffers.resize(num_inputs);
    m_impl->output_buffers.resize(num_outputs);

    // Get buffer sizes needed by NPU
    for(uint32_t i=0; i<num_inputs; ++i) {
        EnnGetBufferInfo(m_impl->model_id, ENN_DIR_IN, i, &m_impl->input_buffers[i]);
    }
    for(uint32_t i=0; i<num_outputs; ++i) {
        EnnGetBufferInfo(m_impl->model_id, ENN_DIR_OUT, i, &m_impl->output_buffers[i]);
    }

    XINFER_LOG_INFO("Loaded Exynos ENN Model: " + model_path);
    return true;
}

void ExynosBackend::predict(const std::vector<core::Tensor>& inputs, 
                            std::vector<core::Tensor>& outputs) {
    
    // 1. Prepare Inputs
    // ENN API usually requires binding specific EnnBuffer structures
    
    if (inputs.size() != m_impl->input_buffers.size()) {
        XINFER_LOG_ERROR("Input count mismatch.");
        return;
    }

    std::vector<EnnMemReq> mem_reqs;
    
    for (size_t i = 0; i < inputs.size(); ++i) {
        EnnMemReq req;
        req.dir = ENN_DIR_IN;
        req.index = i;
        
        // Check for Zero-Copy (ION Buffer)
        // If xInfer Tensor wraps an ION fd, pass it directly.
        // Otherwise, pass virtual address (driver handles cache/copy).
        req.va = const_cast<void*>(inputs[i].data());
        req.size = inputs[i].size() * sizeof(float);
        
        mem_reqs.push_back(req);
    }

    // 2. Prepare Outputs
    if (outputs.size() != m_impl->output_buffers.size()) {
        outputs.resize(m_impl->output_buffers.size());
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
        // Resize output tensor if empty
        size_t needed_size = m_impl->output_buffers[i].size;
        if (outputs[i].empty()) {
             // Basic resizing (shape info needs to come from model query ideally)
             outputs[i].resize({1, (int64_t)(needed_size/sizeof(float))}, core::DataType::kFLOAT);
        }

        EnnMemReq req;
        req.dir = ENN_DIR_OUT;
        req.index = i;
        req.va = outputs[i].data();
        req.size = needed_size;
        
        mem_reqs.push_back(req);
    }

    // 3. Execute
    EnnReturn ret = EnnExecute(m_impl->context_id, m_impl->model_id, mem_reqs.data(), mem_reqs.size());

    if (ret != ENN_RET_SUCCESS) {
        XINFER_LOG_ERROR("EnnExecute failed: " + m_impl->err_to_str(ret));
    }
    
    // 4. (Optional) Sync
    // If using ION buffers, synchronization is implicit or handled by dma_fence.
    // If using malloc buffers, EnnExecute typically blocks until copy back is done.
}

std::string ExynosBackend::device_name() const {
    return "Samsung Exynos NPU";
}

void ExynosBackend::set_power_mode(EnnPowerMode mode) {
    m_config.power_mode = mode;
    EnnSetPreference(m_impl->context_id, m_impl->map_pref());
}

// =================================================================================
// 3. Auto-Registration
// =================================================================================

namespace {
    volatile bool registered = xinfer::backends::BackendFactory::register_backend(
        xinfer::Target::SAMSUNG_EXYNOS,
        [](const xinfer::Config& config) -> std::unique_ptr<xinfer::IBackend> {
            ExynosConfig enn_cfg;
            enn_cfg.model_path = config.model_path;
            
            // Parse vendor flags
            for(const auto& param : config.vendor_params) {
                if(param == "POWER=BOOST") enn_cfg.power_mode = EnnPowerMode::BOOST;
                if(param == "POWER=LOW") enn_cfg.power_mode = EnnPowerMode::LOW_POWER;
            }
            
            auto backend = std::make_unique<ExynosBackend>(enn_cfg);
            if(backend->load_model(enn_cfg.model_path)) {
                return backend;
            }
            return nullptr;
        }
    );
}

} // namespace xinfer::backends::exynos