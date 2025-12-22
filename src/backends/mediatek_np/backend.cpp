#include <xinfer/backends/mediatek_np/backend.h>
#include <xinfer/core/logging.h>
#include <xinfer/backends/backend_factory.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <algorithm>

// --- NeuroPilot Headers ---
#include <neuron/neuron.h>

namespace xinfer::backends::mediatek {

// =================================================================================
// 1. PImpl Implementation
// =================================================================================

struct MediaTekBackend::Impl {
    MediaTekConfig config;

    // Neuron Handles
    NeuronModel* model = nullptr;
    NeuronCompilation* compilation = nullptr;
    NeuronExecution* execution = nullptr;
    
    // Memory handles
    // In a real implementation, we need persistent memory handles if using shared memory
    std::vector<NeuronMemory*> input_mem_handles;
    std::vector<NeuronMemory*> output_mem_handles;

    explicit Impl(const MediaTekConfig& cfg) : config(cfg) {}

    ~Impl() {
        if (execution) NeuronExecution_free(execution);
        if (compilation) NeuronCompilation_free(compilation);
        if (model) NeuronModel_free(model);
        
        // Clean up memory handles
        for (auto* mem : input_mem_handles) NeuronMemory_free(mem);
        for (auto* mem : output_mem_handles) NeuronMemory_free(mem);
    }
    
    // --- Helper: Check Neuron Status ---
    bool check_status(int status, const std::string& msg) {
        if (status != NEURON_NO_ERROR) {
            XINFER_LOG_ERROR("NeuroPilot Error (" + std::to_string(status) + "): " + msg);
            return false;
        }
        return true;
    }
};

// =================================================================================
// 2. Public API Implementation
// =================================================================================

MediaTekBackend::MediaTekBackend(const MediaTekConfig& config)
    : m_config(config), m_impl(std::make_unique<Impl>(config)) {
}

MediaTekBackend::~MediaTekBackend() = default;

bool MediaTekBackend::load_model(const std::string& model_path) {
    // 1. Read the compiled model binary (.dla / .pte)
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        XINFER_LOG_ERROR("Failed to open model file: " + model_path);
        return false;
    }

    size_t size = file.tellg();
    file.seekg(0);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);

    // 2. Restore Model from Buffer
    int status = NeuronModel_restoreFromBuffer(reinterpret_cast<void*>(buffer.data()), size, &m_impl->model);
    if (!m_impl->check_status(status, "Failed to restore model from buffer")) return false;

    // 3. Create Compilation
    status = NeuronCompilation_create(m_impl->model, &m_impl->compilation);
    if (!m_impl->check_status(status, "Failed to create compilation")) return false;

    // 4. Set Preference (Power vs Latency)
    int32_t pref_code = NEURON_PREFER_FAST_SINGLE_ANSWER; // Default
    switch(m_config.preference) {
        case NeuronPreference::PREFER_LOW_POWER: pref_code = NEURON_PREFER_LOW_POWER; break;
        case NeuronPreference::PREFER_SUSTAINED_SPEED: pref_code = NEURON_PREFER_SUSTAINED_SPEED; break;
        // ... mappings
    }
    NeuronCompilation_setPreference(m_impl->compilation, pref_code);

    // 5. Finish Compilation (Finalizes layout on APU)
    status = NeuronCompilation_finish(m_impl->compilation);
    if (!m_impl->check_status(status, "Failed to finish compilation")) return false;

    XINFER_LOG_INFO("Loaded NeuroPilot Model: " + model_path);
    return true;
}

void MediaTekBackend::predict(const std::vector<core::Tensor>& inputs, 
                              std::vector<core::Tensor>& outputs) {
    
    // 1. Create Execution Context (if not exists)
    // Note: Re-using execution objects is faster than creating new ones every frame
    if (!m_impl->execution) {
        NeuronExecution_create(m_impl->compilation, &m_impl->execution);
    }

    // 2. Set Inputs
    // For this implementation, we assume raw pointer setting (SetInputFromMemory)
    // In a zero-copy scenario (ION Buffer), use NeuronMemory_createFromFd
    for (size_t i = 0; i < inputs.size(); ++i) {
        NeuronOperandType type; 
        type.type = NEURON_TENSOR_FLOAT32; // Simplified, usually query model for type
        
        // Pass the raw data pointer to Neuron
        // WARNING: Neuron expects the pointer to remain valid during compute
        int status = NeuronExecution_setInput(m_impl->execution, i, &type, 
                                              const_cast<void*>(inputs[i].data()), 
                                              inputs[i].size() * sizeof(float));
        
        if (!m_impl->check_status(status, "SetInput failed index " + std::to_string(i))) return;
    }

    // 3. Set Outputs
    // Ensure output tensors are allocated
    // (In real implementation, query model output dimensions to resize vector)
    for (size_t i = 0; i < outputs.size(); ++i) {
         if (outputs[i].empty()) {
             // Placeholder resize (user usually provides correct shapes or we query model)
             outputs[i].resize({1, 1000}, core::DataType::kFLOAT);
         }

         NeuronOperandType type;
         type.type = NEURON_TENSOR_FLOAT32;

         int status = NeuronExecution_setOutput(m_impl->execution, i, &type,
                                                outputs[i].data(),
                                                outputs[i].size() * sizeof(float));
         
         if (!m_impl->check_status(status, "SetOutput failed index " + std::to_string(i))) return;
    }

    // 4. Compute
    int status = NeuronExecution_compute(m_impl->execution);
    if (!m_impl->check_status(status, "Compute failed")) return;

    // Neuron execution is synchronous by default unless using event objects
}

std::string MediaTekBackend::device_name() const {
    return "MediaTek APU (Neuron)";
}

void MediaTekBackend::set_boost_mode(bool enable, int duration_ms) {
    // This typically uses a separate library (libapu_mdw) or Neuron extensions
    // Not part of the standard Neuron C API, implementation skipped.
    XINFER_LOG_INFO("MediaTek Boost Mode: " + std::string(enable ? "ON" : "OFF"));
}

// =================================================================================
// 3. Auto-Registration
// =================================================================================

namespace {
    volatile bool registered = xinfer::backends::BackendFactory::register_backend(
        xinfer::Target::MEDIATEK_NEUROPILOT,
        [](const xinfer::Config& config) -> std::unique_ptr<xinfer::IBackend> {
            MediaTekConfig mtk_cfg;
            mtk_cfg.model_path = config.model_path;
            
            // Parse vendor flags
            for(const auto& param : config.vendor_params) {
                if(param == "PREF=LOW_POWER") mtk_cfg.preference = NeuronPreference::PREFER_LOW_POWER;
                if(param == "PREF=LATENCY") mtk_cfg.preference = NeuronPreference::PREFER_FAST_SINGLE_ANSWER;
            }
            
            auto backend = std::make_unique<MediaTekBackend>(mtk_cfg);
            if(backend->load_model(mtk_cfg.model_path)) {
                return backend;
            }
            return nullptr;
        }
    );
}

} // namespace xinfer::backends::mediatek