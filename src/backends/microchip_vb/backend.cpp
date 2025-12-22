#include <xinfer/backends/microchip_vb/backend.h>
#include <xinfer/core/logging.h>
#include <xinfer/backends/backend_factory.h>

#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>

// --- VectorBlox SDK Headers ---
#include "vbx_cnn_api.h"

namespace xinfer::backends::microchip {

// =================================================================================
// 1. PImpl Implementation
// =================================================================================

struct VectorBloxBackend::Impl {
    VectorBloxConfig config;

    // The Raw Model Blob (Loaded from disk into DDR)
    std::vector<uint8_t> model_blob;
    
    // VBX Runtime State
    vbx_cnn_t* cnn = nullptr;
    vbx_cnn_model_info_t* model_info = nullptr;
    
    // IO Buffers (Pointers into the model_blob or allocated buffers)
    void* input_buffer_ptr = nullptr;
    void* output_buffer_ptr = nullptr;

    explicit Impl(const VectorBloxConfig& cfg) : config(cfg) {}

    ~Impl() {
        // VBX C-API cleanup if required
        // Usually vbx_cnn_t is just a struct pointer, main cleanup is memory
    }

    // --------------------------------------------------------------------------
    // Helper: Cache Management (Platform Specific)
    // --------------------------------------------------------------------------
    void flush_cache(void* addr, size_t size) {
        // On Linux (PolarFire SoC), this might be handled by the kernel driver 
        // or a specific asm instruction 'cbo.clean'.
        // For standard user-space, we assume the VBX driver handles it or we rely on 
        // non-cached memory mapping (O_SYNC).
        
        // Placeholder for the "RISC-V clean" logic
        // __asm__ volatile ("fence rw,rw"); 
    }

    void invalidate_cache(void* addr, size_t size) {
        // Placeholder for "RISC-V invalidate" logic
    }
};

// =================================================================================
// 2. Public API Implementation
// =================================================================================

VectorBloxBackend::VectorBloxBackend(const VectorBloxConfig& config)
    : m_config(config), m_impl(std::make_unique<Impl>(config)) {
}

VectorBloxBackend::~VectorBloxBackend() = default;

bool VectorBloxBackend::load_model(const std::string& model_path) {
    // 1. Initialize Hardware
    // vbx_cnn_init usually takes a control register base address
    // In a Linux env, this maps to /dev/mem or a UIO device.
    if (vbx_cnn_init(nullptr) != 0) {
        XINFER_LOG_ERROR("Failed to initialize VectorBlox Hardware.");
        return false;
    }

    // 2. Load Model Binary (.vnnx / .blob)
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        XINFER_LOG_ERROR("Failed to open model file: " + model_path);
        return false;
    }

    size_t size = file.tellg();
    file.seekg(0);
    m_impl->model_blob.resize(size);
    file.read(reinterpret_cast<char*>(m_impl->model_blob.data()), size);

    // 3. Parse Model Info
    // The blob contains header info describing inputs/outputs
    m_impl->model_info = vbx_cnn_model_get_info(m_impl->model_blob.data());
    
    if (!m_impl->model_info) {
        XINFER_LOG_ERROR("Invalid VectorBlox model blob.");
        return false;
    }

    // 4. Resolve Buffers
    // VBX inputs/outputs are usually at fixed offsets within a scratchpad or the blob itself
    int input_idx = 0; // Assuming single input model
    int output_idx = 0;

    m_impl->input_buffer_ptr = vbx_cnn_get_io_ptr(m_impl->model_blob.data(), input_idx, VBX_CNN_INPUT);
    m_impl->output_buffer_ptr = vbx_cnn_get_io_ptr(m_impl->model_blob.data(), output_idx, VBX_CNN_OUTPUT);

    XINFER_LOG_INFO("Loaded VectorBlox Model: " + model_path);
    return true;
}

void VectorBloxBackend::predict(const std::vector<core::Tensor>& inputs, 
                                std::vector<core::Tensor>& outputs) {
    
    // 1. Quantize & Copy Inputs
    // VectorBlox requires INT8/UINT8. We convert from xInfer Float32.
    const auto& input = inputs[0];
    
    vbx_cnn_io_info_t* input_info = vbx_cnn_get_io_info(m_impl->model_info, 0, VBX_CNN_INPUT);
    float scale = input_info->scale;
    int zero_point = input_info->zero_point;
    
    uint8_t* hw_in_ptr = static_cast<uint8_t*>(m_impl->input_buffer_ptr);
    const float* src_ptr = static_cast<const float*>(input.data());
    size_t count = input.size();

    for(size_t i=0; i<count; ++i) {
        int32_t val = static_cast<int32_t>(round(src_ptr[i] / scale) + zero_point);
        hw_in_ptr[i] = static_cast<uint8_t>(std::max(0, std::min(255, val)));
    }

    // 2. Flush Cache
    // Ensure the data we just wrote to DDR is visible to the FPGA
    m_impl->flush_cache(hw_in_ptr, count);

    // 3. Run Inference
    // This function writes to the FPGA control registers to start the accelerator
    int status = vbx_cnn_model_run(m_impl->model_blob.data());
    
    if (status != 0) {
        XINFER_LOG_ERROR("VectorBlox execution failed.");
        return;
    }

    // 4. Invalidate Cache
    // Ensure we read fresh results from the FPGA, not stale CPU cache lines
    vbx_cnn_io_info_t* output_info = vbx_cnn_get_io_info(m_impl->model_info, 0, VBX_CNN_OUTPUT);
    size_t out_bytes = output_info->elements * sizeof(uint8_t);
    m_impl->invalidate_cache(m_impl->output_buffer_ptr, out_bytes);

    // 5. Dequantize Outputs
    if (outputs[0].empty()) {
        outputs[0].resize({1, (int64_t)output_info->elements}, core::DataType::kFLOAT);
    }
    
    uint8_t* hw_out_ptr = static_cast<uint8_t*>(m_impl->output_buffer_ptr);
    float* dst_ptr = static_cast<float*>(outputs[0].data());
    
    float out_scale = output_info->scale;
    int out_zp = output_info->zero_point;

    for(size_t i=0; i<output_info->elements; ++i) {
        dst_ptr[i] = (static_cast<float>(hw_out_ptr[i]) - out_zp) * out_scale;
    }
}

std::string VectorBloxBackend::device_name() const {
    return "Microchip VectorBlox";
}

float VectorBloxBackend::get_fpga_temp() const {
    return 0.0f; // Requires system controller access
}

// =================================================================================
// 3. Auto-Registration
// =================================================================================

namespace {
    volatile bool registered = xinfer::backends::BackendFactory::register_backend(
        xinfer::Target::MICROCHIP_VECTORBLOX,
        [](const xinfer::Config& config) -> std::unique_ptr<xinfer::IBackend> {
            VectorBloxConfig vb_cfg;
            vb_cfg.model_path = config.model_path;
            
            // Parse vendor params
            for(const auto& param : config.vendor_params) {
                if(param == "CORE=V1000") vb_cfg.core_type = VbxCoreType::V1000;
            }
            
            auto backend = std::make_unique<VectorBloxBackend>(vb_cfg);
            if(backend->load_model(vb_cfg.model_path)) {
                return backend;
            }
            return nullptr;
        }
    );
}

} // namespace xinfer::backends::microchip