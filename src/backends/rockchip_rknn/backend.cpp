#include <xinfer/backends/rockchip_rknn/backend.h>
#include <xinfer/core/logging.h>
#include <xinfer/backends/backend_factory.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <algorithm>

// --- Rockchip Headers ---
#include "rknn_api.h"

namespace xinfer::backends::rknn {

// =================================================================================
// 1. PImpl Implementation
// =================================================================================

struct RknnBackend::Impl {
    RknnConfig config;
    
    // RKNN Context Handle
    rknn_context ctx = 0;

    // IO Metadata
    rknn_input_output_num io_num;
    std::vector<rknn_tensor_attr> input_attrs;
    std::vector<rknn_tensor_attr> output_attrs;

    // Buffer for model file content
    std::vector<unsigned char> model_data;

    explicit Impl(const RknnConfig& cfg) : config(cfg) {
        memset(&io_num, 0, sizeof(io_num));
    }

    ~Impl() {
        if (ctx > 0) {
            rknn_destroy(ctx);
        }
    }

    // --- Helper: Map xInfer CoreMask to RKNN Enum ---
    rknn_core_mask map_core_mask(RknnCoreMask mask) {
        switch(mask) {
            case RknnCoreMask::CORE_0:     return RKNN_NPU_CORE_0;
            case RknnCoreMask::CORE_1:     return RKNN_NPU_CORE_1;
            case RknnCoreMask::CORE_2:     return RKNN_NPU_CORE_2;
            case RknnCoreMask::CORE_0_1:   return RKNN_NPU_CORE_0_1;
            case RknnCoreMask::CORE_0_1_2: return RKNN_NPU_CORE_0_1_2;
            case RknnCoreMask::AUTO: 
            default:                       return RKNN_NPU_CORE_AUTO;
        }
    }
};

// =================================================================================
// 2. Public API Implementation
// =================================================================================

RknnBackend::RknnBackend(const RknnConfig& config)
    : m_config(config), m_impl(std::make_unique<Impl>(config)) {
}

RknnBackend::~RknnBackend() = default;

bool RknnBackend::load_model(const std::string& model_path) {
    // 1. Load Model Binary
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        XINFER_LOG_ERROR("Failed to open RKNN model: " + model_path);
        return false;
    }
    size_t size = file.tellg();
    file.seekg(0);
    m_impl->model_data.resize(size);
    file.read(reinterpret_cast<char*>(m_impl->model_data.data()), size);

    // 2. Initialize RKNN Context
    int ret = rknn_init(&m_impl->ctx, m_impl->model_data.data(), size, 0, NULL);
    if (ret < 0) {
        XINFER_LOG_ERROR("rknn_init failed with code: " + std::to_string(ret));
        return false;
    }

    // 3. Set Core Mask (RK3588 Multi-Core NPU)
    if (m_config.core_mask != RknnCoreMask::AUTO) {
        ret = rknn_set_core_mask(m_impl->ctx, m_impl->map_core_mask(m_config.core_mask));
        if (ret < 0) {
            XINFER_LOG_WARN("Failed to set NPU core mask. Falling back to AUTO.");
        }
    }

    // 4. Query Input/Output Numbers
    ret = rknn_query(m_impl->ctx, RKNN_QUERY_IN_OUT_NUM, &m_impl->io_num, sizeof(m_impl->io_num));
    if (ret < 0) {
        XINFER_LOG_ERROR("Failed to query IO numbers.");
        return false;
    }

    // 5. Query Input Attributes
    m_impl->input_attrs.resize(m_impl->io_num.n_input);
    for (uint32_t i = 0; i < m_impl->io_num.n_input; ++i) {
        m_impl->input_attrs[i].index = i;
        ret = rknn_query(m_impl->ctx, RKNN_QUERY_INPUT_ATTR, &m_impl->input_attrs[i], sizeof(rknn_tensor_attr));
        if (ret < 0) {
            XINFER_LOG_ERROR("Failed to query Input Attr " + std::to_string(i));
            return false;
        }
    }

    // 6. Query Output Attributes
    m_impl->output_attrs.resize(m_impl->io_num.n_output);
    for (uint32_t i = 0; i < m_impl->io_num.n_output; ++i) {
        m_impl->output_attrs[i].index = i;
        ret = rknn_query(m_impl->ctx, RKNN_QUERY_OUTPUT_ATTR, &m_impl->output_attrs[i], sizeof(rknn_tensor_attr));
        if (ret < 0) {
            XINFER_LOG_ERROR("Failed to query Output Attr " + std::to_string(i));
            return false;
        }
    }

    XINFER_LOG_INFO("Loaded RKNN Model: " + model_path);
    return true;
}

void RknnBackend::predict(const std::vector<core::Tensor>& inputs, 
                          std::vector<core::Tensor>& outputs) {
    
    // Validation
    if (inputs.size() != m_impl->io_num.n_input) {
        XINFER_LOG_ERROR("Input count mismatch. Expected " + std::to_string(m_impl->io_num.n_input));
        return;
    }

    // 1. Set Inputs
    std::vector<rknn_input> rknn_inputs(m_impl->io_num.n_input);
    for (uint32_t i = 0; i < m_impl->io_num.n_input; ++i) {
        memset(&rknn_inputs[i], 0, sizeof(rknn_input));
        
        rknn_inputs[i].index = i;
        rknn_inputs[i].type = RKNN_TENSOR_FLOAT32; // Assuming xInfer passes float
        rknn_inputs[i].size = inputs[i].size() * sizeof(float);
        rknn_inputs[i].fmt = RKNN_TENSOR_NCHW; // Matches xInfer standard
        rknn_inputs[i].buf = const_cast<void*>(inputs[i].data());
        
        // Pass-through indicates we want RKNN to handle internal conversion 
        // if model expects INT8 but we pass Float32.
        rknn_inputs[i].pass_through = 0; 
    }

    int ret = rknn_inputs_set(m_impl->ctx, m_impl->io_num.n_input, rknn_inputs.data());
    if (ret < 0) {
        XINFER_LOG_ERROR("rknn_inputs_set failed.");
        return;
    }

    // 2. Run Inference
    ret = rknn_run(m_impl->ctx, NULL);
    if (ret < 0) {
        XINFER_LOG_ERROR("rknn_run failed.");
        return;
    }

    // 3. Get Outputs
    std::vector<rknn_output> rknn_outputs(m_impl->io_num.n_output);
    for (uint32_t i = 0; i < m_impl->io_num.n_output; ++i) {
        memset(&rknn_outputs[i], 0, sizeof(rknn_output));
        rknn_outputs[i].index = i;
        rknn_outputs[i].want_float = 1; // Force driver to dequantize to float
    }

    ret = rknn_outputs_get(m_impl->ctx, m_impl->io_num.n_output, rknn_outputs.data(), NULL);
    if (ret < 0) {
        XINFER_LOG_ERROR("rknn_outputs_get failed.");
        return;
    }

    // 4. Copy to xInfer Tensors
    if (outputs.size() != m_impl->io_num.n_output) {
        outputs.resize(m_impl->io_num.n_output);
    }

    for (uint32_t i = 0; i < m_impl->io_num.n_output; ++i) {
        // Determine shape from attributes
        std::vector<int64_t> shape;
        for(int d=0; d<m_impl->output_attrs[i].n_dims; ++d) {
            shape.push_back(m_impl->output_attrs[i].dims[d]);
        }

        // Resize if needed
        if (outputs[i].empty()) {
            outputs[i].resize(shape, core::DataType::kFLOAT);
        }

        // Copy
        size_t size_bytes = rknn_outputs[i].size;
        std::memcpy(outputs[i].data(), rknn_outputs[i].buf, size_bytes);
    }

    // 5. Release RKNN Output Memory
    rknn_outputs_release(m_impl->ctx, m_impl->io_num.n_output, rknn_outputs.data());
}

std::string RknnBackend::device_name() const {
    return "Rockchip NPU (RKNPU2)";
}

std::string RknnBackend::get_sdk_version() const {
    rknn_sdk_version version;
    if (rknn_query(m_impl->ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version)) >= 0) {
        return std::string(version.api_version) + " / " + std::string(version.drv_version);
    }
    return "Unknown";
}

// =================================================================================
// 3. Auto-Registration
// =================================================================================

namespace {
    volatile bool registered = xinfer::backends::BackendFactory::register_backend(
        xinfer::Target::ROCKCHIP_RKNN,
        [](const xinfer::Config& config) -> std::unique_ptr<xinfer::IBackend> {
            RknnConfig rknn_cfg;
            rknn_cfg.model_path = config.model_path;
            
            // Parse vendor flags
            for(const auto& param : config.vendor_params) {
                if(param == "CORE=0") rknn_cfg.core_mask = RknnCoreMask::CORE_0;
                if(param == "CORE=1") rknn_cfg.core_mask = RknnCoreMask::CORE_1;
                if(param == "CORE=ALL") rknn_cfg.core_mask = RknnCoreMask::CORE_0_1_2;
            }
            
            auto backend = std::make_unique<RknnBackend>(rknn_cfg);
            if(backend->load_model(rknn_cfg.model_path)) {
                return backend;
            }
            return nullptr;
        }
    );
}

} // namespace xinfer::backends::rknn