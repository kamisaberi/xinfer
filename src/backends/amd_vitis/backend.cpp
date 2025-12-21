#include <xinfer/backends/amd_vitis/backend.h>
#include <xinfer/core/logging.h>
#include <xinfer/backends/backend_factory.h>

// --- Vitis AI / VART Headers ---
#include <xir/graph/graph.hpp>
#include <vart/runner.hpp>
#include <vart/runner_ext.hpp>
#include <vart/tensor_buffer.hpp>

#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace xinfer::backends::vitis {

// =================================================================================
// 1. PImpl Implementation
// =================================================================================

struct VitisBackend::Impl {
    VitisConfig config;

    // VART Graph and Runner
    std::unique_ptr<xir::Graph> graph;
    std::unique_ptr<vart::Runner> runner;

    // Tensor Metadata (Shapes, Scales)
    std::vector<const xir::Tensor*> input_tensor_meta;
    std::vector<const xir::Tensor*> output_tensor_meta;

    explicit Impl(const VitisConfig& cfg) : config(cfg) {}

    // --------------------------------------------------------------------------
    // Helper: Find the specific DPU subgraph
    // --------------------------------------------------------------------------
    const xir::Subgraph* get_dpu_subgraph(const xir::Graph* graph) {
        auto root = graph->get_root_subgraph();
        auto children = root->children_topological_sort();

        for (auto* c : children) {
            if (c->get_attr<std::string>("device") == "DPU") {
                return c;
            }
        }
        return nullptr;
    }

    // --------------------------------------------------------------------------
    // Helper: Data Scaling (FP32 <-> INT8)
    // --------------------------------------------------------------------------
    void copy_and_scale_input(const core::Tensor& src, vart::TensorBuffer* dst_buf, float scale, int zero_point) {
        uint64_t data_size = 0;
        int8_t* dst_data = reinterpret_cast<int8_t*>(dst_buf->data(std::vector<int>{0}, &data_size).first);
        const float* src_data = static_cast<const float*>(src.data());
        size_t count = src.size(); // Number of elements

        // Vectorized loop (OpenMP or NEON/SIMD suggested here for max speed)
        for (size_t i = 0; i < count; ++i) {
            float val = src_data[i] * scale;
            // Clamp to INT8 range
            int32_t val_i32 = static_cast<int32_t>(round(val)) + zero_point;
            dst_data[i] = static_cast<int8_t>(std::max(-128, std::min(127, val_i32)));
        }
    }

    void copy_and_unscale_output(vart::TensorBuffer* src_buf, core::Tensor& dst, float scale, int zero_point) {
        uint64_t data_size = 0;
        int8_t* src_data = reinterpret_cast<int8_t*>(src_buf->data(std::vector<int>{0}, &data_size).first);
        float* dst_data = static_cast<float*>(dst.data());
        size_t count = dst.size();

        float inv_scale = 1.0f / scale;

        for (size_t i = 0; i < count; ++i) {
            dst_data[i] = (static_cast<float>(src_data[i]) - zero_point) * inv_scale;
        }
    }
};

// =================================================================================
// 2. Public API Implementation
// =================================================================================

VitisBackend::VitisBackend(const VitisConfig& config)
    : m_config(config), m_impl(std::make_unique<Impl>(config)) {
}

VitisBackend::~VitisBackend() = default;

bool VitisBackend::load_model(const std::string& model_path) {
    try {
        // 1. Deserialize the .xmodel
        m_impl->graph = xir::Graph::deserialize(model_path);

        // 2. Find the DPU subgraph
        auto subgraph = m_impl->get_dpu_subgraph(m_impl->graph.get());
        if (!subgraph) {
            XINFER_LOG_ERROR("No DPU subgraph found in xmodel: " + model_path);
            return false;
        }

        // 3. Create the Runner
        m_impl->runner = vart::Runner::create_runner(subgraph, "run");

        // 4. Cache Input/Output Metadata for Scaling
        m_impl->input_tensor_meta = m_impl->runner->get_input_tensors();
        m_impl->output_tensor_meta = m_impl->runner->get_output_tensors();

        XINFER_LOG_INFO("Loaded Vitis DPU Model: " + subgraph->get_name());
        return true;

    } catch (const std::exception& e) {
        XINFER_LOG_ERROR("Vitis Backend Load Failed: " + std::string(e.what()));
        return false;
    }
}

void VitisBackend::predict(const std::vector<core::Tensor>& inputs,
                           std::vector<core::Tensor>& outputs) {

    // Get VART Input Buffers
    auto input_buffers = m_impl->runner->get_inputs();
    auto output_buffers = m_impl->runner->get_outputs();

    // 1. Prepare Inputs (Scale FP32 -> INT8)
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto* tensor_meta = m_impl->input_tensor_meta[i];

        // Get DPU quantization info
        int32_t fixpos = tensor_meta->get_attr<int32_t>("fix_point");
        float scale = std::pow(2.0f, fixpos);
        // Note: DPU typically assumes symmetric INT8 (zero_point = 0)

        // Copy and Scale
        m_impl->copy_and_scale_input(inputs[i], input_buffers[i], scale, 0);
    }

    // 2. Run Inference (Async + Wait)
    // We use execute_async because it's non-blocking on the CPU, allowing us to
    // potentially prepare the next frame while DPU works (pipeline).
    auto job_id = m_impl->runner->execute_async(input_buffers, output_buffers);
    auto status = m_impl->runner->wait(job_id.first, -1); // Wait indefinitely

    if (status != 0) {
        XINFER_LOG_ERROR("DPU execution failed.");
        return;
    }

    // 3. Process Outputs (Unscale INT8 -> FP32)
    if (outputs.size() != output_buffers.size()) {
        outputs.resize(output_buffers.size());
    }

    for (size_t i = 0; i < output_buffers.size(); ++i) {
        auto* tensor_meta = m_impl->output_tensor_meta[i];
        int32_t fixpos = tensor_meta->get_attr<int32_t>("fix_point");
        float scale = std::pow(2.0f, fixpos);

        // Populate output shape if empty
        if (outputs[i].empty()) {
            std::vector<int64_t> shape;
            for(auto d : tensor_meta->get_shape()) shape.push_back(d);
            outputs[i].resize(shape, core::DataType::kFLOAT);
        }

        m_impl->copy_and_unscale_output(output_buffers[i], outputs[i], scale, 0);
    }
}

std::string VitisBackend::device_name() const {
    return "Xilinx DPU (Vitis AI)";
}

float VitisBackend::get_input_scale(size_t index) const {
    if (index < m_impl->input_tensor_meta.size()) {
        int32_t fixpos = m_impl->input_tensor_meta[index]->get_attr<int32_t>("fix_point");
        return std::pow(2.0f, fixpos);
    }
    return 1.0f;
}

float VitisBackend::get_output_scale(size_t index) const {
    if (index < m_impl->output_tensor_meta.size()) {
        int32_t fixpos = m_impl->output_tensor_meta[index]->get_attr<int32_t>("fix_point");
        return std::pow(2.0f, fixpos);
    }
    return 1.0f;
}

// =================================================================================
// 3. Auto-Registration
// =================================================================================

namespace {
    volatile bool registered = xinfer::backends::BackendFactory::register_backend(
        xinfer::Target::AMD_VITIS,
        [](const xinfer::Config& config) -> std::unique_ptr<xinfer::IBackend> {
            VitisConfig vitis_cfg;
            vitis_cfg.model_path = config.model_path;

            // Parse vendor flags
            for(const auto& param : config.vendor_params) {
                if(param.find("NUM_RUNNERS=") != std::string::npos) {
                     vitis_cfg.num_runners = std::stoi(param.substr(12));
                }
            }

            auto backend = std::make_unique<VitisBackend>(vitis_cfg);
            if(backend->load_model(vitis_cfg.model_path)) {
                return backend;
            }
            return nullptr;
        }
    );
}

} // namespace xinfer::backends::vitis