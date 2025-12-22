#include <xinfer/backends/intel_ov/backend.h>
#include <xinfer/core/logging.h>
#include <xinfer/backends/backend_factory.h>

// --- OpenVINO Headers ---
#include <openvino/openvino.hpp>

#include <iostream>
#include <vector>
#include <memory>
#include <map>

namespace xinfer::backends::openvino {

// =================================================================================
// 1. Helper: Type Mapping
// =================================================================================

static ov::element::Type map_dtype(core::DataType type) {
    switch (type) {
        case core::DataType::kFLOAT:   return ov::element::f32;
        case core::DataType::kFLOAT16: return ov::element::f16;
        case core::DataType::kINT8:    return ov::element::i8;
        case core::DataType::kUINT8:   return ov::element::u8;
        case core::DataType::kINT32:   return ov::element::i32;
        case core::DataType::kINT64:   return ov::element::i64;
        default: return ov::element::undefined;
    }
}

static core::DataType map_ov_type(ov::element::Type type) {
    if (type == ov::element::f32) return core::DataType::kFLOAT;
    if (type == ov::element::f16) return core::DataType::kFLOAT16;
    if (type == ov::element::i8)  return core::DataType::kINT8;
    if (type == ov::element::u8)  return core::DataType::kUINT8;
    if (type == ov::element::i32) return core::DataType::kINT32;
    if (type == ov::element::i64) return core::DataType::kINT64;
    return core::DataType::kUNKNOWN;
}

// =================================================================================
// 2. PImpl Implementation
// =================================================================================

struct OpenVINOBackend::Impl {
    OpenVINOConfig config;

    // OpenVINO Core Runtime Objects
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;

    // Caches for fast tensor access
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    explicit Impl(const OpenVINOConfig& cfg) : config(cfg) {}

    // --- Helper: Device Name Resolution ---
    std::string get_device_string() {
        switch (config.device_type) {
            case DeviceType::GPU:    return "GPU";
            case DeviceType::NPU:    return "NPU"; // Core Ultra / Meteor Lake
            case DeviceType::AUTO:   return "AUTO";
            case DeviceType::HETERO: return "HETERO:GPU,CPU";
            case DeviceType::CPU:    
            default:                 return "CPU";
        }
    }
};

// =================================================================================
// 3. Public API Implementation
// =================================================================================

OpenVINOBackend::OpenVINOBackend(const OpenVINOConfig& config)
    : m_config(config), m_impl(std::make_unique<Impl>(config)) {
}

OpenVINOBackend::~OpenVINOBackend() = default;

bool OpenVINOBackend::load_model(const std::string& model_path) {
    try {
        // 1. Configure Core Properties
        // Set Cache Directory to speed up load times (creates .blob files)
        if (!m_config.cache_dir.empty()) {
            m_impl->core.set_property(ov::cache_dir(m_config.cache_dir));
        }

        // 2. Read Model IR (.xml)
        XINFER_LOG_INFO("Reading OpenVINO IR: " + model_path);
        m_impl->model = m_impl->core.read_model(model_path);

        // 3. Set Device Properties
        ov::AnyMap properties;
        
        // Performance Hint (Latency vs Throughput)
        if (m_config.perf_hint == PerformanceHint::LATENCY)
            properties[ov::hint::performance_mode] = ov::hint::PerformanceMode::LATENCY;
        else if (m_config.perf_hint == PerformanceHint::THROUGHPUT)
            properties[ov::hint::performance_mode] = ov::hint::PerformanceMode::THROUGHPUT;

        // Inference Precision Hint
        if (m_config.inference_precision == OvPrecision::FP16)
            properties[ov::hint::inference_precision] = ov::element::f16;
        
        // Num Streams (Parallelism)
        if (m_config.num_streams > 0)
            properties[ov::num_streams] = m_config.num_streams;

        // 4. Compile Model (AOT compilation to hardware binary)
        std::string device = m_impl->get_device_string();
        XINFER_LOG_INFO("Compiling model for device: " + device);
        
        m_impl->compiled_model = m_impl->core.compile_model(m_impl->model, device, properties);

        // 5. Create Inference Request
        m_impl->infer_request = m_impl->compiled_model.create_infer_request();

        // 6. Cache I/O Names
        // Note: Modern OpenVINO usually accesses by index or port, but names are safe.
        for (const auto& input : m_impl->compiled_model.inputs()) {
            m_impl->input_names.push_back(input.get_any_name());
        }
        for (const auto& output : m_impl->compiled_model.outputs()) {
            m_impl->output_names.push_back(output.get_any_name());
        }

        return true;

    } catch (const std::exception& e) {
        XINFER_LOG_ERROR("OpenVINO Load Failed: " + std::string(e.what()));
        return false;
    }
}

void OpenVINOBackend::predict(const std::vector<core::Tensor>& inputs, 
                              std::vector<core::Tensor>& outputs) {
    try {
        if (inputs.size() != m_impl->input_names.size()) {
            XINFER_LOG_ERROR("Input count mismatch.");
            return;
        }

        // 1. Prepare Inputs (Zero-Copy Wrapping)
        for (size_t i = 0; i < inputs.size(); ++i) {
            const auto& tensor = inputs[i];
            
            // Map xInfer shape to OpenVINO shape
            ov::Shape ov_shape;
            for (auto dim : tensor.shape()) ov_shape.push_back(dim);

            // Create ov::Tensor using external pointer (Zero-Copy)
            // This tells OpenVINO: "Use the memory at tensor.data(), don't allocate new."
            ov::Tensor wrapper(
                map_dtype(tensor.dtype()), 
                ov_shape, 
                const_cast<void*>(tensor.data())
            );

            m_impl->infer_request.set_tensor(m_impl->input_names[i], wrapper);
        }

        // 2. Run Inference
        m_impl->infer_request.infer();

        // 3. Process Outputs
        if (outputs.size() != m_impl->output_names.size()) {
            outputs.resize(m_impl->output_names.size());
        }

        for (size_t i = 0; i < m_impl->output_names.size(); ++i) {
            ov::Tensor out_tensor = m_impl->infer_request.get_tensor(m_impl->output_names[i]);
            
            // Get shape and resize output tensor if needed
            std::vector<int64_t> out_shape;
            for (auto dim : out_tensor.get_shape()) out_shape.push_back(dim);

            // If output tensor is empty or shape changed (dynamic batching), resize it
            if (outputs[i].empty() || outputs[i].shape() != out_shape) {
                outputs[i].resize(out_shape, map_ov_type(out_tensor.get_element_type()));
            }

            // Copy Data
            // Note: If 'outputs[i]' was also wrapped, we could do zero-copy here too,
            // but for safety (lifetime management), we usually memcpy output.
            size_t byte_size = out_tensor.get_byte_size();
            std::memcpy(outputs[i].data(), out_tensor.data(), byte_size);
        }

    } catch (const std::exception& e) {
        XINFER_LOG_ERROR("OpenVINO Inference Failed: " + std::string(e.what()));
    }
}

std::string OpenVINOBackend::device_name() const {
    try {
        // Query runtime parameter for full device name
        return m_impl->compiled_model.get_property(ov::device::full_name).as<std::string>();
    } catch (...) {
        return "Intel OpenVINO Device";
    }
}

std::string OpenVINOBackend::get_input_layout(size_t index) const {
    // OpenVINO 2.0 has mostly moved away from strict Layout classes in runtime,
    // handling it via PrePostProcessor, but we can inspect model info.
    if (index < m_impl->model->inputs().size()) {
        auto layout = m_impl->model->input(index).get_layout();
        return layout.to_string();
    }
    return "";
}

// =================================================================================
// 4. Auto-Registration
// =================================================================================

namespace {
    volatile bool registered = xinfer::backends::BackendFactory::register_backend(
        xinfer::Target::INTEL_OV,
        [](const xinfer::Config& config) -> std::unique_ptr<xinfer::IBackend> {
            OpenVINOConfig ov_cfg;
            ov_cfg.model_path = config.model_path;
            
            // Parse vendor flags
            for(const auto& param : config.vendor_params) {
                if(param == "DEVICE=GPU") ov_cfg.device_type = DeviceType::GPU;
                if(param == "DEVICE=NPU") ov_cfg.device_type = DeviceType::NPU;
                if(param == "HINT=THROUGHPUT") ov_cfg.perf_hint = PerformanceHint::THROUGHPUT;
                if(param.find("THREADS=") != std::string::npos) {
                    ov_cfg.num_threads = std::stoi(param.substr(8));
                }
            }
            
            auto backend = std::make_unique<OpenVINOBackend>(ov_cfg);
            if(backend->load_model(ov_cfg.model_path)) {
                return backend;
            }
            return nullptr;
        }
    );
}

} // namespace xinfer::backends::openvino