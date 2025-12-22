#include <xinfer/backends/hailo_rt/backend.h>
#include <xinfer/core/logging.h>
#include <xinfer/backends/backend_factory.h>

// --- HailoRT Headers ---
#include <hailo/hailort.hpp>

#include <iostream>
#include <vector>
#include <memory>
#include <map>

namespace xinfer::backends::hailo {

using namespace hailort;

// =================================================================================
// 1. PImpl Implementation
// =================================================================================

struct HailoBackend::Impl {
    HailoConfig config;

    // HailoRT Core Objects
    std::unique_ptr<VDevice> vdevice;
    std::shared_ptr<ConfiguredNetworkGroup> network_group;
    
    // Virtual Streams (The pipelines for data)
    std::vector<InputVStream> input_vstreams;
    std::vector<OutputVStream> output_vstreams;
    
    // Metadata mapping
    std::map<std::string, size_t> input_name_to_index;
    std::map<std::string, size_t> output_name_to_index;

    explicit Impl(const HailoConfig& cfg) : config(cfg) {}

    ~Impl() {
        // Clean up streams before network group and device
        input_vstreams.clear();
        output_vstreams.clear();
        network_group.reset(); // Release shared ptr
        vdevice.reset();
    }

    // --------------------------------------------------------------------------
    // Helper: Map xInfer format to Hailo format
    // --------------------------------------------------------------------------
    hailo_format_type_t get_hailo_format(StreamFormat fmt) {
        switch(fmt) {
            case StreamFormat::USER_UINT8:   return HAILO_FORMAT_TYPE_UINT8;
            case StreamFormat::USER_FLOAT32: return HAILO_FORMAT_TYPE_FLOAT32;
            case StreamFormat::AUTO: default: return HAILO_FORMAT_TYPE_AUTO;
        }
    }
};

// =================================================================================
// 2. Public API Implementation
// =================================================================================

HailoBackend::HailoBackend(const HailoConfig& config)
    : m_config(config), m_impl(std::make_unique<Impl>(config)) {
}

HailoBackend::~HailoBackend() = default;

bool HailoBackend::load_model(const std::string& model_path) {
    try {
        // 1. Create VDevice (Scan for PCIe/USB devices)
        hailo_vdevice_params_t params = HailoRTDefaults::get_vdevice_params();
        
        // Use multiplexer if requested (allows sharing chip with other apps)
        if (m_config.use_multiplexer) {
            params.device_count = 1;
            params.group_id = "xinfer_group"; 
            params.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;
        }

        auto vdevice_exp = VDevice::create(params);
        if (!vdevice_exp) {
            XINFER_LOG_ERROR("Failed to create Hailo VDevice. Error: " + std::to_string(vdevice_exp.status()));
            return false;
        }
        m_impl->vdevice = vdevice_exp.release();

        // 2. Load HEF File
        auto hef_exp = Hef::create(model_path);
        if (!hef_exp) {
            XINFER_LOG_ERROR("Failed to load HEF file: " + model_path);
            return false;
        }
        Hef hef = hef_exp.release();

        // 3. Configure Network Group
        // A HEF can contain multiple network groups, usually we take the first one (Default)
        auto configure_params = hef.create_configure_params(HAILO_STREAM_INTERFACE_PCIE);
        if (!configure_params) {
            XINFER_LOG_ERROR("Failed to create configure params.");
            return false;
        }

        auto network_groups_exp = m_impl->vdevice->configure(hef, configure_params.value());
        if (!network_groups_exp) {
            XINFER_LOG_ERROR("Failed to configure network group on device.");
            return false;
        }
        
        // Take the first network group
        m_impl->network_group = network_groups_exp.value()[0];

        // 4. Create VStreams (Input/Output pipelines)
        // ------------------------------------------------------
        
        // Define Input Params (Format override)
        auto input_vstream_params = m_impl->network_group->make_input_vstream_params(
            true, // Quantized?
            m_impl->get_hailo_format(m_config.input_format), // User type (e.g. Float32)
            HAILO_FORMAT_TYPE_UINT8 // Device type (usually INT8/UINT8)
        );
        
        if (!input_vstream_params) {
             XINFER_LOG_ERROR("Failed to create input vstream params.");
             return false;
        }

        // Create Input Streams
        auto input_streams_exp = VStreamsBuilder::create_input_vstreams(
            *m_impl->network_group, input_vstream_params.value());
        
        if (!input_streams_exp) {
            XINFER_LOG_ERROR("Failed to create input vstreams.");
            return false;
        }
        m_impl->input_vstreams = input_streams_exp.release();

        // Define Output Params
        auto output_vstream_params = m_impl->network_group->make_output_vstream_params(
            true, 
            m_impl->get_hailo_format(m_config.output_format),
            HAILO_FORMAT_TYPE_UINT8
        );

        // Create Output Streams
        auto output_streams_exp = VStreamsBuilder::create_output_vstreams(
            *m_impl->network_group, output_vstream_params.value());
            
        if (!output_streams_exp) {
            XINFER_LOG_ERROR("Failed to create output vstreams.");
            return false;
        }
        m_impl->output_vstreams = output_streams_exp.release();

        // 5. Cache Name Mapping for Indexing
        for (size_t i = 0; i < m_impl->input_vstreams.size(); ++i) {
            m_impl->input_name_to_index[m_impl->input_vstreams[i].name()] = i;
        }

        XINFER_LOG_INFO("Loaded Hailo HEF: " + model_path);
        return true;

    } catch (const std::exception& e) {
        XINFER_LOG_ERROR("Hailo Backend Load Exception: " + std::string(e.what()));
        return false;
    }
}

void HailoBackend::predict(const std::vector<core::Tensor>& inputs, 
                           std::vector<core::Tensor>& outputs) {
    
    // Validation
    if (inputs.size() != m_impl->input_vstreams.size()) {
        XINFER_LOG_ERROR("Input count mismatch.");
        return;
    }

    // 1. Write to Input VStreams
    // ----------------------------------------------------------------
    for (size_t i = 0; i < inputs.size(); ++i) {
        // We assume input ordering matches stream creation order. 
        // In prod, check names via input_name_to_index.
        
        size_t size_bytes = inputs[i].size() * core::element_size(inputs[i].dtype());
        
        auto status = m_impl->input_vstreams[i].write(
            const_cast<void*>(inputs[i].data()), 
            size_bytes
        );

        if (status != HAILO_SUCCESS) {
            XINFER_LOG_ERROR("Failed to write to Hailo Input Stream " + std::to_string(i));
            return;
        }
    }

    // 2. Read from Output VStreams (Blocking)
    // ----------------------------------------------------------------
    if (outputs.size() != m_impl->output_vstreams.size()) {
        outputs.resize(m_impl->output_vstreams.size());
    }

    for (size_t i = 0; i < m_impl->output_vstreams.size(); ++i) {
        // Get frame size info from the stream
        size_t frame_size = m_impl->output_vstreams[i].get_frame_size();
        
        // Prepare output tensor
        if (outputs[i].empty()) {
             // HailoRT shapes aren't always explicitly queryable as a vector in easy form, 
             // often just flat byte size. For now, we alloc flat buffer.
             // In real app, we parse HEF info to reshape.
             outputs[i].resize({(int64_t)frame_size / 4}, core::DataType::kFLOAT); // Assuming float output
        }

        // Read
        auto status = m_impl->output_vstreams[i].read(
            outputs[i].data(), 
            frame_size
        );

        if (status != HAILO_SUCCESS) {
            XINFER_LOG_ERROR("Failed to read from Hailo Output Stream " + std::to_string(i));
        }
    }
}

std::string HailoBackend::device_name() const {
    return "Hailo-8 AI Processor";
}

float HailoBackend::get_chip_temperature() const {
    // Requires hailort 4.10+ APIs
    // auto info = m_impl->vdevice->get_chip_temperature();
    return 0.0f; // Placeholder
}

// =================================================================================
// 3. Auto-Registration
// =================================================================================

namespace {
    volatile bool registered = xinfer::backends::BackendFactory::register_backend(
        xinfer::Target::HAILO_RT,
        [](const xinfer::Config& config) -> std::unique_ptr<xinfer::IBackend> {
            HailoConfig h_cfg;
            h_cfg.model_path = config.model_path;
            
            // Parse vendor flags
            for(const auto& param : config.vendor_params) {
                if(param == "DEVICE=USB") h_cfg.device_type = DeviceInterface::USB;
                if(param == "FORMAT=UINT8") h_cfg.input_format = StreamFormat::USER_UINT8;
            }
            
            auto backend = std::make_unique<HailoBackend>(h_cfg);
            if(backend->load_model(h_cfg.model_path)) {
                return backend;
            }
            return nullptr;
        }
    );
}

} // namespace xinfer::backends::hailo