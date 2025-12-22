#include <xinfer/backends/lattice_sensai/backend.h>
#include <xinfer/core/logging.h>
#include <xinfer/backends/backend_factory.h>

#include <iostream>
#include <vector>
#include <fstream>
#include <thread>
#include <chrono>
#include <cstring>
#include <cmath>

// --- Platform Headers ---
#ifdef XINFER_ENABLE_FTDI
#include <ftdi.h>
#endif

// Linux Headers for SPI
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/spi/spidev.h>

namespace xinfer::backends::lattice {

// =================================================================================
// Constants: Generic Lattice CNN IP Register Map
// (These depend on your specific bitstream configuration in Lattice Propel/Radiant)
// =================================================================================
constexpr uint32_t REG_CONTROL      = 0x0000;
constexpr uint32_t REG_STATUS       = 0x0004;
constexpr uint32_t REG_CYCLES       = 0x0008;

constexpr uint32_t STATUS_BUSY      = (1 << 0);
constexpr uint32_t STATUS_DONE      = (1 << 1);
constexpr uint32_t CMD_START        = (1 << 0);

// =================================================================================
// 1. PImpl Implementation
// =================================================================================

struct LatticeBackend::Impl {
    LatticeConfig config;

#ifdef XINFER_ENABLE_FTDI
    struct ftdi_context* ftdi = nullptr;
#endif
    int spi_fd = -1;

    explicit Impl(const LatticeConfig& cfg) : config(cfg) {}

    ~Impl() {
#ifdef XINFER_ENABLE_FTDI
        if (ftdi) {
            ftdi_usb_close(ftdi);
            ftdi_free(ftdi);
        }
#endif
        if (spi_fd >= 0) close(spi_fd);
    }

    // --------------------------------------------------------------------------
    // Connectivity Helpers
    // --------------------------------------------------------------------------
    bool connect() {
        if (config.interface == ConnectionInterface::USB_FTDI) {
#ifdef XINFER_ENABLE_FTDI
            ftdi = ftdi_new();
            if (!ftdi) return false;
            // Common Lattice FTDI VID/PID (0x0403, 0x6010 is standard FT2232)
            // In real app, make these configurable via config.device_address
            if (ftdi_usb_open(ftdi, 0x0403, 0x6010) < 0) {
                XINFER_LOG_ERROR("Failed to open FTDI device.");
                return false;
            }
            return true;
#else
            XINFER_LOG_ERROR("xInfer built without libftdi support.");
            return false;
#endif
        } 
        else if (config.interface == ConnectionInterface::SPI_DEV) {
            spi_fd = open(config.device_address.c_str(), O_RDWR);
            if (spi_fd < 0) {
                XINFER_LOG_ERROR("Failed to open SPI device: " + config.device_address);
                return false;
            }
            // Setup SPI Speed, Mode, etc. (Omitted for brevity)
            return true;
        }
        return false;
    }

    // --------------------------------------------------------------------------
    // Hardware Abstraction Layer (HAL)
    // --------------------------------------------------------------------------
    void write32(uint32_t address, uint32_t value) {
        // Prepare packet: [OpCode, Addr3, Addr2, Addr1, Addr0, Val3, Val2, Val1, Val0]
        // Implementation depends heavily on the specific "Soft CPU" or "Bridge" logic
        // inside your FPGA bitstream. 
        
        // Simulation for xInfer structure:
        // log_debug("Write32 Addr: " + hex(address) + " Val: " + hex(value));
    }

    uint32_t read32(uint32_t address) {
        // Send Read Command -> Receive 4 bytes
        return 0; // Placeholder
    }

    void burst_write(uint32_t address, const void* data, size_t size) {
        // Efficient bulk transfer logic
    }

    void burst_read(uint32_t address, void* data, size_t size) {
        // Efficient bulk transfer logic
    }
};

// =================================================================================
// 2. Public API Implementation
// =================================================================================

LatticeBackend::LatticeBackend(const LatticeConfig& config)
    : m_config(config), m_impl(std::make_unique<Impl>(config)) {
}

LatticeBackend::~LatticeBackend() = default;

bool LatticeBackend::load_model(const std::string& model_path) {
    // 1. Connect to HW
    if (!m_impl->connect()) return false;

    // 2. Load Neural Network Instructions (.bin)
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        XINFER_LOG_ERROR("Failed to open model file: " + model_path);
        return false;
    }

    size_t size = file.tellg();
    file.seekg(0);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);

    // 3. Upload to FPGA Instruction Memory (IMEM)
    // Assuming IMEM starts at 0x8000 for this hypothetical IP
    XINFER_LOG_INFO("Uploading NN commands to FPGA...");
    m_impl->burst_write(0x8000, buffer.data(), size);
    
    XINFER_LOG_INFO("Lattice sensAI initialized.");
    return true;
}

bool LatticeBackend::flash_bitstream(const std::string& bit_path) {
    XINFER_LOG_WARN("Runtime Bitstream flashing via FTDI/JTAG not implemented yet.");
    // Requires low-level JTAG state machine logic implementation
    return false;
}

void LatticeBackend::predict(const std::vector<core::Tensor>& inputs, 
                             std::vector<core::Tensor>& outputs) {
    
    // 1. Pre-process and Write Inputs
    // Lattice sensAI typically uses INT8 or INT16 input
    // We assume the model expects INT8 here.
    
    const auto& input = inputs[0];
    size_t num_elements = input.size();
    std::vector<int8_t> quantized_input(num_elements);
    
    const float* src_data = static_cast<const float*>(input.data());
    
    // Simple quantization (Real app should use calibration data)
    // Assuming range [-1, 1] mapped to [-128, 127]
    for(size_t i=0; i<num_elements; ++i) {
        quantized_input[i] = static_cast<int8_t>(src_data[i] * 127.0f);
    }

    // Write to FPGA Data Memory (DMEM)
    m_impl->burst_write(m_config.input_base_addr, quantized_input.data(), quantized_input.size());

    // 2. Trigger Execution
    // Write '1' to the Control Register Start Bit
    m_impl->write32(REG_CONTROL, CMD_START);

    // 3. Poll for Completion
    // Busy wait (Spinlock) - appropriate for low latency embedded
    int timeout = 10000;
    while (timeout-- > 0) {
        uint32_t status = m_impl->read32(REG_STATUS);
        if (status & STATUS_DONE) break;
        std::this_thread::sleep_for(std::chrono::microseconds(10)); // Yield slightly
    }

    if (timeout <= 0) {
        XINFER_LOG_ERROR("Lattice FPGA Timed Out.");
        return;
    }

    // 4. Read Outputs
    // Assuming Output is also INT8
    size_t out_elements = outputs[0].size();
    std::vector<int8_t> raw_output(out_elements);
    
    m_impl->burst_read(m_config.output_base_addr, raw_output.data(), raw_output.size());

    // Dequantize back to Float32 for xInfer pipeline
    // In real app, keep it int8 if possible for speed
    float* out_data = static_cast<float*>(outputs[0].data());
    for(size_t i=0; i<out_elements; ++i) {
        out_data[i] = static_cast<float>(raw_output[i]) / 127.0f;
    }
}

std::string LatticeBackend::device_name() const {
    return "Lattice FPGA (sensAI)";
}

// =================================================================================
// 3. Auto-Registration
// =================================================================================

namespace {
    volatile bool registered = xinfer::backends::BackendFactory::register_backend(
        xinfer::Target::LATTICE_SENSAI,
        [](const xinfer::Config& config) -> std::unique_ptr<xinfer::IBackend> {
            LatticeConfig lat_cfg;
            lat_cfg.model_path = config.model_path;
            
            for(const auto& param : config.vendor_params) {
                if(param == "INTERFACE=SPI") lat_cfg.interface = ConnectionInterface::SPI_DEV;
                if(param.find("DEV_ADDR=") != std::string::npos) {
                     lat_cfg.device_address = param.substr(9);
                }
            }
            
            auto backend = std::make_unique<LatticeBackend>(lat_cfg);
            if(backend->load_model(lat_cfg.model_path)) {
                return backend;
            }
            return nullptr;
        }
    );
}

} // namespace xinfer::backends::lattice