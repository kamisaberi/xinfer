#include <xinfer/backends/ambarella_cv/backend.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/utils.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <stdexcept>

// Linux System Headers for interacting with the VP Driver
#include <fcntl.h>      // open
#include <unistd.h>     // close, read, write
#include <sys/ioctl.h>  // ioctl
#include <sys/mman.h>   // mmap

namespace xinfer::backends::ambarella {

// =================================================================================
// 1. PROPRIETARY IMPLEMENTATION (The "PImpl" Hidden Class)
// =================================================================================
// This struct hides all the NDA-specific headers and dirty driver logic
// from the public API.
struct AmbarellaBackend::Impl {
    int fd_cavalry = -1;       // File descriptor for /dev/cavalry
    void* cma_mem_virt = nullptr; // Virtual address of the mapped CMA memory
    uintptr_t cma_mem_phys = 0;   // Physical address (needed by the VP hardware)
    size_t cma_size = 0;

    // Internal handles for the loaded DAG (Directed Acyclic Graph)
    uint32_t net_id = 0;

    // Pointers into the CMA pool for inputs/outputs
    void* input_dram_ptr = nullptr;
    void* output_dram_ptr = nullptr;

    explicit Impl(const AmbarellaConfig& cfg) {
        cma_size = cfg.memory_pool_size;
        initialize_driver(cfg);
    }

    ~Impl() {
        if (cma_mem_virt) {
            munmap(cma_mem_virt, cma_size);
        }
        if (fd_cavalry >= 0) {
            close(fd_cavalry);
        }
    }

    // --- Internal Helper: Open the Driver ---
    void initialize_driver(const AmbarellaConfig& cfg) {
        // In a real Ambarella system, this is usually /dev/cavalry_vmem or similar
        fd_cavalry = open("/dev/cavalry", O_RDWR);

        if (fd_cavalry < 0) {
            XINFER_LOG_ERROR("Failed to open Ambarella Cavalry driver. Are you running on the target HW?");
            // For development on PC, we might allow a 'simulation' mode,
            // but for this example, we assume hardware presence.
            return;
        }

        // Request CMA (Contiguous Memory) from the driver
        // This is a pseudo-ioctl call representing typical embedded logic
        // ioctl(fd_cavalry, CAVALRY_ALLOC_MEM, &cma_info);

        // Map the physical memory to user-space so the CPU can write images to it
        cma_mem_virt = mmap(nullptr, cma_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_cavalry, 0);

        if (cma_mem_virt == MAP_FAILED) {
            throw std::runtime_error("Ambarella Backend: mmap failed for CMA pool.");
        }

        XINFER_LOG_INFO("Ambarella VP Driver Initialized. CMA Pool: " + std::to_string(cma_size / 1024 / 1024) + " MB");
    }

    // --- Internal Helper: Load Binary ---
    bool load_cavalry_binary(const std::string& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) return false;

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size)) return false;

        // In real SDK: ioctl(fd_cavalry, CAVALRY_LOAD_DAG, &load_struct);
        // We simulate assigning pointers here:

        // Hypothetical: The first 1MB of CMA is for Input, the second for Output
        // (Real SDKs use query_network_io() to find exact offsets)
        input_dram_ptr = static_cast<char*>(cma_mem_virt);
        output_dram_ptr = static_cast<char*>(cma_mem_virt) + (1024 * 1024 * 4); // 4MB offset

        XINFER_LOG_INFO("Loaded Cavalry Binary: " + path);
        return true;
    }

    // --- Internal Helper: Execute ---
    void execute_hardware() {
        // In real SDK: ioctl(fd_cavalry, CAVALRY_RUN_DAG, &run_struct);
        // This blocks until the VP (Vector Processor) finishes.

        // Simulation delay for "hardware execution"
        // usleep(2000);
    }
};

// =================================================================================
// 2. PUBLIC API IMPLEMENTATION
// =================================================================================

AmbarellaBackend::AmbarellaBackend(const AmbarellaConfig& config)
    : m_config(config), m_impl(std::make_unique<Impl>(config)) {
}

AmbarellaBackend::~AmbarellaBackend() = default;

bool AmbarellaBackend::load_model(const std::string& model_path) {
    if (m_impl->fd_cavalry < 0) {
        XINFER_LOG_ERROR("Driver not initialized.");
        return false;
    }
    return m_impl->load_cavalry_binary(model_path);
}

void AmbarellaBackend::predict(const std::vector<core::Tensor>& inputs,
                               std::vector<core::Tensor>& outputs) {

    // Step 1: Zero-Copy Handling (The critical part for Aegis Sky)
    // -----------------------------------------------------------
    // If the input Tensor is already in CMA memory (e.g., from the Camera driver),
    // we should pass the physical address directly to the VP.
    // If it's in standard CPU RAM, we must memcpy it to the VP's DRAM.

    if (inputs.empty()) return;

    // For simplicity, assuming 1 input, 1 output for this example
    const auto& input = inputs[0];

    // Check if memory copy is needed
    if (input.memory_type() == core::MemoryType::CmaContiguous) {
        // Ideal Path: Zero Copy
        // We would update the DAG input pointer to point to input.physical_address()
        // m_impl->update_input_phys_addr(input.physical_address());
    } else {
        // Slow Path: CPU -> CMA Copy
        // This is necessary if data comes from OpenCV or Network
        size_t size_bytes = input.size() * core::element_size(input.dtype());
        std::memcpy(m_impl->input_dram_ptr, input.data(), size_bytes);

        // Flush CPU Cache so the hardware sees the new data
        // (In C++ we might need a cache flush helper here)
        // xinfer::core::flush_cache(m_impl->input_dram_ptr, size_bytes);
    }

    // Step 2: Trigger Hardware Execution
    // -----------------------------------------------------------
    m_impl->execute_hardware();

    // Step 3: Retrieve Results
    // -----------------------------------------------------------
    // Ambarella VP writes results directly to CMA output buffer.
    // We wrap this memory in a Tensor or copy it back.

    if (!outputs.empty()) {
        auto& output = outputs[0];

        // If the implementation allows zero-copy output:
        // output.set_external_data(m_impl->output_dram_ptr, size, ...);

        // Otherwise, copy back to Host
        size_t out_bytes = output.size() * core::element_size(output.dtype());
        std::memcpy(output.data(), m_impl->output_dram_ptr, out_bytes);

        // Invalidate Cache so CPU sees new data
        // xinfer::core::invalidate_cache(output.data(), out_bytes);
    }
}

std::string AmbarellaBackend::device_name() const {
    return "Ambarella CVFlow (VP" + std::to_string(m_config.vp_instance_id) + ")";
}

} // namespace xinfer::backends::ambarella