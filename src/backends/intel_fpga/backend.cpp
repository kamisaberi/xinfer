#include <xinfer/backends/intel_fpga/backend.h>
#include <xinfer/core/logging.h>
#include <xinfer/backends/backend_factory.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <algorithm>

// --- Intel FPGA / OpenCL Headers ---
// Using standard CL headers which are the backbone of Intel FPGA interaction
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#include <CL/cl.h>

namespace xinfer::backends::intel_fpga {

// =================================================================================
// 1. PImpl Implementation
// =================================================================================

struct IntelFpgaBackend::Impl {
    IntelFpgaConfig config;

    // OpenCL / Runtime Resources
    cl_platform_id platform_id = nullptr;
    cl_device_id device_id = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr; // Represents the Bitstream (.aocx)
    cl_kernel kernel = nullptr;   // Represents the DLA IP Core

    // DLA Graph Memory
    cl_mem graph_mem = nullptr;   // Buffer holding the .bin model

    explicit Impl(const IntelFpgaConfig& cfg) : config(cfg) {}

    ~Impl() {
        if (graph_mem) clReleaseMemObject(graph_mem);
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }

    // --- Helper: Initialize OpenCL Context ---
    bool init_opencl() {
        cl_int status;
        cl_uint num_platforms;
        
        // 1. Get Intel FPGA Platform
        status = clGetPlatformIDs(1, &platform_id, &num_platforms);
        if (status != CL_SUCCESS) return false;

        // 2. Get Device (Accelerator)
        status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ACCELERATOR, 1, &device_id, nullptr);
        if (status != CL_SUCCESS) {
            XINFER_LOG_ERROR("No Intel FPGA Accelerator found.");
            return false;
        }

        // 3. Create Context
        context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &status);
        if (status != CL_SUCCESS) return false;

        // 4. Create Command Queue
        queue = clCreateCommandQueue(context, device_id, 0, &status);
        return (status == CL_SUCCESS);
    }

    // --- Helper: Load Bitstream (.aocx) ---
    bool program_fpga(const std::string& bitstream_path) {
        if (bitstream_path.empty()) return true; // Assume already programmed

        std::ifstream file(bitstream_path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            XINFER_LOG_ERROR("Bitstream file not found: " + bitstream_path);
            return false;
        }

        size_t size = file.tellg();
        file.seekg(0);
        std::vector<unsigned char> binary(size);
        file.read(reinterpret_cast<char*>(binary.data()), size);

        cl_int status;
        const unsigned char* bins = binary.data();
        program = clCreateProgramWithBinary(context, 1, &device_id, &size, &bins, &status, nullptr);
        
        if (status != CL_SUCCESS) return false;

        status = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);
        if (status != CL_SUCCESS) return false;

        // Create the DLA Kernel (Name depends on the bitstream, usually "dla_inference")
        kernel = clCreateKernel(program, "dla_inference", &status);
        return (status == CL_SUCCESS);
    }

    // --- Helper: Load Model Graph (.bin) ---
    bool load_graph_binary(const std::string& model_path) {
        std::ifstream file(model_path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) return false;

        size_t size = file.tellg();
        file.seekg(0);
        
        std::vector<char> host_buffer(size);
        file.read(host_buffer.data(), size);

        cl_int status;
        // Allocate buffer on FPGA DDR for the graph instructions
        graph_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, host_buffer.data(), &status);
        
        return (status == CL_SUCCESS);
    }
};

// =================================================================================
// 2. Public API Implementation
// =================================================================================

IntelFpgaBackend::IntelFpgaBackend(const IntelFpgaConfig& config)
    : m_config(config), m_impl(std::make_unique<Impl>(config)) {
}

IntelFpgaBackend::~IntelFpgaBackend() = default;

bool IntelFpgaBackend::load_model(const std::string& model_path) {
    // 1. Initialize Runtime
    if (!m_impl->init_opencl()) {
        XINFER_LOG_ERROR("Failed to initialize Intel FPGA OpenCL context.");
        return false;
    }

    // 2. Program Bitstream (if provided)
    if (!m_config.bitstream_path.empty()) {
        if (!m_impl->program_fpga(m_config.bitstream_path)) {
            XINFER_LOG_ERROR("Failed to program FPGA with bitstream.");
            return false;
        }
        XINFER_LOG_INFO("FPGA programmed successfully.");
    } else {
        XINFER_LOG_INFO("Using existing FPGA bitstream.");
        // Note: In real app, we'd need to attach to existing kernel here if not programming
    }

    // 3. Load DLA Graph
    if (!m_impl->load_graph_binary(model_path)) {
        XINFER_LOG_ERROR("Failed to load DLA graph binary: " + model_path);
        return false;
    }

    XINFER_LOG_INFO("Loaded Intel FPGA DLA Model: " + model_path);
    return true;
}

bool IntelFpgaBackend::program_bitstream(const std::string& aocx_path) {
    return m_impl->program_fpga(aocx_path);
}

void IntelFpgaBackend::predict(const std::vector<core::Tensor>& inputs, 
                               std::vector<core::Tensor>& outputs) {
    
    // Simplification: In a real DLA usage, we map buffers to args 0, 1, 2...
    // Input Buffer -> Kernel Arg 0
    // Output Buffer -> Kernel Arg 1
    // Graph Buffer -> Kernel Arg 2

    cl_int status;

    // 1. Prepare Input Buffers
    // For high performance (Aegis Sky), we assume SVM (Shared Virtual Memory)
    // allowing Zero-Copy.
    
    // Fallback: Create CL Buffers and Copy
    size_t input_size = inputs[0].size() * sizeof(float);
    cl_mem input_mem = clCreateBuffer(m_impl->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                      input_size, (void*)inputs[0].data(), &status);

    // 2. Prepare Output Buffers
    size_t output_size = outputs[0].size() * sizeof(float); // Need actual size logic
    cl_mem output_mem = clCreateBuffer(m_impl->context, CL_MEM_WRITE_ONLY, 
                                       output_size, nullptr, &status);

    // 3. Set Kernel Arguments
    clSetKernelArg(m_impl->kernel, 0, sizeof(cl_mem), &input_mem);
    clSetKernelArg(m_impl->kernel, 1, sizeof(cl_mem), &output_mem);
    clSetKernelArg(m_impl->kernel, 2, sizeof(cl_mem), &m_impl->graph_mem); // The model logic

    // 4. Enqueue Kernel
    // This triggers the DLA IP to run the graph instructions
    status = clEnqueueTask(m_impl->queue, m_impl->kernel, 0, nullptr, nullptr);
    
    if (status != CL_SUCCESS) {
        XINFER_LOG_ERROR("FPGA Kernel Execution Failed.");
    } else {
        // 5. Read Result
        clEnqueueReadBuffer(m_impl->queue, output_mem, CL_TRUE, 0, output_size, 
                            outputs[0].data(), 0, nullptr, nullptr);
    }

    // Cleanup ephemeral buffers (in real app, reuse these!)
    clReleaseMemObject(input_mem);
    clReleaseMemObject(output_mem);
}

std::string IntelFpgaBackend::device_name() const {
    return "Intel FPGA DLA (OpenCL)";
}

// =================================================================================
// 3. Auto-Registration
// =================================