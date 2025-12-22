#pragma once

#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <stdexcept>

// Forward declare common types if they are in core headers
// (Or include them here if they are small enums)
namespace xinfer {

// --- 1. Common Enums for the Compiler ---

/**
 * @brief Target Hardware Platform
 * Used to select the correct driver via the Factory.
 */
enum class Target {
    // Desktop / Server
    NVIDIA_TRT,      // .engine
    INTEL_OV,        // .xml / .bin
    AMD_RYZEN_AI,    // .dll (Vitis EP)
    APPLE_COREML,    // .mlmodelc
    
    // FPGA / Adaptive
    AMD_VITIS,       // .xmodel (Xilinx DPU)
    INTEL_FPGA,      // .bin (AI Suite DLA)
    MICROCHIP_VECTORBLOX, // .blob
    LATTICE_SENSAI,  // .bin
    
    // Mobile / SoC
    QUALCOMM_QNN,    // .so / .bin
    ROCKCHIP_RKNN,   // .rknn
    MEDIATEK_NEUROPILOT, // .pte
    SAMSUNG_EXYNOS,  // .nnc
    
    // Specialized Edge
    HAILO_RT,        // .hef
    AMBARELLA_CV,    // .cavalry
    GOOGLE_TPU       // .tflite
};

/**
 * @brief Math Precision for the compiled engine.
 */
enum class Precision {
    FP32 = 0,
    FP16 = 1,  // Half-precision (Standard for GPU/NPU)
    INT8 = 2,  // Quantized (Requires Calibration)
    INT4 = 3,  // Ultra-low precision (FPGA/Blackwell)
    BF16 = 4   // Brain Float (TPU/Newer ARM)
};

// --- 2. Configuration Struct ---

/**
 * @brief Universal Compile Configuration
 * Holds all the data a driver needs to build a model.
 */
struct CompileConfig {
    // Input Source
    std::string input_path;        // Path to local .onnx or .pt file
    std::string onnx_url;          // Optional: URL to download from

    // Target Output
    Target target;                 // Which chip are we building for?
    std::string output_path;       // Where to save the result

    // Optimization Settings
    Precision precision = Precision::FP16;
    
    // Calibration Data (Required for INT8)
    // Path to a folder of images or a calibration cache file.
    std::string calibration_data_path;

    // Vendor Specific Flags
    // Allows passing arbitrary flags to the underlying tool.
    // e.g., "WORKSPACE=1024", "DPU_ARCH=B4096", "CORE=0"
    std::vector<std::string> vendor_params;

    // Helper: Check if a vendor param exists
    bool has_param(const std::string& key) const {
        for(const auto& p : vendor_params) {
            if (p.find(key) != std::string::npos) return true;
        }
        return false;
    }
};

// --- 3. The Abstract Interface ---

namespace compiler {

/**
 * @brief Interface for all AI Compiler Drivers.
 * 
 * Every driver (TrtDriver, RknnDriver, etc.) must implement this class.
 */
class ICompiler {
public:
    virtual ~ICompiler() = default;

    /**
     * @brief The main entry point.
     * Takes the config, orchestrates the external tool (trtexec, rknn-toolkit, etc.),
     * and produces the engine file.
     * 
     * @return true if compilation succeeded and output file exists.
     */
    virtual bool compile(const CompileConfig& config) = 0;

    /**
     * @brief Checks if the environment is ready.
     * e.g., Checks if 'trtexec' is in PATH or if the Vitis Docker is present.
     * 
     * @return true if the toolchain is installed.
     */
    virtual bool validate_environment() = 0;

    /**
     * @brief Returns the friendly name of the driver.
     * e.g., "NVIDIA TensorRT Driver" or "Rockchip RKNN Toolkit"
     */
    virtual std::string get_name() const = 0;

protected:
    // Helper: Utility to execute shell commands and capture output/errors.
    // Useful for all drivers since they all "shell out" to CLI tools.
    int exec_cmd(const std::string& cmd) const {
        return std::system(cmd.c_str());
    }
};

// --- 4. Helper Functions (Optional) ---

// String conversion helpers (implementation in .cpp)
Target stringToTarget(const std::string& s);
Precision stringToPrecision(const std::string& s);

} // namespace compiler
} // namespace xinfer