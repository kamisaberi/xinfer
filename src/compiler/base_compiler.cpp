#include <xinfer/compiler/base_compiler.h>
#include <algorithm>
#include <iostream>

namespace xinfer {

// Helper to lowercase strings
static std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return std::tolower(c); });
    return s;
}

namespace compiler {

Target stringToTarget(const std::string& s) {
    std::string key = to_lower(s);
    
    if (key == "nv-trt" || key == "nvidia") return Target::NVIDIA_TRT;
    if (key == "intel-ov" || key == "openvino") return Target::INTEL_OV;
    if (key == "amd-ryzen" || key == "ryzen-ai") return Target::AMD_RYZEN_AI;
    if (key == "apple" || key == "coreml") return Target::APPLE_COREML;
    
    if (key == "amd-vitis" || key == "xilinx") return Target::AMD_VITIS;
    if (key == "intel-fpga") return Target::INTEL_FPGA;
    if (key == "microchip" || key == "vectorblox") return Target::MICROCHIP_VECTORBLOX;
    if (key == "lattice") return Target::LATTICE_SENSAI;
    
    if (key == "qcom" || key == "qnn") return Target::QUALCOMM_QNN;
    if (key == "rockchip" || key == "rknn") return Target::ROCKCHIP_RKNN;
    if (key == "mediatek" || key == "neuropilot") return Target::MEDIATEK_NEUROPILOT;
    if (key == "samsung" || key == "exynos") return Target::SAMSUNG_EXYNOS;
    
    if (key == "hailo") return Target::HAILO_RT;
    if (key == "ambarella") return Target::AMBARELLA_CV;
    if (key == "tpu" || key == "coral") return Target::GOOGLE_TPU;

    throw std::runtime_error("Unknown target platform: " + s);
}

Precision stringToPrecision(const std::string& s) {
    std::string key = to_lower(s);
    if (key == "fp32" || key == "float") return Precision::FP32;
    if (key == "fp16" || key == "half") return Precision::FP16;
    if (key == "int8" || key == "quantized") return Precision::INT8;
    if (key == "int4") return Precision::INT4;
    return Precision::FP16; // Default safe fallback
}

} // namespace compiler
} // namespace xinfer