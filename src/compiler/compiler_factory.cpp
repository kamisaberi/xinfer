#include <xinfer/compiler/compiler_factory.h>
#include <xinfer/core/logging.h>

// --- Include All Driver Headers ---
// Desktop / Server
#include <xinfer/compiler/drivers/trt_driver.h>
#include <xinfer/compiler/drivers/ov_driver.h>
#include <xinfer/compiler/drivers/ryzen_ai_driver.h>
#include <xinfer/compiler/drivers/coreml_driver.h>

// FPGA / Adaptive
#include <xinfer/compiler/drivers/vitis_driver.h>
#include <xinfer/compiler/drivers/intel_fpga_driver.h>
#include <xinfer/compiler/drivers/vbx_driver.h>
#include <xinfer/compiler/drivers/sensai_driver.h>

// Mobile / SoC
#include <xinfer/compiler/drivers/qnn_driver.h>
#include <xinfer/compiler/drivers/rknn_driver.h>
#include <xinfer/compiler/drivers/neuropilot_driver.h>
#include <xinfer/compiler/drivers/exynos_driver.h>

// Specialized Edge
#include <xinfer/compiler/drivers/hailo_driver.h>
#include <xinfer/compiler/drivers/cavalry_driver.h>
#include <xinfer/compiler/drivers/edgetpu_driver.h>

namespace xinfer::compiler {

std::unique_ptr<ICompiler> CompilerFactory::create(Target target) {
    switch (target) {
        // --- Desktop & Server ---
        case Target::NVIDIA_TRT:
            return std::make_unique<TrtDriver>();
            
        case Target::INTEL_OV:
            return std::make_unique<OpenVINODriver>();

        case Target::AMD_RYZEN_AI:
            return std::make_unique<RyzenAIDriver>();

        case Target::APPLE_COREML:
            return std::make_unique<CoreMLDriver>();

        // --- FPGA & Adaptive (Aegis Sky) ---
        case Target::AMD_VITIS:
            return std::make_unique<VitisDriver>();

        case Target::INTEL_FPGA:
            return std::make_unique<IntelFpgaDriver>();

        case Target::MICROCHIP_VECTORBLOX:
            return std::make_unique<VectorBloxDriver>();

        case Target::LATTICE_SENSAI:
            return std::make_unique<LatticeDriver>();

        // --- Mobile & SoC ---
        case Target::QUALCOMM_QNN:
            return std::make_unique<QnnDriver>();

        case Target::ROCKCHIP_RKNN:
            return std::make_unique<RknnDriver>();

        case Target::MEDIATEK_NEUROPILOT:
            return std::make_unique<NeuroPilotDriver>();

        case Target::SAMSUNG_EXYNOS:
            return std::make_unique<ExynosDriver>();

        // --- Specialized Edge (Blackbox SIEM / Vision) ---
        case Target::HAILO_RT:
            return std::make_unique<HailoDriver>();

        case Target::AMBARELLA_CV:
            return std::make_unique<AmbarellaDriver>();

        case Target::GOOGLE_TPU:
            return std::make_unique<EdgeTpuDriver>();

        default:
            XINFER_LOG_ERROR("CompilerFactory: Unknown or unimplemented target platform.");
            return nullptr;
    }
}

} // namespace xinfer::compiler