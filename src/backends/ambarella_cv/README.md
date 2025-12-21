# Ambarella CVFlow Backend for xInfer

This backend enables hardware acceleration on Ambarella CV2x, CV5x, and CV3 (AI Domain Controller) chips using the **Cavalry** vector processor.

## ⚠️ Proprietary SDK Requirement

This module requires the **Ambarella CVFlow SDK**, which is **NOT** open source. You must download it from the Ambarella Cooper Portal (NDA required).

### Installation Setup

1.  Download the SDK (e.g., `cvflow_sdk_3.0.tar.gz`).
2.  Extract it to a folder on your system (e.g., `/opt/ambarella/cvflow`).
3.  Set the environment variable before building `xInfer`:

    ```bash
    export AMBA_CVFLOW_SDK=/opt/ambarella/cvflow
    ```

### Hardware Setup (Aegis Sky)

For the **Aegis Sky** project, ensure the kernel modules are loaded on the target device before running the application:

```bash
# On the Target Device (Ambarella Board)
modprobe cavalry
modprobe cma_heap
```

### Supported Models

This backend accepts **`.cavalry`** binary files produced by `CNNGen`.

```cpp
xinfer::zoo::vision::DetectorConfig config;
config.target = xinfer::Target::AMBARELLA_CV;
config.model_path = "/usr/local/bin/yolov8_ad.cavalry";
config.vendor_params = { "VP_INSTANCE=0", "PRIORITY=99" };
```
```

---

### 3. One more C++ Detail: The Auto-Registration

How does `xInfer` know this backend exists? You usually need a small snippet to register it with your `BackendFactory`. You can add this to the bottom of your `backend.cpp` or create a tiny `provider.cpp`.

**Add this to the bottom of `backend.cpp`:**

```cpp
// ... inside src/backends/ambarella_cv/backend.cpp

// Automatic Registration
#include <xinfer/backends/backend_factory.h>

namespace {
    // This static block runs at startup
    volatile bool registered = xinfer::backends::BackendFactory::register_backend(
        xinfer::Target::AMBARELLA_CV,
        [](const xinfer::Config& config) -> std::unique_ptr<xinfer::IBackend> {
            // Convert generic config to Ambarella config
            AmbarellaConfig amba_cfg;
            amba_cfg.model_path = config.model_path;
            
            // Parse vendor specific flags
            for(const auto& param : config.vendor_params) {
                if(param.find("VP_INSTANCE=") != std::string::npos) {
                     amba_cfg.vp_instance_id = std::stoi(param.substr(12));
                }
            }
            
            return std::make_unique<AmbarellaBackend>(amba_cfg);
        }
    );
}
```

### Summary of Folder Structure
Now your `backends/ambarella_cv/` is complete:

```text
backends/ambarella_cv/
├── CMakeLists.txt      <-- Build logic (Checks for SDK)
├── README.md           <-- Instructions for NDA setup
├── backend.cpp         <-- Implementation + Registration
├── backend.h           <-- Public Interface
├── config.h            <-- Hardware struct
└── types.h             <-- Error codes / Precision enums
```