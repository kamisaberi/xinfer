# Intel FPGA Backend for xInfer

This backend executes models on **Intel Agilex**, **Stratix 10**, and **Arria 10** FPGAs. It interfaces with the **Deep Learning Accelerator (DLA)** IP Core via the OpenCL runtime.

## 🛠️ Prerequisites

1.  **Intel FPGA AI Suite:**
    You must have the Intel FPGA AI Suite installed (part of the Intel FPGA SDK).
2.  **OpenCL Runtime:**
    The BSP (Board Support Package) for your specific FPGA card must be installed and the OpenCL drivers loaded.

## ⚙️ Model Compilation

The workflow for Intel FPGA is two-staged:

1.  **Bitstream (.aocx):** This configures the FPGA fabric with the DLA IP Core. This is usually done once per system boot.
2.  **Graph Binary (.bin):** This is your neural network compiled into instructions for the DLA.

**Compile ONNX to DLA Binary:**
```bash
xinfer-cli compile --target intel-fpga \
                   --onnx model.onnx \
                   --output model.bin \
                   --precision int8
```

## 💻 C++ Usage

```cpp
xinfer::zoo::vision::DetectorConfig config;
config.target = xinfer::Target::INTEL_FPGA;

// The compiled graph instructions
config.model_path = "models/resnet50.bin";

// Optional: If you need to program the FPGA bitstream at runtime
// (WARNING: This takes several seconds)
config.vendor_params = { "BITSTREAM=/opt/intel/dla_bitstreams/agilex_dla.aocx" };
```

## ⚠️ Performance Note

*   **Zero-Copy (SVM):** For maximum performance (like **Aegis Sky** tracking), ensure your FPGA Board Support Package (BSP) supports **Shared Virtual Memory (SVM)**. This allows the CPU to write image data to a pointer that the FPGA reads directly via PCIe, skipping the `clEnqueueWriteBuffer` copy step.
