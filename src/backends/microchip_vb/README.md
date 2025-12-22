# Microchip VectorBlox Backend for xInfer

This backend executes models on **Microchip PolarFire SoC** and **PolarFire FPGA** devices. It interfaces with the **VectorBlox CNN IP Core**.

## 🛠️ Prerequisites

1.  **VectorBlox SDK:**
    You must have the [VectorBlox SDK](https://github.com/Microchip-FPGA-Tools/VectorBlox) available.
    Set the path: `export VECTORBLOX_SDK=/path/to/sdk`.

2.  **FPGA Bitstream:**
    The FPGA must be programmed with a Libero design containing the VBX IP (e.g., the `V1000` configuration).

## ⚙️ Model Conversion

VectorBlox uses **`.vnnx`** files (OpenVINO intermediate) converted to a binary blob.

1.  **Quantize:** Use the VectorBlox calibration tools to convert your model to INT8.
2.  **Convert to BLOB:** Use the `vnnx_to_blob` script provided by the SDK.

**xInfer CLI Workflow:**
```bash
xinfer-cli compile --target microchip-vb \
                   --onnx model.onnx \
                   --output model.blob \
                   --precision int8
```

## ⚠️ Cache Coherency (Crucial)

On the **PolarFire SoC**, the MSS (Microprocessor Subsystem) and the FPGA Fabric share DDR memory via the AXI bus.
*   **Problem:** The CPU L1/L2 caches are NOT automatically coherent with the FPGA.
*   **Solution:** xInfer attempts to flush caches manually. If you experience "garbage outputs," ensure your Linux kernel is configured to map the tensor memory as `Non-Cached` or ensure `xInfer` is running with root privileges to execute cache maintenance instructions.
