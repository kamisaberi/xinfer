# AMD Vitis AI Backend for xInfer

This backend executes models on Xilinx FPGA hardware (Zynq UltraScale+, Kria, Versal) using the **Vitis AI Runtime (VART)**.

## ⚠️ Hardware Requirements

This backend is designed to run on the **Target Device** (Edge), not on your x86 Host (unless you have an Alveo card).

*   **Supported Boards:** Kria KV260/KR260, ZCU102, ZCU104, Versal VCK190.
*   **System:** PetaLinux or Ubuntu 22.04 (Kria-Ubuntu).

## 🛠️ Installation on Target (Kria/Zynq)

1.  **Install Vitis-AI Runtime (VART):**
    On the Kria board (Ubuntu), install the runtime via apt:
    ```bash
    sudo apt update
    sudo apt install xrt vart vitis-ai-library
    ```

2.  **Load DPU Firmware (Bitstream):**
    Ensure the DPU overlay is loaded. On Kria, use `xmutil`:
    ```bash
    sudo xmutil unloadapp
    sudo xmutil loadapp kv260-smartcam
    ```
    *(Note: 'kv260-smartcam' contains the B4096 DPU fingerprint)*

3.  **Build xInfer:**
    ```bash
    mkdir build && cd build
    cmake .. -DXINFER_ENABLE_VITIS=ON
    make -j
    ```

## ⚙️ Usage

### 1. Compile Model (On Host PC)
You must compile your ONNX model to `.xmodel` using the **Vitis AI Compiler** inside the Docker container on your host machine.

```bash
# On Host PC
xinfer-cli compile --target amd-vitis --onnx model.onnx --output model.xmodel --vendor-params DPU_ARCH=DPUCZDX8G_ISA1_B4096
```

### 2. Run Inference (On Target Board)

Transfer the `.xmodel` and the `xinfer` binary to the board.

```cpp
xinfer::zoo::vision::DetectorConfig config;
config.target = xinfer::Target::AMD_VITIS;
config.model_path = "model.xmodel";
```

## 📈 Performance Tips

*   **Batching:** Vitis AI supports multi-threading. Set `vendor_params = {"NUM_RUNNERS=4"}` to use 4 CPU threads driving the DPU simultaneously.
*   **Profiling:** Run your application with `vitis_ai_profiler ./my_app` to see the layer-by-layer execution time on the DPU hardware.
