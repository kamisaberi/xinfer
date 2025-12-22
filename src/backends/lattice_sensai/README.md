# Lattice sensAI Backend for xInfer

This backend allows `xInfer` to act as a Host Controller for Lattice FPGAs (CrossLink-NX, Certus-NX, iCE40) running the **sensAI CNN Accelerator IP**.

Since these FPGAs typically do not run a full OS, `xInfer` communicates with them over standard buses (USB/FTDI or SPI) to offload inference tasks.

## 🛠️ Prerequisites

1.  **Host Dependencies:**
    ```bash
    sudo apt install libftdi1-dev
    ```
2.  **FPGA Configuration:**
    The Lattice FPGA must be pre-programmed with a bitstream containing:
    *   The **CNN Compact Accelerator** IP.
    *   A control bridge (e.g., RISC-V soft core or SPI Slave to Wishbone bridge) that maps the IP's registers to the external bus.

## ⚙️ Workflow

1.  **Train & Compile:**
    *   Train in TensorFlow/Keras.
    *   Use **Lattice Neural Network Compiler** (from the sensAI stack) to generate the Firmware/Command binary (`.bin`).

2.  **Convert for xInfer:**
    Simply rename the output `.bin` command stream to match your xInfer model path.

3.  **Run xInfer:**
    
    **USB Mode (Standard Dev Boards):**
    ```cpp
    xinfer::zoo::vision::DetectorConfig config;
    config.target = xinfer::Target::LATTICE_SENSAI;
    config.model_path = "models/face_detect_cmd.bin";
    config.vendor_params = { "INTERFACE=USB" };
    ```

    **SPI Mode (Embedded Linux Host):**
    ```cpp
    config.vendor_params = { "INTERFACE=SPI", "DEV_ADDR=/dev/spidev0.0" };
    ```

## ⚠️ Addressing Information

Since memory maps vary based on how you build your FPGA design in Lattice Propel, you may need to modify `input_base_addr` and `output_base_addr` in `LatticeConfig` to match your specific system-on-chip address map.
