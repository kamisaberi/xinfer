# MediaTek NeuroPilot Backend for xInfer

This backend executes models on **MediaTek Genio** and **Dimensity** SoCs using the **Neuron Runtime**.

## 🛠️ Prerequisites

1.  **Hardware:** A MediaTek Genio board (e.g., Genio 1200 EVK) or Dimensity phone.
2.  **SDK:** The NeuroPilot SDK (Neuron) libraries must be present in the system image (`/usr/lib/libneuronusdk_adapter.so`).

## ⚙️ Model Conversion

MediaTek uses a proprietary binary format (`.dla` or `.pte`).

**Compile TFLite/ONNX to NeuroPilot:**
You must use the `ncc` (Neuron Compiler) tool provided by MediaTek.

```bash
# Using xInfer CLI (wraps ncc)
xinfer-cli compile --target mediatek-np \
                   --onnx model.onnx \
                   --output model.dla \
                   --precision int8
```

## 💻 C++ Usage

```cpp
xinfer::zoo::vision::DetectorConfig config;
config.target = xinfer::Target::MEDIATEK_NEUROPILOT;
config.model_path = "models/mobilenet.dla";

// Optional: Optimize for low latency (Aegis Sky)
config.vendor_params = { "PREF=LATENCY" };
```

## ⚠️ Performance Note

*   **Quantization:** The APU is heavily optimized for INT8. FP32 models will often run on the CPU or run very slowly on the APU. Always use quantized models.
*   **Boost Mode:** For critical sections in **Aegis Sky**, you can trigger the scheduler boost to force max clocks, but watch thermal throttling.
