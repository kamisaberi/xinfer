# Intel OpenVINO Backend for xInfer

This backend enables high-performance inference on Intel hardware, including **CPUs** (Xeon, Core), **Integrated GPUs** (Iris Xe), **Discrete GPUs** (Arc), and **NPUs** (Core Ultra / Meteor Lake).

## 🛠️ Installation

You need the OpenVINO Runtime installed on your system.

### Option 1: Linux (APT)
```bash
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
echo "deb https://apt.repos.intel.com/openvino/2024 ubuntu22 main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2024.list
sudo apt update
sudo apt install openvino-libraries-dev
```

### Option 2: Windows / MacOS
Download the installer from [Intel's Website](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html).

## ⚙️ Model Conversion

OpenVINO uses an "Intermediate Representation" (IR) format consisting of an `.xml` (topology) and `.bin` (weights) file.

**Using xInfer CLI:**
```bash
xinfer-cli compile --target intel-ov \
                   --onnx model.onnx \
                   --output model.xml \
                   --precision fp16
```
*(This wraps the `ovc` tool internally)*

## 💻 C++ Usage

```cpp
xinfer::zoo::vision::DetectorConfig config;
config.target = xinfer::Target::INTEL_OV;
config.model_path = "models/yolov8.xml";

// Optional: Force execution on the Integrated GPU
config.vendor_params = { "DEVICE=GPU", "HINT=THROUGHPUT" };

// Optional: Use the new NPU (Neural Processing Unit) on Meteor Lake laptops
// config.vendor_params = { "DEVICE=NPU" };
```

## ⚠️ Performance Tips

*   **Throughput vs Latency:**
    *   For **Blackbox SIEM** (processing logs/packets in bulk), set `HINT=THROUGHPUT`. This batches requests to saturate CPU cores.
    *   For **Real-time Video**, set `HINT=LATENCY`.
*   **First-Inference Latency:** OpenVINO compiles kernels (JIT) on the first run. Use the `cache_dir` option in config to save these kernels to disk (`cl_cache`), making subsequent startups instant.
