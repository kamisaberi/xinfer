# Apple Core ML Backend for xInfer

This backend executes inference on macOS and iOS devices, leveraging the **Apple Neural Engine (ANE)** and **Metal GPU**.

## ⚠️ Model Format Requirement

Core ML **does not** load `.mlmodel` files directly at runtime. It requires a compiled **`.mlmodelc`** (Model Directory).

### How to Compile Models

You must compile your models using the `xinfer-cli` (which wraps `xcrun coremlcompiler` or `coremltools`).

**Option 1: Using xInfer CLI (Recommended)**
```bash
xinfer-cli compile --target apple-coreml \
                   --onnx model.onnx \
                   --output model.mlmodelc \
                   --precision fp16
```

**Option 2: Manual Compilation (macOS only)**
If you have an `.mlmodel` file, you can compile it manually:
```bash
xcrun coremlcompiler compile MyModel.mlmodel ./output_folder
mv ./output_folder/MyModel.mlmodelc .
```

## ⚙️ C++ Usage

```cpp
xinfer::zoo::vision::DetectorConfig config;
config.target = xinfer::Target::APPLE_COREML;

// Point to the compiled folder, NOT a file
config.model_path = "models/yolov8.mlmodelc"; 

// Optional: Force ANE usage
config.vendor_params = { "COMPUTE_UNIT=ALL" }; 
```

## 📈 Performance Tips

*   **FP16 is King:** The Apple Neural Engine (ANE) runs almost exclusively in FP16. Always export your models with FP16 precision for a 2x-5x speedup over FP32.
*   **Batching:** Core ML prefers small batches (often Batch=1) for real-time applications. Large batches may force the model onto the CPU or GPU instead of the ANE.
