#include <xinfer/backends/google_tpu/backend.h>
#include <xinfer/core/logging.h>
#include <xinfer/backends/backend_factory.h>

// --- TensorFlow Lite Headers ---
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/tools/gen_op_registration.h>

// --- libedgetpu Headers ---
// Note: This header comes from the libedgetpu-dev package
#include <edgetpu.h>

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace xinfer::backends::google_tpu {

// =================================================================================
// 1. PImpl Implementation
// =================================================================================

struct EdgeTpuBackend::Impl {
    EdgeTpuConfig config;

    // TFLite Resources
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;

    // The TPU Delegate
    // Note: edgetpu::EdgeTpuContext manages the pointer lifetime in newer APIs,
    // but the raw C-delegate is needed for the Interpreter.
    std::shared_ptr<edgetpu::EdgeTpuContext> tpu_context;
    TfLiteDelegate* tpu_delegate = nullptr;

    explicit Impl(const EdgeTpuConfig& cfg) : config(cfg) {}

    ~Impl() {
        // Order matters: Interpreter uses Delegate, Delegate uses Context.
        interpreter.reset();

        // Clean up delegate
        if (tpu_delegate) {
            // Note: libedgetpu usually provides a specific deleter or
            // relies on the context. For standard TFLite delegates:
            // This specific cleanup depends on the libedgetpu version.
            // Modern C++ API manages it via shared_ptr context usually.
        }
    }

    // --------------------------------------------------------------------------
    // Helper: Quantize (Float32 -> INT8/UINT8)
    // --------------------------------------------------------------------------
    void quantize_input(int tensor_idx, const core::Tensor& src) {
        TfLiteTensor* target = interpreter->tensor(tensor_idx);

        const float* src_data = static_cast<const float*>(src.data());
        size_t count = src.size();

        float scale = target->params.scale;
        int32_t zero_point = target->params.zero_point;

        if (target->type == kTfLiteUInt8) {
            uint8_t* dst_data = target->data.uint8;
            for (size_t i = 0; i < count; ++i) {
                int32_t val = static_cast<int32_t>(std::round(src_data[i] / scale) + zero_point);
                dst_data[i] = static_cast<uint8_t>(std::max(0, std::min(255, val)));
            }
        } else if (target->type == kTfLiteInt8) {
            int8_t* dst_data = target->data.int8;
            for (size_t i = 0; i < count; ++i) {
                int32_t val = static_cast<int32_t>(std::round(src_data[i] / scale) + zero_point);
                dst_data[i] = static_cast<int8_t>(std::max(-128, std::min(127, val)));
            }
        } else if (target->type == kTfLiteFloat32) {
            // Fallback if input layer wasn't quantized
            std::memcpy(target->data.f, src_data, count * sizeof(float));
        }
    }

    // --------------------------------------------------------------------------
    // Helper: Dequantize (INT8/UINT8 -> Float32)
    // --------------------------------------------------------------------------
    void dequantize_output(int tensor_idx, core::Tensor& dst) {
        const TfLiteTensor* src = interpreter->tensor(tensor_idx);
        float* dst_data = static_cast<float*>(dst.data());
        size_t count = dst.size();

        float scale = src->params.scale;
        int32_t zero_point = src->params.zero_point;

        if (src->type == kTfLiteUInt8) {
            const uint8_t* src_data = src->data.uint8;
            for (size_t i = 0; i < count; ++i) {
                dst_data[i] = (static_cast<float>(src_data[i]) - zero_point) * scale;
            }
        } else if (src->type == kTfLiteInt8) {
            const int8_t* src_data = src->data.int8;
            for (size_t i = 0; i < count; ++i) {
                dst_data[i] = (static_cast<float>(src_data[i]) - zero_point) * scale;
            }
        } else if (src->type == kTfLiteFloat32) {
            std::memcpy(dst_data, src->data.f, count * sizeof(float));
        }
    }
};

// =================================================================================
// 2. Public API Implementation
// =================================================================================

EdgeTpuBackend::EdgeTpuBackend(const EdgeTpuConfig& config)
    : m_config(config), m_impl(std::make_unique<Impl>(config)) {
}

EdgeTpuBackend::~EdgeTpuBackend() = default;

bool EdgeTpuBackend::load_model(const std::string& model_path) {
    try {
        // 1. Load Model File
        m_impl->model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
        if (!m_impl->model) {
            XINFER_LOG_ERROR("Failed to load TFLite model: " + model_path);
            return false;
        }

        // 2. Initialize TPU Delegate
        edgetpu::EdgeTpuManager::GetSingleton()->SetVerbosity(0); // Quiet

        // Select specific device if requested
        std::vector<edgetpu::EdgeTpuManager::DeviceEnumerationRecord> available_tpus =
            edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpus();

        if (available_tpus.empty()) {
            XINFER_LOG_ERROR("No Google Edge TPU devices found attached.");
            return false;
        }

        // Default to first device if index out of bounds
        size_t dev_idx = (m_config.device_index < available_tpus.size()) ? m_config.device_index : 0;

        // Open Device
        m_impl->tpu_context = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(
            available_tpus[dev_idx].type,
            available_tpus[dev_idx].path
        );

        if (!m_impl->tpu_context) {
            XINFER_LOG_ERROR("Failed to open Edge TPU Device.");
            return false;
        }

        // Create Delegate
        m_impl->tpu_delegate = m_impl->tpu_context->CreateDelegate();

        // 3. Build Interpreter
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder builder(*m_impl->model, resolver);

        builder(&m_impl->interpreter);
        if (!m_impl->interpreter) {
            XINFER_LOG_ERROR("Failed to construct TFLite Interpreter.");
            return false;
        }

        // 4. Apply Delegate
        if (m_impl->interpreter->ModifyGraphWithDelegate(m_impl->tpu_delegate) != kTfLiteOk) {
            XINFER_LOG_ERROR("Failed to apply Edge TPU Delegate to model.");
            return false;
        }

        // 5. Allocate Tensors
        if (m_impl->interpreter->AllocateTensors() != kTfLiteOk) {
            XINFER_LOG_ERROR("Failed to allocate TFLite tensors.");
            return false;
        }

        XINFER_LOG_INFO("Loaded Edge TPU Model: " + model_path);
        return true;

    } catch (const std::exception& e) {
        XINFER_LOG_ERROR("Edge TPU Backend Load Exception: " + std::string(e.what()));
        return false;
    }
}

void EdgeTpuBackend::predict(const std::vector<core::Tensor>& inputs,
                             std::vector<core::Tensor>& outputs) {

    // 1. Prepare Inputs (Float -> INT8 Quantization)
    const std::vector<int>& input_indices = m_impl->interpreter->inputs();

    if (inputs.size() != input_indices.size()) {
        XINFER_LOG_ERROR("Input count mismatch.");
        return;
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        m_impl->quantize_input(input_indices[i], inputs[i]);
    }

    // 2. Run Inference
    if (m_impl->interpreter->Invoke() != kTfLiteOk) {
        XINFER_LOG_ERROR("TFLite Invoke failed.");
        return;
    }

    // 3. Prepare Outputs (INT8 -> Float Dequantization)
    const std::vector<int>& output_indices = m_impl->interpreter->outputs();

    if (outputs.size() != output_indices.size()) {
        outputs.resize(output_indices.size());
    }

    for (size_t i = 0; i < output_indices.size(); ++i) {
        // Get output shape from TFLite tensor
        TfLiteTensor* out_tensor = m_impl->interpreter->tensor(output_indices[i]);

        // Resize xInfer tensor if needed
        if (outputs[i].empty()) {
            std::vector<int64_t> shape;
            for (int k = 0; k < out_tensor->dims->size; ++k) {
                shape.push_back(out_tensor->dims->data[k]);
            }
            outputs[i].resize(shape, core::DataType::kFLOAT);
        }

        m_impl->dequantize_output(output_indices[i], outputs[i]);
    }
}

std::string EdgeTpuBackend::device_name() const {
    return "Google Edge TPU (TFLite)";
}

float EdgeTpuBackend::get_input_scale(int index) const {
    const std::vector<int>& input_indices = m_impl->interpreter->inputs();
    if (index >= 0 && index < input_indices.size()) {
        TfLiteTensor* t = m_impl->interpreter->tensor(input_indices[index]);
        return t->params.scale;
    }
    return 1.0f;
}

int32_t EdgeTpuBackend::get_input_zero_point(int index) const {
    const std::vector<int>& input_indices = m_impl->interpreter->inputs();
    if (index >= 0 && index < input_indices.size()) {
        TfLiteTensor* t = m_impl->interpreter->tensor(input_indices[index]);
        return t->params.zero_point;
    }
    return 0;
}

// =================================================================================
// 3. Auto-Registration
// =================================================================================

namespace {
    volatile bool registered = xinfer::backends::BackendFactory::register_backend(
        xinfer::Target::GOOGLE_TPU,
        [](const xinfer::Config& config) -> std::unique_ptr<xinfer::IBackend> {
            EdgeTpuConfig tpu_cfg;
            tpu_cfg.model_path = config.model_path;

            // Parse vendor flags
            for(const auto& param : config.vendor_params) {
                if(param.find("DEVICE_INDEX=") != std::string::npos) {
                     tpu_cfg.device_index = std::stoi(param.substr(13));
                }
            }

            auto backend = std::make_unique<EdgeTpuBackend>(tpu_cfg);
            if(backend->load_model(tpu_cfg.model_path)) {
                return backend;
            }
            return nullptr;
        }
    );
}

} // namespace xinfer::backends::google_tpu