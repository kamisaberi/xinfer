#include <xinfer/serving/server.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>
#include <xinfer/core/utils.h>
#include <xinfer/backends/backend_factory.h>

// Third-party single-header libraries
// Ensure these exist in your 'third_party' folder
#include "httplib.h"
#include "json.hpp"

#include <iostream>
#include <filesystem>
#include <chrono>
#include <mutex>
#include <map>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace xinfer::serving {

// =================================================================================
// Internal Implementation (PImpl)
// =================================================================================

struct ModelServer::Impl {
    ServerConfig config_;
    httplib::Server svr_;

    // Model Cache: Map<ModelName, BackendInstance>
    // We use a mutex to make model loading thread-safe
    std::map<std::string, std::unique_ptr<backends::IBackend>> loaded_models_;
    std::mutex model_mutex_;

    Impl(const ServerConfig& config) : config_(config) {
        setup_routes();
        // Set thread pool size
        svr_.new_task_queue = [this] { return new httplib::ThreadPool(config_.num_threads); };
    }

    // --- Helper: Load Model from Disk ---
    backends::IBackend* get_or_load_model(const std::string& model_name) {
        std::lock_guard<std::mutex> lock(model_mutex_);

        // 1. Check Cache
        if (loaded_models_.find(model_name) != loaded_models_.end()) {
            return loaded_models_[model_name].get();
        }

        // 2. Resolve File Path
        // We look for supported extensions in the repository path
        std::vector<std::string> extensions = {".engine", ".rknn", ".xml", ".xmodel", ".onnx"};
        std::string found_path = "";

        for (const auto& ext : extensions) {
            std::string p = config_.model_repository_path + "/" + model_name + ext;
            if (fs::exists(p)) {
                found_path = p;
                break;
            }
        }

        if (found_path.empty()) {
            XINFER_LOG_ERROR("Model file not found for: " + model_name);
            return nullptr;
        }

        // 3. Determine Target based on extension (Simple Heuristic)
        // In production, this might be configured via a config.json per model
        Target target = Target::INTEL_OV; // Default fallback
        if (found_path.find(".engine") != std::string::npos) target = Target::NVIDIA_TRT;
        else if (found_path.find(".rknn") != std::string::npos) target = Target::ROCKCHIP_RKNN;
        else if (found_path.find(".xmodel") != std::string::npos) target = Target::AMD_VITIS;

        // 4. Load
        XINFER_LOG_INFO("Loading model '" + model_name + "' from " + found_path);
        auto backend = backends::BackendFactory::create(target);

        if (!backend->load_model(found_path)) {
            XINFER_LOG_ERROR("Failed to load backend for " + found_path);
            return nullptr;
        }

        loaded_models_[model_name] = std::move(backend);
        return loaded_models_[model_name].get();
    }

    // --- Route Setup ---
    void setup_routes() {
        // 1. Health Check
        svr_.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            json response = {
                {"status", "live"},
                {"version", "xInfer 1.0.0"},
                {"backend", "C++ Optimized"}
            };
            res.set_content(response.dump(), "application/json");
        });

        // 2. Inference Endpoint
        // POST /v1/models/{name}:predict
        // Body: { "input": [...], "shape": [...] }
        svr_.Post(R"(/v1/models/(.*):predict)", [&](const httplib::Request& req, httplib::Response& res) {
            auto start_time = std::chrono::high_resolution_clock::now();

            std::string model_name = req.matches[1];

            // A. Get Model
            auto* backend = get_or_load_model(model_name);
            if (!backend) {
                json err = {{"error", "Model not found or failed to load"}, {"model", model_name}};
                res.status = 404;
                res.set_content(err.dump(), "application/json");
                return;
            }

            try {
                // B. Parse JSON
                auto json_body = json::parse(req.body);

                // Validate inputs
                if (!json_body.contains("input") || !json_body.contains("shape")) {
                    throw std::runtime_error("JSON must contain 'input' array and 'shape' array.");
                }

                std::vector<float> input_data = json_body["input"].get<std::vector<float>>();
                std::vector<int64_t> shape = json_body["shape"].get<std::vector<int64_t>>();

                // C. Prepare Tensors
                core::Tensor input_tensor(shape, core::DataType::kFLOAT);
                if (input_tensor.size() != input_data.size()) {
                     throw std::runtime_error("Input data size does not match shape dimensions.");
                }

                // Copy data (JSON -> Tensor)
                std::memcpy(input_tensor.data(), input_data.data(), input_data.size() * sizeof(float));

                core::Tensor output_tensor;

                // D. Run Inference
                backend->predict({input_tensor}, {output_tensor});

                // E. Serialize Response
                auto end_time = std::chrono::high_resolution_clock::now();
                double duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;

                const float* out_ptr = static_cast<const float*>(output_tensor.data());
                std::vector<float> out_vec(out_ptr, out_ptr + output_tensor.size());

                json response = {
                    {"model_name", model_name},
                    {"output", out_vec},
                    {"shape", output_tensor.shape()},
                    {"inference_time_ms", duration_ms}
                };

                res.set_content(response.dump(), "application/json");

                if (config_.enable_logging) {
                    XINFER_LOG_INFO("Inference success: " + model_name + " (" + std::to_string(duration_ms) + "ms)");
                }

            } catch (const std::exception& e) {
                XINFER_LOG_ERROR("Inference Request Failed: " + std::string(e.what()));
                json err = {{"error", e.what()}};
                res.status = 400;
                res.set_content(err.dump(), "application/json");
            }
        });
    }
};

// =================================================================================
// Public API
// =================================================================================

ModelServer::ModelServer(const ServerConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

ModelServer::~ModelServer() = default;
ModelServer::ModelServer(ModelServer&&) noexcept = default;
ModelServer& ModelServer::operator=(ModelServer&&) noexcept = default;

void ModelServer::start() {
    XINFER_LOG_INFO("xInfer Server starting at http://" + pimpl_->config_.host + ":" + std::to_string(pimpl_->config_.port));
    XINFER_LOG_INFO("Model Repository: " + pimpl_->config_.model_repository_path);

    pimpl_->svr_.listen(pimpl_->config_.host.c_str(), pimpl_->config_.port);
}

void ModelServer::stop() {
    XINFER_LOG_INFO("Stopping xInfer Server...");
    pimpl_->svr_.stop();
}

bool ModelServer::is_running() const {
    return pimpl_->svr_.is_running();
}

} // namespace xinfer::serving