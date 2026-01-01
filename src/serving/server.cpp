#include <xinfer/serving/server.h>
#include <xinfer/serving/model_repository.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// Helper libs
#include "httplib.h"
#include "json.hpp"

#include <chrono>
#include <iostream>

using json = nlohmann::json;

namespace xinfer::serving {

struct ModelServer::Impl {
    ServerConfig config;
    httplib::Server svr;
    std::unique_ptr<ModelRepository> repo;

    Impl(const ServerConfig& cfg) : config(cfg) {
        repo = std::make_unique<ModelRepository>(config.model_repo_path);

        // Configure Thread Pool
        svr.new_task_queue = [this] { return new httplib::ThreadPool(config.num_threads); };

        setup_routes();
    }

    void setup_routes() {
        // --- GET /health ---
        svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
            json j = {{"status", "ok"}, {"engine", "xInfer"}};
            res.set_content(j.dump(), "application/json");
        });

        // --- POST /v1/models/:name/predict ---
        svr.Post(R"(/v1/models/(.*):predict)", [&](const httplib::Request& req, httplib::Response& res) {
            auto start_t = std::chrono::high_resolution_clock::now();

            std::string model_name = req.matches[1];
            auto* backend = repo->get_model(model_name);

            if (!backend) {
                res.status = 404;
                res.set_content(R"({"error": "Model not found"})", "application/json");
                return;
            }

            try {
                auto j = json::parse(req.body);

                // Parse Input
                if (!j.contains("input") || !j.contains("shape")) throw std::runtime_error("Missing 'input' or 'shape'");

                std::vector<float> data = j["input"].get<std::vector<float>>();
                std::vector<int64_t> shape = j["shape"].get<std::vector<int64_t>>();

                // Create Tensor
                core::Tensor input(shape, core::DataType::kFLOAT);
                std::memcpy(input.data(), data.data(), data.size() * sizeof(float));

                core::Tensor output;

                // Run Inference
                backend->predict({input}, {output});

                // Prepare Response
                auto end_t = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration_cast<std::chrono::microseconds>(end_t - start_t).count() / 1000.0;

                const float* out_ptr = static_cast<const float*>(output.data());
                std::vector<float> out_vec(out_ptr, out_ptr + output.size());

                json resp;
                resp["output"] = out_vec;
                resp["shape"] = output.shape();
                resp["time_ms"] = ms;

                res.set_content(resp.dump(), "application/json");

            } catch (const std::exception& e) {
                res.status = 400;
                json err = {{"error", e.what()}};
                res.set_content(err.dump(), "application/json");
            }
        });
    }
};

ModelServer::ModelServer(const ServerConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

ModelServer::~ModelServer() { stop(); }

void ModelServer::start() {
    XINFER_LOG_INFO("Serving: Listening on port " + std::to_string(pimpl_->config.port));
    pimpl_->svr.listen(pimpl_->config.host.c_str(), pimpl_->config.port);
}

void ModelServer::stop() {
    pimpl_->svr.stop();
}

} // namespace xinfer::serving