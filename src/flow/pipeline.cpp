#include <xinfer/flow/pipeline.h>
#include <xinfer/core/logging.h>

// Third-party JSON (Ensure json.hpp is in third_party/)
#include "json.hpp"
#include <fstream>
#include <thread>
#include <atomic>
#include <chrono>
#include <iostream>

using json = nlohmann::json;

namespace xinfer::flow {

// =================================================================================
// Global Node Registry
// =================================================================================
// Singleton pattern to store factory functions for available nodes
static std::map<std::string, NodeCreator>& get_registry() {
    static std::map<std::string, NodeCreator> registry;
    return registry;
}

void Pipeline::register_node(const std::string& type_name, NodeCreator creator) {
    get_registry()[type_name] = creator;
    XINFER_LOG_INFO("Registered Flow Node: " + type_name);
}

// =================================================================================
// PImpl Implementation
// =================================================================================

struct Pipeline::Impl {
    struct NodeEntry {
        std::string id;
        std::string type;
        std::unique_ptr<INode> instance;
    };

    // For a simple pipeline, we store nodes in execution order.
    // (A more complex version would use a directed graph).
    std::vector<NodeEntry> execution_order_;

    std::atomic<bool> running_{false};
    int target_fps_ = 0;

    bool build_graph(const json& j) {
        execution_order_.clear();

        // 1. Parse Nodes
        if (!j.contains("nodes") || !j["nodes"].is_array()) {
            XINFER_LOG_ERROR("Pipeline config missing 'nodes' array.");
            return false;
        }

        for (const auto& node_cfg : j["nodes"]) {
            std::string type = node_cfg.value("type", "Unknown");
            std::string id = node_cfg.value("id", "node_" + std::to_string(execution_order_.size()));

            // Look up factory
            auto& reg = get_registry();
            if (reg.find(type) == reg.end()) {
                XINFER_LOG_ERROR("Unknown Node Type: " + type + " (ID: " + id + ")");
                return false;
            }

            // Create Instance
            auto node_ptr = reg[type]();

            // Extract Parameters
            std::map<std::string, std::string> params;
            if (node_cfg.contains("params")) {
                for (auto& element : node_cfg["params"].items()) {
                    // Convert all params to strings for the interface
                    if (element.value().is_string()) params[element.key()] = element.value();
                    else params[element.key()] = element.value().dump();
                }
            }

            // Initialize Node
            try {
                node_ptr->init(params);
            } catch (const std::exception& e) {
                XINFER_LOG_ERROR("Failed to init node " + id + ": " + e.what());
                return false;
            }

            // Store
            NodeEntry entry;
            entry.id = id;
            entry.type = type;
            entry.instance = std::move(node_ptr);

            execution_order_.push_back(std::move(entry));
        }

        // 2. Parse Edges (Optional for linear execution)
        // In this simple version, we assume the JSON 'nodes' array determines the order.
        // A full graph version would parse "edges" and topological sort.

        XINFER_LOG_INFO("Pipeline built with " + std::to_string(execution_order_.size()) + " nodes.");
        return true;
    }
};

// =================================================================================
// Public API
// =================================================================================

Pipeline::Pipeline() : pimpl_(std::make_unique<Impl>()) {}
Pipeline::~Pipeline() = default;

bool Pipeline::load(const std::string& json_path) {
    std::ifstream f(json_path);
    if (!f.is_open()) {
        XINFER_LOG_ERROR("Could not open pipeline config: " + json_path);
        return false;
    }

    try {
        json j = json::parse(f);

        if (j.contains("fps_limit")) {
            pimpl_->target_fps_ = j["fps_limit"];
        }

        return pimpl_->build_graph(j);
    } catch (const std::exception& e) {
        XINFER_LOG_ERROR("JSON Parsing error: " + std::string(e.what()));
        return false;
    }
}

void Pipeline::stop() {
    pimpl_->running_ = false;
}

void Pipeline::run() {
    pimpl_->running_ = true;

    XINFER_LOG_INFO("Pipeline started.");

    while (pimpl_->running_) {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Create fresh packet for this frame
        Packet current_packet;
        current_packet.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            start_time.time_since_epoch()).count();

        // Pipeline Execution Logic
        for (auto& node_entry : pimpl_->execution_order_) {
            try {
                // Pass packet to node
                Packet result = node_entry.instance->process(current_packet);

                // If a node returns EOS, abort this frame/stream
                if (result.is_eos) {
                    XINFER_LOG_INFO("Pipeline EOS reached at node: " + node_entry.id);
                    pimpl_->running_ = false;
                    break;
                }

                // For linear chain, output of N is input of N+1
                // We merge data to allow downstream nodes access to original inputs if needed
                for (const auto& kv : result.data) {
                    current_packet.data[kv.first] = kv.second;
                }

            } catch (const std::exception& e) {
                XINFER_LOG_ERROR("Error in node '" + node_entry.id + "': " + e.what());
                pimpl_->running_ = false;
                break;
            }
        }

        // FPS Limiter
        if (pimpl_->target_fps_ > 0) {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            int target_ms = 1000 / pimpl_->target_fps_;
            if (duration_ms < target_ms) {
                std::this_thread::sleep_for(std::chrono::milliseconds(target_ms - duration_ms));
            }
        }
    }

    XINFER_LOG_INFO("Pipeline stopped.");
}

} // namespace xinfer::flow