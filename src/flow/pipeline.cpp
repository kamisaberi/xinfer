#include <xinfer/flow/pipeline.h>
#include <xinfer/core/logging.h>

// Third-party JSON
#include "json.hpp"
#include <fstream>
#include <thread>
#include <atomic>
#include <chrono>

using json = nlohmann::json;

namespace xinfer::flow {

// =================================================================================
// Global Node Registry
// =================================================================================
static std::map<std::string, NodeCreator>& get_registry() {
    static std::map<std::string, NodeCreator> registry;
    return registry;
}

void Pipeline::register_node(const std::string& type_name, NodeCreator creator) {
    get_registry()[type_name] = creator;
}

// =================================================================================
// PImpl Implementation
// =================================================================================

struct Pipeline::Impl {
    struct NodeEntry {
        std::string id;
        std::string type;
        std::unique_ptr<INode> instance;
        std::vector<int> output_indices; // Adjacency list indices
    };

    std::vector<NodeEntry> execution_order_; // Linear execution list
    std::atomic<bool> running_{false};
    int target_fps_ = 0;

    bool build_graph(const json& j) {
        // 1. Parse Nodes
        std::map<std::string, int> id_map; // ID string -> Vector Index

        for (const auto& node_cfg : j["nodes"]) {
            std::string type = node_cfg["type"];
            std::string id = node_cfg["id"];

            // Look up factory
            auto& reg = get_registry();
            if (reg.find(type) == reg.end()) {
                XINFER_LOG_ERROR("Unknown Node Type: " + type);
                return false;
            }

            // Create Instance
            auto node_ptr = reg[type]();

            // Convert params JSON object to std::map<string, string>
            std::map<std::string, std::string> params;
            if (node_cfg.contains("params")) {
                for (auto& element : node_cfg["params"].items()) {
                    // Handle different types by converting to string
                    if (element.value().is_string()) params[element.key()] = element.value();
                    else params[element.key()] = element.value().dump();
                }
            }

            node_ptr->init(params);

            // Store
            NodeEntry entry;
            entry.id = id;
            entry.type = type;
            entry.instance = std::move(node_ptr);

            id_map[id] = execution_order_.size();
            execution_order_.push_back(std::move(entry));
        }

        // 2. Parse Edges (For simple linear pipelines, we just rely on order in JSON arrays
        // but a real graph needs topological sort.
        // Here we assume the JSON 'nodes' list is already sorted topologically).

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
        if (j.contains("fps_limit")) pimpl_->target_fps_ = j["fps_limit"];
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

    // Main Loop
    while (pimpl_->running_) {
        auto start_time = std::chrono::high_resolution_clock::now();

        Packet current_packet;
        current_packet.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            start_time.time_since_epoch()).count();

        // Execute Nodes Sequentially
        for (auto& node_entry : pimpl_->execution_order_) {
            try {
                // Pass packet to node
                // The node consumes inputs from packet and adds outputs to it
                // For a linear chain, we just overwrite/append.
                Packet result = node_entry.instance->process(current_packet);

                // If a node returns empty/EOS, we abort this frame
                if (result.is_eos) {
                    XINFER_LOG_INFO("Pipeline EOS reached.");
                    pimpl_->running_ = false;
                    break;
                }

                // Update packet data (merge)
                // For linear, usually result becomes next input
                current_packet = result;

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
}

} // namespace xinfer::flow