#pragma once

#include <string>
#include <map>
#include <memory>
#include <any>
#include <vector>

namespace xinfer::flow {

    // Data passed between nodes
    struct Packet {
        std::map<std::string, std::any> data;
        long long timestamp;
        bool is_eos = false; // End of Stream
    };

    class INode {
    public:
        virtual ~INode() = default;

        /**
         * @brief Initialize the node with parameters from JSON.
         */
        virtual void init(const std::map<std::string, std::string>& params) = 0;

        /**
         * @brief Process a single packet.
         *
         * @param input Data from the previous node.
         * @return Data to send to the next node (or empty Packet if filtering).
         */
        virtual Packet process(const Packet& input) = 0;

        /**
         * @brief Reset state (e.g. for trackers or temporal models).
         */
        virtual void reset() {}
    };

    // Factory type for creating nodes
    using NodeCreator = std::function<std::unique_ptr<INode>()>;

} // namespace xinfer::flow