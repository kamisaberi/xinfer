#pragma once

#include <string>
#include <map>
#include <memory>
#include <any>
#include <vector>
#include <functional>

namespace xinfer::flow {

    /**
     * @brief Data packet passed between nodes.
     * Contains a dictionary of data (images, tensors, strings) and metadata.
     */
    struct Packet {
        std::map<std::string, std::any> data; // Key-Value store
        long long timestamp;
        bool is_eos = false; // End of Stream signal
    };

    /**
     * @brief Abstract Base Class for a Pipeline Node.
     */
    class INode {
    public:
        virtual ~INode() = default;

        /**
         * @brief Initialize the node with parameters from JSON.
         * @param params Key-Value string pairs from the config file.
         */
        virtual void init(const std::map<std::string, std::string>& params) = 0;

        /**
         * @brief Process a single packet.
         *
         * @param input Data from the previous node.
         * @return Processed data to send to the next node.
         */
        virtual Packet process(const Packet& input) = 0;

        /**
         * @brief Reset internal state (optional).
         */
        virtual void reset() {}
    };

    // Factory type for creating nodes dynamically
    using NodeCreator = std::function<std::unique_ptr<INode>()>;

} // namespace xinfer::flow