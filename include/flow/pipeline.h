#pragma once

#include "node_interface.h"
#include <string>
#include <vector>
#include <memory>

namespace xinfer::flow {

    class Pipeline {
    public:
        Pipeline();
        ~Pipeline();

        /**
         * @brief Load and build a pipeline from a JSON configuration file.
         *
         * @param json_path Path to the pipeline definition file.
         * @return true if parsing and node creation were successful.
         */
        bool load(const std::string& json_path);

        /**
         * @brief Run the pipeline loop.
         * Blocks until the pipeline finishes (EOS) or stop() is called.
         */
        void run();

        /**
         * @brief Signal the pipeline to stop execution.
         */
        void stop();

        /**
         * @brief Register a custom node type dynamically.
         * Used by internal modules to register their wrappers (e.g., DetectorNode).
         */
        static void register_node(const std::string& type_name, NodeCreator creator);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::flow