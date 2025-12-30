#pragma once

#include "node_interface.h"
#include <string>
#include <vector>
#include <memory>

namespace xinfer::flow {

    struct PipelineConfig {
        std::string name;
        int fps_limit = 0; // 0 = Uncapped
    };

    class Pipeline {
    public:
        Pipeline();
        ~Pipeline();

        /**
         * @brief Load a pipeline from a JSON file.
         *
         * @param json_path Path to pipeline.json
         * @return true if successful.
         */
        bool load(const std::string& json_path);

        /**
         * @brief Run the pipeline.
         * Blocks until the pipeline finishes (EOS) or stop() is called.
         */
        void run();

        /**
         * @brief Stop execution.
         */
        void stop();

        // Register custom nodes dynamically
        static void register_node(const std::string& type_name, NodeCreator creator);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::flow