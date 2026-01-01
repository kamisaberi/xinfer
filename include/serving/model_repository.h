#pragma once

#include <string>
#include <memory>
#include <map>
#include "types.h"
#include <xinfer/backends/backend_factory.h>

namespace xinfer::serving {

    class ModelRepository {
    public:
        explicit ModelRepository(const std::string& repo_path);
        ~ModelRepository();

        /**
         * @brief Get a model instance. Loads from disk if not already cached.
         * Thread-safe.
         */
        backends::IBackend* get_model(const std::string& model_name);

        /**
         * @brief Unload a model to free memory.
         */
        void unload_model(const std::string& model_name);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}