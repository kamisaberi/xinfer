#pragma once

#include <string>
#include <memory>
#include <map>
#include <vector>
#include <mutex>

// xInfer Core
#include <xinfer/backends/backend_factory.h>

namespace xinfer::serving {

    /**
     * @brief Manages the lifecycle of inference backends.
     *
     * Features:
     * - Lazy Loading: Models are loaded only when first requested.
     * - Auto-Discovery: Infers hardware target from file extensions (.engine, .rknn).
     * - Thread Safety: Safe to request models from multiple HTTP threads.
     */
    class ModelRepository {
    public:
        /**
         * @brief Initialize with a path to a directory containing models.
         * @param repo_path e.g., "/var/models/" or "./models"
         */
        explicit ModelRepository(const std::string& repo_path);
        ~ModelRepository();

        // Move semantics
        ModelRepository(ModelRepository&&) noexcept;
        ModelRepository& operator=(ModelRepository&&) noexcept;
        ModelRepository(const ModelRepository&) = delete;
        ModelRepository& operator=(const ModelRepository&) = delete;

        /**
         * @brief Retrieve a loaded backend for inference.
         *
         * If the model exists on disk but isn't loaded, this triggers the load.
         *
         * @param model_name The filename without extension (e.g., "yolov8").
         * @return Pointer to the backend, or nullptr if not found/failed.
         */
        backends::IBackend* get_model(const std::string& model_name);

        /**
         * @brief Check if a model exists in the repository configuration.
         */
        bool exists(const std::string& model_name) const;

        /**
         * @brief Force unload a model to free memory.
         */
        void unload_model(const std::string& model_name);

        /**
         * @brief List all available models (loaded and unloaded).
         */
        std::vector<std::string> list_models() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::serving