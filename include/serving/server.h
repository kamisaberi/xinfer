#pragma once

#include <memory>
#include "types.h"

namespace xinfer::serving {

    /**
     * @brief A lightweight REST API server for xInfer models.
     *
     * Endpoints:
     * - GET  /health                  : Check server status
     * - POST /v1/models/{name}:predict : Run inference
     */
    class ModelServer {
    public:
        explicit ModelServer(const ServerConfig& config);
        ~ModelServer();

        // Move semantics
        ModelServer(ModelServer&&) noexcept;
        ModelServer& operator=(ModelServer&&) noexcept;
        ModelServer(const ModelServer&) = delete;
        ModelServer& operator=(const ModelServer&) = delete;

        /**
         * @brief Starts the server loop.
         * This function blocks until stop() is called or a signal is received.
         */
        void start();

        /**
         * @brief Stops the server gracefully.
         */
        void stop();

        /**
         * @brief Check if the server is currently running.
         */
        bool is_running() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::serving