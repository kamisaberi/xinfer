#pragma once
#include <memory>
#include "types.h"

namespace xinfer::serving {

    class ModelServer {
    public:
        explicit ModelServer(const ServerConfig& config);
        ~ModelServer();

        // Blocking call
        void start();
        void stop();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}