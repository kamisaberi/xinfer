#pragma once

#include <string>
#include <vector>
#include <memory>
#include <include/core/tensor.h>

namespace xinfer::zoo::special {

    struct FluidSimulatorConfig {
        int resolution_x = 256;
        int resolution_y = 256;
        float viscosity = 0.1f;
        float diffusion = 0.1f;
        float timestep = 0.01f;
    };

    class FluidSimulator {
    public:
        explicit FluidSimulator(const FluidSimulatorConfig& config);
        ~FluidSimulator();

        FluidSimulator(const FluidSimulator&) = delete;
        FluidSimulator& operator=(const FluidSimulator&) = delete;
        FluidSimulator(FluidSimulator&&) noexcept;
        FluidSimulator& operator=(FluidSimulator&&) noexcept;

        void step(core::Tensor& velocity_field, core::Tensor& density_field);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::special

