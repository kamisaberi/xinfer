#pragma once

#include <string>
#include <vector>
#include <memory>

namespace xinfer::core { class Tensor; }

namespace xinfer::zoo::special {

    enum class TradingAction {
        HOLD,
        BUY,
        SELL
    };

    struct TradingSignal {
        TradingAction action;
        float confidence;
    };

    struct HFTConfig {
        std::string engine_path;
        // Add any specific config parameters for the HFT model
    };

    class HFTModel {
    public:
        explicit HFTModel(const HFTConfig& config);
        ~HFTModel();

        HFTModel(const HFTModel&) = delete;
        HFTModel& operator=(const HFTModel&) = delete;
        HFTModel(HFTModel&&) noexcept;
        HFTModel& operator=(HFTModel&&) noexcept;

        TradingSignal predict(const core::Tensor& market_data_tensor);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::special

