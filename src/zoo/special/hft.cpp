#include <include/zoo/special/hft.h>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>

#include <include/core/engine.h>

namespace xinfer::zoo::special {

    struct HFTModel::Impl {
        HFTModelConfig config_;
        std::unique_ptr<core::InferenceEngine> engine_;
    };

    HFTModel::HFTModel(const HFTModelConfig& config)
        : pimpl_(new Impl{config})
    {
        if (!std::ifstream(pimpl_->config_.engine_path).good()) {
            throw std::runtime_error("HFT engine file not found: " + pimpl_->config_.engine_path);
        }

        pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);
    }

    HFTModel::~HFTModel() = default;
    HFTModel::HFTModel(HFTModel&&) noexcept = default;
    HFTModel& HFTModel::operator=(HFTModel&&) noexcept = default;

    TradingSignal HFTModel::predict(const core::Tensor& market_data_tensor) {
        if (!pimpl_) throw std::runtime_error("HFTModel is in a moved-from state.");

        auto output_tensors = pimpl_->engine_->infer({market_data_tensor});
        const core::Tensor& logits_tensor = output_tensors[0];

        std::vector<float> logits(logits_tensor.num_elements());
        logits_tensor.copy_to_host(logits.data());

        auto max_it = std::max_element(logits.begin(), logits.end());
        int max_idx = std::distance(logits.begin(), max_it);
        float max_val = *max_it;

        float sum_exp = 0.0f;
        for (float logit : logits) {
            sum_exp += expf(logit - max_val);
        }
        float confidence = 1.0f / sum_exp;

        TradingSignal signal;
        signal.action = static_cast<TradingAction>(max_idx);
        signal.confidence = confidence;

        return signal;
    }

} // namespace xinfer::zoo::special