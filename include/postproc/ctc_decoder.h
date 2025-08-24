#pragma once

#include <include/core/tensor.h>
#include <vector>
#include <string>
#include <tuple>

namespace xinfer::postproc::ctc {

    std::pair<std::string, float> decode(
        const core::Tensor& logits,
        const std::vector<std::string>& character_map
    );

} // namespace xinfer::postproc::ctc

