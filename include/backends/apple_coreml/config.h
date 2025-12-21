#pragma once

#include <string>
#include "types.h"

namespace xinfer::backends::coreml {

    /**
     * @brief Configuration for Apple Core ML Backend
     */
    struct CoreMLConfig {
        // Path to the compiled model bundle (.mlmodelc directory)
        // Note: Core ML requires the compiled folder, not the source .mlmodel
        std::string model_path;

        // Hardware execution strategy
        ComputeUnit compute_unit = ComputeUnit::ALL;

        // Allow low precision accumulation on GPU/ANE?
        // Setting this to true can significantly speed up inference but may reduce accuracy.
        bool allow_low_precision = true;

        // For ANE: wait for command completion?
        // False = Async execution (better pipeline throughput)
        bool wait_for_completion = true;
    };

} // namespace xinfer::backends::coreml