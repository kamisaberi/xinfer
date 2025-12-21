#pragma once

namespace xinfer::backends::ambarella {

    // Ambarella CVFlow specific error codes
    enum class CavalryError {
        SUCCESS = 0,
        DAG_LOAD_FAILED = -1,
        DMA_FAILURE = -2,
        VP_HANG = -3
    };

    // CVFlow supports specific fixed-point formats
    enum class cv_precision_t {
        FIX_8 = 0,  // Standard INT8
        FIX_16 = 1, // INT16
        FIX_4 = 2   // 4-bit packed (Ultra fast)
    };

}