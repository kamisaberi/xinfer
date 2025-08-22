#pragma once
#include <string>
#include <vector>

namespace xinfer::builders {

    /**
     * @class INT8Calibrator
     * @brief An abstract interface for providing calibration data for INT8 quantization.
     *
     * A user must implement this interface to provide batches of representative
     * input data. The TensorRT builder will run inference on this data to measure
     * the activation distributions and determine the optimal scaling factors for INT8.
     */
    class INT8Calibrator {
    public:
        virtual ~INT8Calibrator() = default;

        // The batch size of the calibration data
        virtual int get_batch_size() const = 0;

        // Get the next batch of calibration data. The implementation should load
        // data into the provided GPU pointer.
        // Returns false when there is no more data.
        virtual bool get_next_batch(void* gpu_binding) = 0;
    };

    // You would also provide a default, easy-to-use implementation, e.g.:
    // class DataLoaderCalibrator : public INT8Calibrator {
    // public:
    //     DataLoaderCalibrator(xt::dataloaders::ExtendedDataLoader& loader, ...);
    //     // ... implementation ...
    // };

} // namespace xinfer::builders
