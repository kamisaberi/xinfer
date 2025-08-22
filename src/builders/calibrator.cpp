#include <include/builders/calibrator.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

#define CHECK_CUDA(call) { /* ... */ }

namespace xinfer::builders {

struct DataLoaderCalibrator::Impl {
    xt::dataloaders::ExtendedDataLoader& loader_;
    decltype(loader_.begin()) current_batch_iterator_;
    size_t batch_size_;
    size_t input_size_bytes_;
    void* d_buffer_ = nullptr; // A single GPU buffer for the calibration data

    Impl(xt::dataloaders::ExtendedDataLoader& loader)
        : loader_(loader), current_batch_iterator_(loader_.begin()) {

        batch_size_ = loader_.options().batch_size;

        // Get the shape of the first item to determine buffer size
        auto first_batch = *current_batch_iterator_;
        auto first_tensor = first_batch.data;

        input_size_bytes_ = first_tensor.nbytes();

        // Allocate a single GPU buffer that we will reuse for every batch
        CHECK_CUDA(cudaMalloc(&d_buffer_, input_size_bytes_));
    }

    ~Impl() {
        if (d_buffer_) {
            cudaFree(d_buffer_);
        }
    }
};

DataLoaderCalibrator::DataLoaderCalibrator(xt::dataloaders::ExtendedDataLoader& loader, const std::string& cache_path)
    : pimpl_(new Impl(loader)) {}

DataLoaderCalibrator::~DataLoaderCalibrator() = default;

int DataLoaderCalibrator::get_batch_size() const {
    return pimpl_->batch_size_;
}

bool DataLoaderCalibrator::get_next_batch(void* gpu_binding) {
    if (pimpl_->current_batch_iterator_ == pimpl_->loader_.end()) {
        // We've reached the end of the dataset
        std::cout << "INT8 Calibration: End of dataset." << std::endl;
        // Reset iterator for potential reuse
        pimpl_->current_batch_iterator_ = pimpl_->loader_.begin();
        return false;
    }

    // Get the current batch data (which is already on the CPU)
    auto current_batch = *pimpl_->current_batch_iterator_;
    auto cpu_tensor = current_batch.data;

    // Copy the batch data from the host tensor to our internal GPU buffer
    CHECK_CUDA(cudaMemcpy(pimpl_->d_buffer_, cpu_tensor.data_ptr(), pimpl_->input_size_bytes_, cudaMemcpyHostToDevice));

    // TensorRT requires us to copy from our buffer to its provided binding pointer
    CHECK_CUDA(cudaMemcpy(gpu_binding, pimpl_->d_buffer_, pimpl_->input_size_bytes_, cudaMemcpyDeviceToDevice));

    // Move to the next batch
    ++pimpl_->current_batch_iterator_;
    return true;
}

} // namespace xinfer::builders