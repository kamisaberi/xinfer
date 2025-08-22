// src/core/tensor.cpp
#include <include/core/tensor.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <numeric>
#include <map>

// Helper to check CUDA calls
#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(error))); \
    } \
}

namespace xinfer::core {

// --- PIMPL Idiom: Implementation details are hidden ---
struct Tensor::Impl {
    void* gpu_data_ = nullptr;
    std::vector<int64_t> shape_;
    DataType dtype_;
    size_t num_elements_ = 0;
    size_t size_bytes_ = 0;

    static size_t sizeof_dtype(DataType dtype) {
        static const std::map<DataType, size_t> type_map = {
            {DataType::kFLOAT, sizeof(float)}, {DataType::kHALF, 2},
            {DataType::kINT8, sizeof(int8_t)}, {DataType::kINT32, sizeof(int32_t)}
        };
        return type_map.at(dtype);
    }

    void allocate(const std::vector<int64_t>& shape, DataType dtype) {
        shape_ = shape;
        dtype_ = dtype;
        num_elements_ = std::accumulate(shape_.begin(), shape_.end(), 1LL, std::multiplies<int64_t>());
        size_bytes_ = num_elements_ * sizeof_dtype(dtype_);
        if (size_bytes_ > 0) {
            CHECK_CUDA(cudaMalloc(&gpu_data_, size_bytes_));
        }
    }

    void release() {
        if (gpu_data_) {
            cudaFree(gpu_data_);
            gpu_data_ = nullptr;
        }
        shape_.clear();
        num_elements_ = 0;
        size_bytes_ = 0;
    }
};

// --- Public API Implementation ---
Tensor::Tensor() : pimpl_(new Impl()) {}

Tensor::Tensor(const std::vector<int64_t>& shape, DataType dtype) : pimpl_(new Impl()) {
    pimpl_->allocate(shape, dtype);
}

Tensor::~Tensor() {
    pimpl_->release();
}

// Move constructor
Tensor::Tensor(Tensor&& other) noexcept : pimpl_(std::move(other.pimpl_)) {}

// Move assignment
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        pimpl_ = std::move(other.pimpl_);
    }
    return *this;
}

void* Tensor::data() const { return pimpl_->gpu_data_; }
const std::vector<int64_t>& Tensor::shape() const { return pimpl_->shape_; }
DataType Tensor::dtype() const { return pimpl_->dtype_; }
size_t Tensor::num_elements() const { return pimpl_->num_elements_; }
size_t Tensor::size_bytes() const { return pimpl_->size_bytes_; }

void Tensor::copy_from_host(const void* cpu_data) {
    if (!pimpl_->gpu_data_) throw std::runtime_error("Tensor is not allocated.");
    CHECK_CUDA(cudaMemcpy(pimpl_->gpu_data_, cpu_data, pimpl_->size_bytes_, cudaMemcpyHostToDevice));
}

void Tensor::copy_to_host(void* cpu_data) const {
    if (!pimpl_->gpu_data_) throw std::runtime_error("Tensor is not allocated.");
    CHECK_CUDA(cudaMemcpy(cpu_data, pimpl_->gpu_data_, pimpl_->size_bytes_, cudaMemcpyDeviceToHost));
}

} // namespace xinfer::core