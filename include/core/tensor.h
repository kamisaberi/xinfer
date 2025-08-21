#pragma once

#include <vector>
#include <cstdint>
#include <memory>

namespace xinfer::core {

    // Enum for data types, mapping to CUDA/TensorRT types
    enum class DataType { kFLOAT, kHALF, kINT8, kINT32 };

    /**
     * @class Tensor
     * @brief A lightweight, RAII-compliant wrapper for a GPU memory buffer.
     *
     * This class manages the lifetime of a GPU buffer. It's designed to be
     * easy to create, move, and pass around, abstracting away raw cudaMalloc/cudaFree calls.
     */
    class Tensor {
    public:
        // Default constructor for an empty tensor
        Tensor();

        // Constructor to allocate a new GPU buffer
        Tensor(const std::vector<int64_t>& shape, DataType dtype);

        // Destructor (automatically calls cudaFree)
        ~Tensor();

        // --- Rule of Five: Making the class movable but not copyable ---
        // (This is crucial for safe resource management)
        Tensor(const Tensor&) = delete; // No copying
        Tensor& operator=(const Tensor&) = delete; // No copying
        Tensor(Tensor&& other) noexcept;      // Move constructor
        Tensor& operator=(Tensor&& other) noexcept; // Move assignment

        // --- Public API ---
        void* data() const; // Get the raw GPU pointer
        const std::vector<int64_t>& shape() const;
        DataType dtype() const;
        size_t num_elements() const;
        size_t size_bytes() const;

        // Helper to copy data from the CPU (host) to this GPU tensor
        void copy_from_host(const void* cpu_data);

        // Helper to copy data from this GPU tensor to the CPU (host)
        void copy_to_host(void* cpu_data) const;

    private:
        struct Impl; // PIMPL idiom to hide CUDA headers from this public header
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::core
