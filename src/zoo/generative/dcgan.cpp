#include <include/zoo/generative/dcgan.h>
#include <stdexcept>
#include <vector>
#include <fstream>
#include <cuda_runtime_api.h>
#include <curand.h>
#include "NvInfer.h"
#include <iostream>

// Bring in the helper macros
#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(error))); \
    } \
}

#define CHECK_CURAND(call) { \
    const curandStatus_t status = call; \
    if (status != CURAND_STATUS_SUCCESS) { \
        throw std::runtime_error("CURAND Error: " + std::to_string(status)); \
    } \
}

// Logger for TensorRT
class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

namespace xinfer::zoo::generative
{
    // --- PIMPL Idiom Implementation ---
    // All the complex state is hidden here.
    struct DCGAN_Generator::Impl
    {
        Logger logger_;
        std::unique_ptr<nvinfer1::IRuntime> runtime_;
        std::unique_ptr<nvinfer1::ICudaEngine> engine_;
        std::unique_ptr<nvinfer1::IExecutionContext> context_;

        void* input_buffer_ = nullptr;
        void* output_buffer_ = nullptr;

        int noise_dim_;
        std::vector<int64_t> output_shape_;

        cudaStream_t stream_;
        curandGenerator_t noise_generator_;

        Impl(const std::string& engine_path)
        {
            // 1. Load the TensorRT engine from file
            std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
            if (!file) throw std::runtime_error("Could not open engine file: " + engine_path);

            std::streamsize size = file.tellg();
            file.seekg(0, std::ios::beg);
            std::vector<char> buffer(size);
            if (!file.read(buffer.data(), size)) throw std::runtime_error("Could not read engine file.");

            runtime_.reset(nvinfer1::createInferRuntime(logger_));
            engine_.reset(runtime_->deserializeCudaEngine(buffer.data(), size));
            if (!engine_) throw std::runtime_error("Failed to deserialize TensorRT Engine.");

            context_.reset(engine_->createExecutionContext());
            if (!context_) throw std::runtime_error("Failed to create TensorRT Execution Context.");

            // 2. Allocate GPU buffers for input and output
            auto input_dims = engine_->getBindingDimensions(0); // Assuming input is at binding 0
            noise_dim_ = input_dims.d[1]; // Shape is [N, C, H, W], so C is noise_dim
            size_t input_size = (size_t)engine_->getMaxBatchSize() * noise_dim_ * 1 * 1 * sizeof(float);
            CHECK_CUDA(cudaMalloc(&input_buffer_, input_size));

            auto output_dims = engine_->getBindingDimensions(1); // Assuming output is at binding 1
            output_shape_ = {(int64_t)engine_->getMaxBatchSize(), output_dims.d[1], output_dims.d[2], output_dims.d[3]};
            size_t output_size = (size_t)output_shape_[0] * output_shape_[1] * output_shape_[2] * output_shape_[3] *
                sizeof(float);
            CHECK_CUDA(cudaMalloc(&output_buffer_, output_size));

            // 3. Initialize CUDA Stream and cuRAND Generator
            CHECK_CUDA(cudaStreamCreate(&stream_));
            CHECK_CURAND(curandCreateGenerator(&noise_generator_, CURAND_RNG_PSEUDO_DEFAULT));
            CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(noise_generator_, 1234ULL));
        }

        ~Impl()
        {
            cudaFree(input_buffer_);
            cudaFree(output_buffer_);
            curandDestroyGenerator(noise_generator_);
            cudaStreamDestroy(stream_);
            // Smart pointers handle the rest
        }
    };

    // --- Public Class Method Implementations ---

    DCGAN_Generator::DCGAN_Generator(const std::string& engine_path)
        : pimpl_(new Impl(engine_path))
    {
    }

    DCGAN_Generator::~DCGAN_Generator() = default;
    DCGAN_Generator::DCGAN_Generator(DCGAN_Generator&&) noexcept = default;
    DCGAN_Generator& DCGAN_Generator::operator=(DCGAN_Generator&&) noexcept = default;


    core::Tensor DCGAN_Generator::generate(int batch_size)
    {
        if (!pimpl_)
        {
            throw std::runtime_error("DCGAN_Generator is in a moved-from state.");
        }
        if (batch_size > pimpl_->engine_->getMaxBatchSize())
        {
            throw std::invalid_argument("Requested batch size exceeds the max batch size of the engine.");
        }

        // 1. Generate random noise directly on the GPU
        CHECK_CURAND(curandGenerateNormal(pimpl_->noise_generator_,
            (float*)pimpl_->input_buffer_,
            (size_t)batch_size * pimpl_->noise_dim_,
            0.0f, 1.0f));

        // 2. Set up the bindings for this specific inference call
        void* bindings[2] = {pimpl_->input_buffer_, pimpl_->output_buffer_};

        // 3. Run inference asynchronously
        pimpl_->context_->enqueueV2(bindings, pimpl_->stream_, nullptr);

        // 4. Create an xInfer Tensor to return the result.
        //    This is a "view" of the internal buffer; no data is copied here.
        std::vector<int64_t> current_shape = {
            (int64_t)batch_size, pimpl_->output_shape_[1], pimpl_->output_shape_[2], pimpl_->output_shape_[3]
        };
        core::Tensor result_tensor(current_shape, core::DataType::kFLOAT);

        // 5. Copy the data from our internal buffer to the user's tensor buffer
        CHECK_CUDA(cudaMemcpyAsync(result_tensor.data(),
            pimpl_->output_buffer_,
            result_tensor.size_bytes(),
            cudaMemcpyDeviceToDevice,
            pimpl_->stream_));

        // 6. Wait for the operations to complete
        CHECK_CUDA(cudaStreamSynchronize(pimpl_->stream_));

        return result_tensor;
    }
} // namespace xinfer::zoo::generative
