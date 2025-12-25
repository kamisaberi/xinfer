#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>

using namespace xinfer;

#define CHECK_CUDA(call) { if(call != cudaSuccess) std::cerr << "CUDA Error" << std::endl; }

int main() {
    Target target = Target::NVIDIA_TRT;

    // 1. Setup Engine
    auto engine = backends::BackendFactory::create(target);
    engine->load_model("yolo_orin.engine");

    // 2. Allocate Unified Memory (Mapped)
    // We allocate ONCE. Both CPU (OpenCV) and GPU (TensorRT) will use this buffer.
    size_t width = 640;
    size_t height = 640;
    size_t channels = 3;
    size_t size_bytes = width * height * channels * sizeof(float);

    void* cpu_ptr = nullptr;

    // Allocate Host Pinned memory that is mapped into Device address space
    CHECK_CUDA(cudaHostAlloc(&cpu_ptr, size_bytes, cudaHostAllocMapped));

    // Get the Device pointer that corresponds to this Host pointer
    void* gpu_ptr = nullptr;
    CHECK_CUDA(cudaHostGetDevicePointer(&gpu_ptr, cpu_ptr, 0));

    std::cout << "Allocated Unified Memory. CPU: " << cpu_ptr << " GPU: " << gpu_ptr << std::endl;

    // 3. Create xInfer Tensor wrapping the GPU pointer
    // We tell xInfer: "Don't malloc, use this existing GPU pointer."
    core::Tensor input_tensor;
    input_tensor.set_external_handle(gpu_ptr, size_bytes, core::MemoryType::CudaDevice);
    // Note: We need to set shape manually since we bypassed allocation
    input_tensor.reshape({1, 3, (int64_t)height, (int64_t)width}, core::DataType::kFLOAT);

    core::Tensor output_tensor;

    // 4. Setup Post-processing
    auto postproc = postproc::create_detection(target);
    postproc::DetectionConfig cfg;
    postproc->init(cfg);

    // 5. Capture Loop
    cv::VideoCapture cap(0);
    cv::Mat frame;
    cv::Mat resized_wrapper(height, width, CV_32FC3, cpu_ptr); // OpenCV writes to cpu_ptr

    while (cap.read(frame)) {
        // Step A: CPU Writes to Shared Memory
        // We use OpenCV to resize/normalize directly into the mapped buffer
        // No cudaMemcpy needed after this!
        cv::Mat resized_u8;
        cv::resize(frame, resized_u8, cv::Size(width, height));

        // Convert to Float and Normalize (0-1) directly into 'cpu_ptr' via 'resized_wrapper'
        resized_u8.convertTo(resized_wrapper, CV_32FC3, 1.0/255.0);

        // Optional: If model needs NCHW, we must permute here on CPU or use CudaPreproc.
        // For Jetson, CudaImagePreprocessor is usually better, but this example demonstrates
        // the memory mapping mechanism.
        // Let's assume the model accepts NHWC or we did the permute in the loop above.

        // Step B: Inference
        // TensorRT reads from 'gpu_ptr', which is physically the same RAM as 'cpu_ptr'
        engine->predict({input_tensor}, {output_tensor});

        // Step C: Post-process
        auto dets = postproc->process({output_tensor});

        std::cout << "Detections: " << dets.size() << std::endl;
    }

    // Cleanup
    CHECK_CUDA(cudaFreeHost(cpu_ptr));
    return 0;
}