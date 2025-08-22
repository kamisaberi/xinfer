#include <include/preproc/image_processor.h>
#include <cuda_runtime.h>

// Helper to check CUDA calls
#define CHECK_CUDA(call) { /* ... */ }

// --- THE FUSED PRE-PROCESSING KERNEL ---
__global__ void fused_preproc_kernel(
    const unsigned char* input_image,
    float* output_tensor,
    int in_w, int in_h, int in_channels,
    int out_w, int out_h,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < out_w && y < out_h) {
        // Simple nearest-neighbor resize mapping
        int in_x = x * in_w / out_w;
        int in_y = y * in_h / out_h;

        int in_idx = (in_y * in_w + in_x) * in_channels;

        // Assuming input is BGR from OpenCV
        float b = static_cast<float>(input_image[in_idx + 0]);
        float g = static_cast<float>(input_image[in_idx + 1]);
        float r = static_cast<float>(input_image[in_idx + 2]);

        // Normalize and convert to NCHW layout
        output_tensor[0 * out_h * out_w + y * out_w + x] = (r / 255.0f - mean_r) / std_r;
        output_tensor[1 * out_h * out_w + y * out_w + x] = (g / 255.0f - mean_g) / std_g;
        output_tensor[2 * out_h * out_w + y * out_w + x] = (b / 255.0f - mean_b) / std_b;
    }
}


namespace xinfer::preproc {

struct ImageProcessor::Impl {
    core::Tensor d_temp_image_; // A temporary GPU buffer to hold the uploaded image
    int out_w_, out_h_;
    std::vector<float> mean_, std_;
};

ImageProcessor::ImageProcessor(int width, int height, const std::vector<float>& mean, const std::vector<float>& std)
    : pimpl_(new Impl()) {
    pimpl_->out_w_ = width;
    pimpl_->out_h_ = height;
    pimpl_->mean_ = mean;
    pimpl_->std_ = std;
}
ImageProcessor::~ImageProcessor() = default;

void ImageProcessor::process(const cv::Mat& cpu_image, core::Tensor& output_tensor) {
    const int in_w = cpu_image.cols;
    const int in_h = cpu_image.rows;
    const int in_c = cpu_image.channels();
    const size_t in_bytes = in_w * in_h * in_c * sizeof(unsigned char);

    // Ensure our temporary GPU buffer is large enough
    if (pimpl_->d_temp_image_.size_bytes() < in_bytes) {
        pimpl_->d_temp_image_ = core::Tensor({in_h, in_w, in_c}, core::DataType::kINT8);
    }

    // Upload the image to the GPU
    CHECK_CUDA(cudaMemcpy(pimpl_->d_temp_image_.data(), cpu_image.data, in_bytes, cudaMemcpyHostToDevice));

    // Launch the fused kernel
    dim3 block(16, 16);
    dim3 grid((pimpl_->out_w_ + block.x - 1) / block.x, (pimpl_->out_h_ + block.y - 1) / block.y);

    fused_preproc_kernel<<<grid, block>>>(
        (const unsigned char*)pimpl_->d_temp_image_.data(),
        (float*)output_tensor.data(),
        in_w, in_h, in_c,
        pimpl_->out_w_, pimpl_->out_h_,
        pimpl_->mean_[0], pimpl_->mean_[1], pimpl_->mean_[2],
        pimpl_->std_[0], pimpl_->std_[1], pimpl_->std_[2]
    );
}

} // namespace xinfer::preproc