#include <include/postproc/ocr_decoder.h>
#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include <algorithm>

#define CHECK_CUDA(call) { /* ... */ }

namespace xinfer::postproc::ocr {

// The CPU-based part for finding bounding boxes from the CRAFT model's output
// A real implementation would be very complex, involving OpenCV's connected components.
// This is a placeholder for that logic.
std::vector<std::vector<cv::Point2f>> decode_detection_output_craft(
    const core::Tensor& region_scores, const core::Tensor& affinity_scores,
    float text_threshold, float box_threshold,
    float scale_w, float scale_h)
{
    // This is a highly complex algorithm involving watershed, connected components,
    // and polygon fitting. For this example, we return a dummy box.
    // In a real product, you would integrate the official CRAFT post-processing code here.
    std::cout << "Warning: CRAFT decoder is a placeholder." << std::endl;
    std::vector<std::vector<cv::Point2f>> boxes;
    // boxes.push_back({{50, 50}, {200, 50}, {200, 100}, {50, 100}});
    return boxes;
}

// Uses OpenCV to warp the detected region into a flat patch
cv::Mat get_warped_text_patch(const cv::Mat& image, const std::vector<cv::Point2f>& box, int target_height) {
    // A real implementation would order the points and calculate a precise
    // output width based on the aspect ratio. This is a simplified version.
    float width = cv::norm(box[0] - box[1]);
    float height = cv::norm(box[1] - box[2]);
    int target_width = static_cast<int>(target_height * (width / height));
    target_width = std::max(1, target_width);

    cv::Point2f src[4] = {box[0], box[1], box[2], box[3]};
    cv::Point2f dst[4] = {{0,0}, {(float)target_width-1, 0}, {(float)target_width-1, (float)target_height-1}, {0, (float)target_height-1}};

    cv::Mat matrix = cv::getPerspectiveTransform(src, dst);
    cv::Mat out_patch;
    cv::warpPerspective(image, out_patch, matrix, cv::Size(target_width, target_height));
    return out_patch;
}


// --- THE CUDA KERNEL FOR CTC DECODING ---
__global__ void ctc_greedy_decode_kernel(
    const float* logits,         // Shape [Timesteps, NumClasses]
    int* decoded_indices,        // Output for raw class indices [Timesteps]
    float* decoded_probs,        // Output for probabilities [Timesteps]
    int T, int C)
{
    // Each thread block processes one timestep
    int t = blockIdx.x;
    if (t >= T) return;

    // Find the max class (argmax) for this timestep
    float max_val = -1e20f;
    int max_idx = -1;
    const float* timestep_logits = logits + t * C;

    for (int c = 0; c < C; ++c) {
        if (timestep_logits[c] > max_val) {
            max_val = timestep_logits[c];
            max_idx = c;
        }
    }

    // Apply softmax to get a probability
    float sum_exp = 0.0f;
    for (int c = 0; c < C; ++c) {
        sum_exp += expf(timestep_logits[c] - max_val);
    }

    decoded_indices[t] = max_idx;
    decoded_probs[t] = expf(max_val - max_val) / sum_exp; // = 1.0 / sum_exp
}

// The C++ function that orchestrates the CTC decoding
std::tuple<std::string, float> decode_recognition_output_ctc(
    const core::Tensor& logits, const std::vector<char>& character_map)
{
    auto shape = logits.shape(); // Expects [1, Timesteps, NumClasses]
    if (shape.size() != 3 || shape[0] != 1) {
        throw std::runtime_error("CTC decode expects a single-batch tensor of shape [1, T, C]");
    }
    const int T = shape[1];
    const int C = shape[2];

    // Allocate temporary GPU memory for the kernel's output
    core::Tensor d_indices({T}, core::DataType::kINT32);
    core::Tensor d_probs({T}, core::DataType::kFLOAT);

    // Launch the kernel
    ctc_greedy_decode_kernel<<<T, 1>>>(
        static_cast<const float*>(logits.data()),
        static_cast<int*>(d_indices.data()),
        static_cast<float*>(d_probs.data()),
        T, C
    );
    CHECK_CUDA(cudaGetLastError());

    // Copy the results back to the CPU
    std::vector<int> h_indices(T);
    std::vector<float> h_probs(T);
    d_indices.copy_to_host(h_indices.data());
    d_probs.copy_to_host(h_probs.data());

    // --- Perform the final CTC collapse on the CPU ---
    std::string text = "";
    float confidence = 1.0f;
    int prev_char_idx = -1;
    for (int i = 0; i < T; ++i) {
        int char_idx = h_indices[i];
        if (char_idx != 0 && char_idx != prev_char_idx) { // Not blank and not a repeat
            text += character_map[char_idx];
            confidence *= h_probs[i];
        }
        prev_char_idx = char_idx;
    }

    return {text, confidence};
}

} // namespace xinfer::postproc::ocr