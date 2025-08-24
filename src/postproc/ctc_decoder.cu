#include <include/postproc/ctc_decoder.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(error))); \
    } \
}

namespace xinfer::postproc::ctc {

__global__ void ctc_greedy_decode_kernel(
    const float* logits,
    int* decoded_indices,
    float* decoded_probs,
    int T, int C)
{
    int t = blockIdx.x;
    if (t >= T) return;

    float max_val = -1e20f;
    int max_idx = -1;
    const float* timestep_logits = logits + t * C;

    for (int c = 0; c < C; ++c) {
        if (timestep_logits[c] > max_val) {
            max_val = timestep_logits[c];
            max_idx = c;
        }
    }

    float sum_exp = 0.0f;
    for (int c = 0; c < C; ++c) {
        sum_exp += expf(timestep_logits[c] - max_val);
    }

    decoded_indices[t] = max_idx;
    decoded_probs[t] = 1.0f / sum_exp;
}

std::pair<std::string, float> decode(
    const core::Tensor& logits,
    const std::vector<std::string>& character_map)
{
    auto shape = logits.shape();
    if (shape.size() != 3 || shape[0] != 1) {
        throw std::runtime_error("CTC decode expects a single-batch tensor of shape [1, T, C]");
    }
    const int T = shape[1];
    const int C = shape[2];

    core::Tensor d_indices({(long long)T}, core::DataType::kINT32);
    core::Tensor d_probs({(long long)T}, core::DataType::kFLOAT);

    ctc_greedy_decode_kernel<<<T, 1>>>(
        static_cast<const float*>(logits.data()),
        static_cast<int*>(d_indices.data()),
        static_cast<float*>(d_probs.data()),
        T, C
    );
    CHECK_CUDA(cudaGetLastError());

    std::vector<int> h_indices(T);
    std::vector<float> h_probs(T);
    d_indices.copy_to_host(h_indices.data());
    d_probs.copy_to_host(h_probs.data());

    std::string text = "";
    float confidence = 1.0f;
    int prev_char_idx = -1;
    for (int i = 0; i < T; ++i) {
        int char_idx = h_indices[i];
        if (char_idx != 0 && char_idx != prev_char_idx) {
            if (char_idx < character_map.size()) {
                text += character_map[char_idx];
                confidence *= h_probs[i];
            }
        }
        prev_char_idx = char_idx;
    }

    return {text, confidence};
}

} // namespace xinfer::postproc::ctc