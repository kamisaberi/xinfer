#include "ctc_decoder.h"
#include <xinfer/core/logging.h>
#include <algorithm>
#include <vector>

namespace xinfer::postproc {

CtcDecoder::CtcDecoder() {}
CtcDecoder::~CtcDecoder() {}

void CtcDecoder::init(const OcrConfig& config) {
    m_config = config;
    
    // Validation
    if (m_config.vocabulary.empty()) {
        XINFER_LOG_WARN("OCR Vocab is empty. Decoded strings will be empty.");
    }
}

// Helper: Decode one time-sequence
std::string CtcDecoder::decode_sequence(const float* seq_data, 
                                        int time_steps, 
                                        int num_classes,
                                        int stride) {
    std::string result;
    int last_index = -1; // -1 indicates no previous character

    for (int t = 0; t < time_steps; ++t) {
        // Find ArgMax for current time step
        const float* probs = seq_data + (t * stride);
        
        int max_idx = -1;
        float max_val = -1.0f;

        for (int c = 0; c < num_classes; ++c) {
            if (probs[c] > max_val) {
                max_val = probs[c];
                max_idx = c;
            }
        }

        // Apply Confidence Threshold
        if (max_val < m_config.min_confidence) {
            max_idx = m_config.blank_index; // Treat low conf as blank
        }

        // CTC Logic:
        // 1. If Blank, reset last_index and continue
        // 2. If Same as last index (and not blank), skip (Merge repeats)
        // 3. Else, append character
        
        if (max_idx != m_config.blank_index) {
            if (max_idx != last_index) {
                // Bounds check for vocab
                if (max_idx >= 0 && max_idx < m_config.vocabulary.size()) {
                    result += m_config.vocabulary[max_idx];
                }
            }
        }
        
        last_index = max_idx;
    }
    return result;
}

std::vector<std::string> CtcDecoder::process(const core::Tensor& logits) {
    std::vector<std::string> results;
    
    auto shape = logits.shape();
    if (shape.size() != 3) {
        XINFER_LOG_ERROR("CTC Decoder expects 3D tensor [T, B, C] or [B, T, C].");
        return results;
    }

    // Heuristic to detect layout:
    // Usually Batch size is small (1-32), Time is large (10-100), Classes is vocab size (30-1000).
    // Standard PyTorch CRNN is [T, B, C].
    // Standard Keras OCR is [B, T, C].
    
    // Let's assume Batch Major [B, T, C] if dim[0] < dim[1], else Time Major [T, B, C].
    // This is heuristic and might need explicit config flag in production.
    bool batch_major = (shape[0] < shape[1]);

    int batch, time_steps, num_classes;
    
    if (batch_major) {
        batch = (int)shape[0];
        time_steps = (int)shape[1];
        num_classes = (int)shape[2];
    } else {
        time_steps = (int)shape[0];
        batch = (int)shape[1];
        num_classes = (int)shape[2];
    }

    const float* data = static_cast<const float*>(logits.data());

    // Iterate over batch
    for (int b = 0; b < batch; ++b) {
        if (batch_major) {
            // [B, T, C] -> Row b is contiguous segment of size (T * C)
            const float* row_ptr = data + (b * time_steps * num_classes);
            // Stride between time steps is num_classes
            results.push_back(decode_sequence(row_ptr, time_steps, num_classes, num_classes));
        } else {
            // [T, B, C] -> Time steps are strided by (Batch * NumClasses)
            // This is non-contiguous for a single sequence.
            // We need a custom stride loop or copy.
            // Stride between time steps is (Batch * NumClasses)
            // Offset for batch b is (b * NumClasses)
            
            // To reuse decode_sequence, we need to gather or handle striding inside.
            // Let's implement a gather here for clarity.
            std::string batch_res;
            int last_idx = -1;
            
            int time_stride = batch * num_classes; // Jump to next time step
            int batch_offset = b * num_classes;    // Offset within time step

            for (int t = 0; t < time_steps; ++t) {
                const float* probs = data + (t * time_stride) + batch_offset;
                
                // ArgMax
                int max_idx = -1; 
                float max_val = -1.0f;
                for(int c=0; c<num_classes; ++c) {
                    if (probs[c] > max_val) { max_val = probs[c]; max_idx = c; }
                }

                if (max_val >= m_config.min_confidence && max_idx != m_config.blank_index) {
                    if (max_idx != last_idx) {
                        if (max_idx < m_config.vocabulary.size()) {
                            batch_res += m_config.vocabulary[max_idx];
                        }
                    }
                }
                last_idx = max_idx;
            }
            results.push_back(batch_res);
        }
    }

    return results;
}

} // namespace xinfer::postproc