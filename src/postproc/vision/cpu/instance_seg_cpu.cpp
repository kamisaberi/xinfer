#include "instance_seg_cpu.h"
#include <xinfer/core/logging.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp> // For NMSBoxes
#include <cmath>
#include <cstring>
#include <algorithm>

namespace xinfer::postproc {

// =================================================================================
// Constructor / Init
// =================================================================================

CpuInstanceSegPostproc::CpuInstanceSegPostproc() {}
CpuInstanceSegPostproc::~CpuInstanceSegPostproc() {}

void CpuInstanceSegPostproc::init(const InstanceSegConfig& config) {
    m_config = config;
}

// =================================================================================
// Main Pipeline
// =================================================================================

std::vector<InstanceResult> CpuInstanceSegPostproc::process(const std::vector<core::Tensor>& tensors) {
    std::vector<InstanceResult> final_results;

    // Validation
    if (tensors.size() < 2) {
        XINFER_LOG_ERROR("Instance Seg requires 2 tensors (Detection Head, Proto Head).");
        return final_results;
    }

    const auto& det_tensor = tensors[0];   // e.g. [1, 116, 8400]
    const auto& proto_tensor = tensors[1]; // e.g. [1, 32, 160, 160]

    // 1. Decode Boxes & Coefficients
    std::vector<RawDetection> proposals;
    proposals.reserve(2000); // Reserve memory to avoid reallocations
    
    decode_yolo_seg(static_cast<const float*>(det_tensor.data()), det_tensor.shape(), proposals);

    if (proposals.empty()) return final_results;

    // 2. Prepare for NMS (OpenCV Format)
    std::vector<cv::Rect> boxes_cv;
    std::vector<float> scores_cv;
    boxes_cv.reserve(proposals.size());
    scores_cv.reserve(proposals.size());

    for (const auto& p : proposals) {
        boxes_cv.emplace_back((int)p.x1, (int)p.y1, (int)(p.x2 - p.x1), (int)(p.y2 - p.y1));
        scores_cv.push_back(p.score);
    }

    // 3. Run NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes_cv, scores_cv, m_config.conf_threshold, m_config.nms_threshold, indices);

    // Limit output count
    if (indices.size() > (size_t)m_config.max_detections) {
        indices.resize(m_config.max_detections);
    }

    // 4. Collect Valid Detections
    std::vector<RawDetection> valid_detections;
    valid_detections.reserve(indices.size());
    for (int idx : indices) {
        valid_detections.push_back(proposals[idx]);
    }

    // 5. Generate Masks
    if (!valid_detections.empty()) {
        resolve_masks(valid_detections, 
                      static_cast<const float*>(proto_tensor.data()), 
                      proto_tensor.shape(), 
                      final_results);
    }

    return final_results;
}

// =================================================================================
// Decoding Logic (YOLOv8-Seg)
// =================================================================================

void CpuInstanceSegPostproc::decode_yolo_seg(const float* data, 
                                             const std::vector<int64_t>& shape,
                                             std::vector<RawDetection>& proposals) {
    // YOLOv8 Output: [Batch, Channels, Anchors]
    // Channels = 4 (Box) + NumClasses + 32 (Mask Coeffs)
    
    int num_channels = (int)shape[1];
    int num_anchors  = (int)shape[2];
    int num_classes  = m_config.num_classes;
    int num_masks    = m_config.num_mask_protos;

    // Pointers into the strided memory (Column-Major access per anchor)
    // Row 0..3: Box
    // Row 4..4+Cls: Class Scores
    // Row 4+Cls..End: Mask Coeffs
    
    const float* p_cx = data;
    const float* p_cy = data + num_anchors;
    const float* p_w  = data + num_anchors * 2;
    const float* p_h  = data + num_anchors * 3;
    const float* p_cls_start = data + num_anchors * 4;
    const float* p_mask_start = data + num_anchors * (4 + num_classes);

    for (int i = 0; i < num_anchors; ++i) {
        // Find best class
        float max_score = 0.0f;
        int class_id = -1;

        for (int c = 0; c < num_classes; ++c) {
            float score = p_cls_start[c * num_anchors + i];
            if (score > max_score) {
                max_score = score;
                class_id = c;
            }
        }

        if (max_score >= m_config.conf_threshold) {
            // Box Decoding
            float cx = p_cx[i];
            float cy = p_cy[i];
            float w  = p_w[i];
            float h  = p_h[i];

            float x1 = cx - w * 0.5f;
            float y1 = cy - h * 0.5f;
            float x2 = cx + w * 0.5f;
            float y2 = cy + h * 0.5f;

            // Extract Mask Coefficients
            std::vector<float> coeffs(num_masks);
            for (int m = 0; m < num_masks; ++m) {
                coeffs[m] = p_mask_start[m * num_anchors + i];
            }

            proposals.push_back({x1, y1, x2, y2, max_score, class_id, std::move(coeffs)});
        }
    }
}

// =================================================================================
// Mask Generation Logic (Matrix Multiply)
// =================================================================================

void CpuInstanceSegPostproc::resolve_masks(const std::vector<RawDetection>& detections,
                                           const float* proto_data,
                                           const std::vector<int64_t>& proto_shape,
                                           std::vector<InstanceResult>& results) {
    // 1. Setup Prototype Matrix
    // Shape: [32, 160, 160] -> Treat as [32, 25600]
    int proto_channels = (int)proto_shape[1]; // 32
    int proto_h = (int)proto_shape[2];        // 160
    int proto_w = (int)proto_shape[3];        // 160
    int proto_area = proto_h * proto_w;

    // Wrap raw pointer (Zero Copy)
    cv::Mat mat_proto(proto_channels, proto_area, CV_32F, const_cast<float*>(proto_data));

    // 2. Setup Coefficients Matrix
    // Shape: [NumDetections, 32]
    int num_dets = (int)detections.size();
    cv::Mat mat_coeffs(num_dets, proto_channels, CV_32F);

    for (int i = 0; i < num_dets; ++i) {
        // Copy float vector to OpenCV row
        memcpy(mat_coeffs.ptr<float>(i), detections[i].mask_coeffs.data(), proto_channels * sizeof(float));
    }

    // 3. Matrix Multiplication (GEMM)
    // [N, 32] * [32, 25600] = [N, 25600]
    // This calculates the linear combination of prototypes for ALL detections at once.
    cv::Mat mat_masks_flat = mat_coeffs * mat_proto; 

    // 4. Post-process each mask
    for (int i = 0; i < num_dets; ++i) {
        const auto& det = detections[i];
        InstanceResult res;
        res.box = {det.x1, det.y1, det.x2, det.y2, det.score, det.class_id};

        // Get the specific row for this detection, reshape to [160, 160]
        cv::Mat mask_raw(proto_h, proto_w, CV_32F, mat_masks_flat.ptr<float>(i));

        // Sigmoid Activation
        cv::Mat mask_sig;
        cv::exp(-mask_raw, mask_sig);
        mask_sig = 1.0f / (1.0f + mask_sig);

        // Resize to Original Image Dimensions
        cv::Mat mask_resized;
        cv::resize(mask_sig, mask_resized, cv::Size(m_config.target_width, m_config.target_height), 0, 0, cv::INTER_LINEAR);

        // Crop: Set pixels outside the bounding box to 0
        // (Fast logic: create a ROI header)
        cv::Rect box_rect(
            std::max(0, (int)det.x1),
            std::max(0, (int)det.y1),
            std::min(m_config.target_width - (int)det.x1, (int)(det.x2 - det.x1)),
            std::min(m_config.target_height - (int)det.y1, (int)(det.y2 - det.y1))
        );

        // Thresholding (0.5) to binary mask
        // Initialize full black mask
        cv::Mat final_mask = cv::Mat::zeros(m_config.target_height, m_config.target_width, CV_8U);
        
        // Only process the ROI inside the box
        if (box_rect.width > 0 && box_rect.height > 0) {
            cv::Mat roi = mask_resized(box_rect);
            // Threshold logic: val > 0.5 ? 255 : 0
            cv::Mat roi_binary;
            cv::threshold(roi, roi_binary, 0.5, 255, cv::THRESH_BINARY);
            roi_binary.convertTo(roi_binary, CV_8U);
            
            // Copy ROI to final mask
            roi_binary.copyTo(final_mask(box_rect));
        }

        // Store in Tensor
        res.mask.resize({1, (int64_t)m_config.target_height, (int64_t)m_config.target_width}, core::DataType::kUINT8);
        memcpy(res.mask.data(), final_mask.data, res.mask.size());

        final_results.push_back(res);
    }

    return final_results;
}

} // namespace xinfer::postproc