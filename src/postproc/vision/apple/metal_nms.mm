#include "metal_nms.h"
#include <xinfer/core/logging.h>

#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>

namespace xinfer::postproc {

// =================================================================================
// 1. Metal Shader Source
// =================================================================================
// We assume boxes are [x1, y1, x2, y2]
static const char* NMS_SHADER_SRC = R"(
#include <metal_stdlib>
using namespace metal;

struct Box {
    float x1, y1, x2, y2;
    // Padding to match C++ alignment if necessary (structs in Metal are packed)
    // We will pass float4 arrays for simplicity.
};

// Calculate IoU between two boxes
inline float iou(float4 a, float4 b) {
    float area_a = (a.z - a.x) * (a.w - a.y);
    float area_b = (b.z - b.x) * (b.w - b.y);

    float inter_x1 = max(a.x, b.x);
    float inter_y1 = max(a.y, b.y);
    float inter_x2 = min(a.z, b.z);
    float inter_y2 = min(a.w, b.y);

    float w = max(0.0f, inter_x2 - inter_x1);
    float h = max(0.0f, inter_y2 - inter_y1);
    float area_inter = w * h;

    return area_inter / (area_a + area_b - area_inter + 1e-6f);
}

// Kernel: Compute IoU Bitmap
// Each thread (id.x) corresponds to one "Candidate" box.
// It checks overlap against all higher-scoring boxes (id.y < id.x) or just fills a row.
//
// Optimization: We map the grid to (N, N).
// Grid coordinates (col, row). 
// We write to a boolean adjacency matrix (flattened char array).
kernel void iou_matrix_kernel(device const float4* boxes [[buffer(0)]],
                              device uchar* iou_mask   [[buffer(1)]],
                              constant uint& num_boxes [[buffer(2)]],
                              constant float& threshold [[buffer(3)]],
                              uint2 gid [[thread_position_in_grid]])
{
    uint col = gid.x; // Box A index
    uint row = gid.y; // Box B index

    if (col >= num_boxes || row >= num_boxes) return;

    // Only compute upper triangle (since IoU is symmetric)
    // and we usually only care if a lower-score box (higher index in sorted list) 
    // overlaps with a higher-score box (lower index).
    // So we assume list is sorted: score[0] > score[1] ...
    
    // We check if Box[row] (High Score) overlaps Box[col] (Low Score)
    // So we only process if row < col
    if (row >= col) {
        iou_mask[row * num_boxes + col] = 0;
        return;
    }

    float4 b1 = boxes[row];
    float4 b2 = boxes[col];

    float val = iou(b1, b2);
    
    // If IoU > threshold, mark as '1' (Suppress 'col' because 'row' exists)
    iou_mask[row * num_boxes + col] = (val > threshold) ? 1 : 0;
}
)";

// =================================================================================
// 2. PImpl Implementation
// =================================================================================

struct MetalNMS::Impl {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipelineState;

    // Cached buffers to reduce allocation overhead
    id<MTLBuffer> d_boxes = nil;
    id<MTLBuffer> d_mask = nil;
    size_t capacity_boxes = 0;

    Impl() {
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            XINFER_LOG_ERROR("Metal Device not found.");
            return;
        }
        queue = [device newCommandQueue];
        build_pipeline();
    }

    void build_pipeline() {
        NSError* error = nil;
        NSString* src = [NSString stringWithUTF8String:NMS_SHADER_SRC];
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:nil error:&error];
        if (!lib) {
            XINFER_LOG_ERROR("NMS Shader Error: " + std::string(error.localizedDescription.UTF8String));
            return;
        }
        
        id<MTLFunction> fn = [lib newFunctionWithName:@"iou_matrix_kernel"];
        pipelineState = [device newComputePipelineStateWithFunction:fn error:&error];
    }

    void reserve(size_t num_boxes) {
        // Prepare flattened box buffer: [x1,y1,x2,y2] * N
        // float4 is 16 bytes.
        size_t box_bytes = num_boxes * sizeof(float) * 4;
        
        // Prepare Mask: N*N bytes (using char for bool)
        size_t mask_bytes = num_boxes * num_boxes * sizeof(uint8_t);

        if (num_boxes > capacity_boxes) {
            d_boxes = [device newBufferWithLength:box_bytes options:MTLResourceStorageModeShared];
            d_mask  = [device newBufferWithLength:mask_bytes options:MTLResourceStorageModeShared];
            capacity_boxes = num_boxes;
        }
    }
};

// =================================================================================
// 3. Class Implementation
// =================================================================================

MetalNMS::MetalNMS() : m_impl(std::make_unique<Impl>()) {}
MetalNMS::~MetalNMS() = default;

std::vector<int> MetalNMS::process(const std::vector<BoundingBox>& boxes, 
                                   float iou_threshold, 
                                   int max_output_boxes) 
{
    if (boxes.empty()) return {};
    if (!m_impl->device) return {}; // Guard

    // 1. Sort indices by score (CPU)
    // We create a list of indices and sort them, rather than sorting the structs,
    // to keep track of original IDs if needed.
    int n = boxes.size();
    std::vector<int> sorted_indices(n);
    for(int i=0; i<n; ++i) sorted_indices[i] = i;

    std::sort(sorted_indices.begin(), sorted_indices.end(), 
        [&boxes](int i1, int i2) {
            return boxes[i1].confidence > boxes[i2].confidence;
        }
    );

    // 2. Upload sorted boxes to Metal Buffer (Shared Memory)
    m_impl->reserve(n);
    
    // float4* pointer to mapped memory
    float* gpu_boxes_ptr = (float*)m_impl->d_boxes.contents;
    
    for (int i = 0; i < n; ++i) {
        const auto& b = boxes[sorted_indices[i]];
        // Pack into float4
        gpu_boxes_ptr[i*4 + 0] = b.x1;
        gpu_boxes_ptr[i*4 + 1] = b.y1;
        gpu_boxes_ptr[i*4 + 2] = b.x2;
        gpu_boxes_ptr[i*4 + 3] = b.y2;
    }

    // 3. Run IoU Matrix Kernel
    @autoreleasepool {
        id<MTLCommandBuffer> cmdbuf = [m_impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdbuf computeCommandEncoder];
        
        [enc setComputePipelineState:m_impl->pipelineState];
        [enc setBuffer:m_impl->d_boxes offset:0 atIndex:0];
        [enc setBuffer:m_impl->d_mask offset:0 atIndex:1];
        [enc setBytes:&n length:sizeof(uint) atIndex:2];
        [enc setBytes:&iou_threshold length:sizeof(float) atIndex:3];

        // 2D Grid [N, N]
        MTLSize gridSize = MTLSizeMake(n, n, 1);
        
        NSUInteger w = m_impl->pipelineState.threadExecutionWidth;
        NSUInteger h = m_impl->pipelineState.maxTotalThreadsPerThreadgroup / w;
        MTLSize threadGroupSize = MTLSizeMake(w, h, 1);

        [enc dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
        [enc endEncoding];
        
        [cmdbuf commit];
        [cmdbuf waitUntilCompleted];
    }

    // 4. Suppression Loop (CPU reading shared memory)
    // The mask[row * N + col] tells us if box 'row' suppresses box 'col'.
    // Since we sorted by score, 'row' is always higher score than 'col' in our kernel logic loop.
    
    uint8_t* mask_ptr = (uint8_t*)m_impl->d_mask.contents;
    std::vector<int> keep_indices;
    std::vector<bool> suppressed(n, false);

    for (int i = 0; i < n; ++i) {
        if (suppressed[i]) continue;

        // Keep this box
        keep_indices.push_back(sorted_indices[i]);
        if (keep_indices.size() >= max_output_boxes) break;

        // Mark all subsequent boxes that overlap significantly with this one
        // The GPU has already computed IoU > thresh for us.
        // We look at the row 'i' in the adjacency matrix.
        uint8_t* row_ptr = mask_ptr + (i * n);
        
        for (int j = i + 1; j < n; ++j) {
            if (!suppressed[j]) {
                if (row_ptr[j]) { // GPU calculated IoU > thresh
                    suppressed[j] = true;
                }
            }
        }
    }

    return keep_indices;
}

} // namespace xinfer::postproc