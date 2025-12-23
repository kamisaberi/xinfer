#include "metal_image.h"
#include <xinfer/core/logging.h>

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <vector>
#include <string>

namespace xinfer::preproc {

// =================================================================================
// 1. Embedded Metal Shader Source
// =================================================================================
// This kernel converts Texture (RGBA) -> Linear Buffer (NCHW Float32)
// and applies Mean/Std normalization.
static const char* PREPROC_SHADER_SRC = R"(
#include <metal_stdlib>
using namespace metal;

struct NormParams {
    float3 mean;
    float3 std;
    float global_scale;
    uint output_width;
    uint output_height;
};

// Kernel: Normalize and Transpose (HWC/RGBA -> NCHW Planar)
kernel void preprocess_nchw(
    texture2d<float, access::read> inTexture [[texture(0)]],
    device float* outBuffer [[buffer(0)]],
    constant NormParams& params [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= inTexture.get_width() || gid.y >= inTexture.get_height()) {
        return;
    }

    // Read pixel (Metal textures are usually normalized 0.0-1.0 float automatically)
    // If input was uint8, Metal reads it as unorm float.
    float4 pixel = inTexture.read(gid);

    // Apply Global Scaling if input wasn't already normalized by texture sampler
    // (Usually texture read gives [0,1], so global_scale might be 1.0 or 255.0 depending on setup)
    float3 rgb = float3(pixel.r, pixel.g, pixel.b);
    
    // Normalize: (x - mean) / std
    float3 normalized = (rgb - params.mean) / params.std;

    // Calculate Output Indices for NCHW Layout
    // Planar offset size
    uint plane_size = params.output_width * params.output_height;
    uint spatial_idx = gid.y * params.output_width + gid.x;

    // Write Red
    outBuffer[spatial_idx] = normalized.r;
    // Write Green
    outBuffer[spatial_idx + plane_size] = normalized.g;
    // Write Blue
    outBuffer[spatial_idx + 2 * plane_size] = normalized.b;
}
)";

// =================================================================================
// 2. PImpl Implementation (Objective-C++)
// =================================================================================

struct MetalImagePreprocessor::Impl {
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLComputePipelineState> pipelineState;
    
    // MPS Resizer
    MPSImageLanczosScale* scaler = nil;

    // Cache texture descriptors to avoid reallocation
    MTLTextureDescriptor* inputDesc = nil;
    
    // Intermediate texture for resizing result (if resize is needed)
    id<MTLTexture> resizedTexture = nil;

    Impl() {
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            XINFER_LOG_ERROR("Failed to create Metal Device.");
            return;
        }
        commandQueue = [device newCommandQueue];
        build_pipeline();
    }

    void build_pipeline() {
        NSError* error = nil;
        NSString* src = [NSString stringWithUTF8String:PREPROC_SHADER_SRC];
        id<MTLLibrary> library = [device newLibraryWithSource:src options:nil error:&error];
        
        if (!library) {
            XINFER_LOG_ERROR("Metal Shader compilation failed: " + std::string(error.localizedDescription.UTF8String));
            return;
        }

        id<MTLFunction> fn = [library newFunctionWithName:@"preprocess_nchw"];
        pipelineState = [device newComputePipelineStateWithFunction:fn error:&error];
        
        if (!pipelineState) {
            XINFER_LOG_ERROR("Failed to create Compute Pipeline.");
        }
    }
};

// =================================================================================
// 3. Class Implementation
// =================================================================================

MetalImagePreprocessor::MetalImagePreprocessor() 
    : m_impl(std::make_unique<Impl>()) {
}

MetalImagePreprocessor::~MetalImagePreprocessor() = default;

void MetalImagePreprocessor::init(const ImagePreprocConfig& config) {
    m_config = config;

    @autoreleasepool {
        // Initialize MPS Scaler
        m_impl->scaler = [[MPSImageLanczosScale alloc] initWithDevice:m_impl->device];
        
        // Prepare Resized Texture Container
        MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                        width:config.target_width
                                                                                       height:config.target_height
                                                                                    mipmapped:NO];
        desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        // On Apple Silicon, we want Shared mode for CPU/GPU access if debugging, 
        // but Private is faster for GPU-only intermediate.
        desc.storageMode = MTLStorageModePrivate; 
        
        m_impl->resizedTexture = [m_impl->device newTextureWithDescriptor:desc];
    }
}

void MetalImagePreprocessor::process(const ImageFrame& src, core::Tensor& dst) {
    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [m_impl->commandQueue commandBuffer];
        
        // 1. Upload Input to Texture
        // ----------------------------------------------------
        MTLTextureDescriptor* inDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                          width:src.width
                                                                                         height:src.height
                                                                                      mipmapped:NO];
        // Assuming src.data is RGBA for simplicity in this snippet. 
        // In prod, check src.format and use appropriate pixel format (e.g. R8 for grayscale).
        // Using `bytesPerRow` calc based on width * 4 (RGBA)
        
        id<MTLTexture> inputTexture = [m_impl->device newTextureWithDescriptor:inDesc];
        MTLRegion region = MTLRegionMake2D(0, 0, src.width, src.height);
        
        [inputTexture replaceRegion:region
                        mipmapLevel:0
                          withBytes:src.data
                        bytesPerRow:src.width * 4];

        // 2. Resize (Optional)
        // ----------------------------------------------------
        id<MTLTexture> sourceForCompute = inputTexture;
        
        if (src.width != m_config.target_width || src.height != m_config.target_height) {
            [m_impl->scaler encodeToCommandBuffer:commandBuffer
                                    sourceTexture:inputTexture
                               destinationTexture:m_impl->resizedTexture];
            sourceForCompute = m_impl->resizedTexture;
        }

        // 3. Compute (Normalize + NCHW)
        // ----------------------------------------------------
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        [computeEncoder setComputePipelineState:m_impl->pipelineState];
        [computeEncoder setTexture:sourceForCompute atIndex:0];

        // Wrap output Tensor in MTLBuffer (Zero-Copy if aligned, copy otherwise)
        // Ideally, xInfer Tensor allocator should use `posix_memalign` to be Metal friendly.
        // Here we use a temporary "no-copy" wrapper if possible, or newBufferWithBytes.
        
        id<MTLBuffer> outBuffer = [m_impl->device newBufferWithBytesNoCopy:dst.data()
                                                                    length:dst.size() * sizeof(float)
                                                                   options:MTLResourceStorageModeShared
                                                               deallocator:nil]; // Tensor owns memory
                                                               
        if (!outBuffer) {
            // Fallback if pointer not aligned: Allocate and copy back later
            outBuffer = [m_impl->device newBufferWithLength:dst.size() * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
        }

        [computeEncoder setBuffer:outBuffer offset:0 atIndex:0];

        // Set Params
        struct NormParams {
            float mean[3]; // Metal float3 is 16-byte aligned usually, padding careful
            float pad1;
            float std[3];
            float pad2;
            float global_scale;
            uint output_width;
            uint output_height;
        } params;

        // Populate params
        params.mean[0] = m_config.norm_params.mean[0];
        params.mean[1] = m_config.norm_params.mean[1];
        params.mean[2] = m_config.norm_params.mean[2];
        params.std[0]  = m_config.norm_params.std[0];
        params.std[1]  = m_config.norm_params.std[1];
        params.std[2]  = m_config.norm_params.std[2];
        params.global_scale = m_config.norm_params.scale_factor;
        params.output_width = m_config.target_width;
        params.output_height = m_config.target_height;

        [computeEncoder setBytes:&params length:sizeof(params) atIndex:1];

        // Dispatch
        NSUInteger w = m_impl->pipelineState.threadExecutionWidth;
        NSUInteger h = m_impl->pipelineState.maxTotalThreadsPerThreadgroup / w;
        MTLSize threadsPerThreadgroup = MTLSizeMake(w, h, 1);
        MTLSize threadsPerGrid = MTLSizeMake(m_config.target_width, m_config.target_height, 1);

        [computeEncoder dispatchThreads:threadsPerGrid
                  threadsPerThreadgroup:threadsPerThreadgroup];
        
        [computeEncoder endEncoding];

        // 4. Commit and Wait
        // ----------------------------------------------------
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // If we had to allocate a temp buffer due to alignment, copy back now
        if (outBuffer.contents != dst.data()) {
            memcpy(dst.data(), outBuffer.contents, dst.size() * sizeof(float));
        }
    }
}

} // namespace xinfer::preproc