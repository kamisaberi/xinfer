# src/postproc/CMakeLists.txt

# --- 1. Define Universal Sources (CPU) ---
set(POSTPROC_SRCS
    factory.cpp
    
    # Vision
    vision/cpu/yolo_postproc.cpp
    vision/cpu/segmentation_cpu.cpp
    vision/cpu/instance_seg_cpu.cpp
    vision/cpu/anomaly_cpu.cpp
    vision/cpu/detection3d_cpu.cpp
    vision/cpu/classification_cpu.cpp
    vision/cpu/tracker.cpp
    
    # Text
    text/cpu/ctc_decoder.cpp
    text/cpu/llm_sampler.cpp
    
    # Generative
    generative/cpu/diffusion_sampler.cpp
)

# --- 2. Platform-Specific Sources ---

# A. NVIDIA CUDA
if(XINFER_ENABLE_TRT)
    message(STATUS "[xInfer] Enabling CUDA Postprocessing kernels")
    list(APPEND POSTPROC_SRCS
        vision/cuda/yolo_postproc.cu
        vision/cuda/segmentation_cuda.cu
        vision/cuda/instance_seg_cuda.cu
        vision/cuda/anomaly_cuda.cu
        vision/cuda/detection3d_cuda.cu
        text/cuda/ctc_decoder.cu
        generative/cuda/diffusion_sampler.cu
    )
endif()

# B. Apple Metal
if(APPLE AND XINFER_ENABLE_COREML)
    message(STATUS "[xInfer] Enabling Apple Metal Postprocessing kernels")
    list(APPEND POSTPROC_SRCS
        vision/apple/metal_nms.mm
    )
endif()

# --- 3. Create Library ---
add_library(xinfer_postproc OBJECT ${POSTPROC_SRCS})

# --- 4. Link Dependencies ---
target_link_libraries(xinfer_postproc PRIVATE
    xinfer_core
    ${OpenCV_LIBS} # CPU NMS and other vision tasks depend on OpenCV
    
    # Conditional Links
    $<$<BOOL:${XINFER_ENABLE_TRT}>:CUDA::cudart>
)

# --- 5. Include Directories ---
target_include_directories(xinfer_postproc PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# --- 6. Platform-Specific Flags ---
if(XINFER_ENABLE_TRT)
    # Set language for .cu files
    set_source_files_properties(
        vision/cuda/yolo_postproc.cu
        vision/cuda/segmentation_cuda.cu
        vision/cuda/instance_seg_cuda.cu
        vision/cuda/anomaly_cuda.cu
        vision/cuda/detection3d_cuda.cu
        text/cuda/ctc_decoder.cu
        generative/cuda/diffusion_sampler.cu
        PROPERTIES LANGUAGE CUDA
    )
endif()

if(APPLE)
    set_source_files_properties(
        vision/apple/metal_nms.mm
        PROPERTIES LANGUAGE OBJCXX
    )
    target_compile_options(xinfer_postproc PRIVATE -fobjc-arc)
endif()