# src/postproc/CMakeLists.txt

# --- Define Source Files ---

# 1. CPU / Universal Sources (Always compiled)
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

# 2. CUDA Sources (Only if TRT/CUDA enabled)
if(XINFER_ENABLE_TRT)
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

# 3. Apple Metal Sources (Only on macOS/iOS)
if(APPLE AND XINFER_ENABLE_COREML)
    list(APPEND POSTPROC_SRCS
            vision/apple/metal_nms.mm
    )
endif()

# --- Create Library ---

add_library(xinfer_postproc OBJECT ${POSTPROC_SRCS})

# --- Include Directories ---

target_include_directories(xinfer_postproc PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS} # Required for CPU implementations
)

# --- Compiler Flags ---

if(XINFER_ENABLE_TRT)
    # Ensure CUDA files are compiled with nvcc
    set_source_files_properties(${POSTPROC_SRCS} PROPERTIES LANGUAGE CXX)
    # (CMake automatically handles .cu extension if CUDA language is enabled in root)
endif()

# Apple specific flags
if(APPLE)
    target_compile_options(xinfer_postproc PRIVATE -fobjc-arc)
endif()