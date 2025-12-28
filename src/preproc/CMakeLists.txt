# src/preproc/CMakeLists.txt

# --- 1. Define Universal Sources (CPU Fallbacks) ---
set(PREPROC_SRCS
    factory.cpp
    
    # Image Fallbacks
    image/cpu/opencv_image.cpp
    image/cpu/avx_image.cpp # For x86
    image/cpu/neon_image.cpp # For ARM

    # Audio Fallback
    audio/cpu/cpu_audio.cpp
    audio/cpu/neon_audio.cpp

    # Text Fallback
    text/cpu/bert_tokenizer.cpp
    text/cpu/bpe_tokenizer.cpp
    
    # Tabular Fallback
    tabular/cpu/log_encoder.cpp
    tabular/cpu/ip_utils.cpp
    tabular/cpu/scalers.cpp
    
    # Graph & Video
    graph/cpu/graph_builder.cpp
    video/cpu/frame_stacker.cpp
)

# --- 2. Platform-Specific Sources ---

# A. NVIDIA CUDA
if(XINFER_ENABLE_TRT)
    message(STATUS "[xInfer] Enabling CUDA Preprocessing kernels")
    list(APPEND PREPROC_SRCS
        image/cuda/cuda_image.cu
        audio/cuda/cufft_audio.cu
        text/cuda/cubert_tokenizer.cu
        tabular/cuda/cudf_wrapper.cu
    )
    # Add a flag to be used by factory.cpp
    add_compile_definitions(XINFER_HAS_CUDA)
endif()

# B. Rockchip RGA
# Find the librga library
find_library(LIB_RGA rga HINTS /usr/lib /usr/lib/aarch64-linux-gnu)
if(LIB_RGA)
    message(STATUS "[xInfer] Enabling Rockchip RGA Preprocessing")
    list(APPEND PREPROC_SRCS image/rga/rga_image.cpp)
    add_compile_definitions(XINFER_HAS_RGA)
    target_include_directories(xinfer_preproc PRIVATE /usr/include/rga) # RGA header path
endif()

# C. Apple Metal
if(APPLE AND XINFER_ENABLE_COREML)
    message(STATUS "[xInfer] Enabling Apple Metal Preprocessing kernels")
    list(APPEND PREPROC_SRCS
        image/apple/metal_image.mm
        audio/apple/accelerate_audio.mm
    )
endif()

# --- 3. Create Library ---
add_library(xinfer_preproc OBJECT ${PREPROC_SRCS})

# --- 4. Link Dependencies ---
target_link_libraries(xinfer_preproc PRIVATE
    xinfer_core
    Threads::Threads
    ${OpenCV_LIBS}
    
    # Conditional Links
    $<$<BOOL:${XINFER_ENABLE_TRT}>:CUDA::cudart>
    $<$<BOOL:${LIB_RGA}>:${LIB_RGA}>
    # Apple Frameworks are handled in the root or via target_link_libraries on the final executable
)

# --- 5. Include Directories ---
target_include_directories(xinfer_preproc PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# --- 6. Platform-Specific Flags ---
# Set language for specific files
if(XINFER_ENABLE_TRT)
    set_source_files_properties(
        image/cuda/cuda_image.cu
        audio/cuda/cufft_audio.cu
        text/cuda/cubert_tokenizer.cu
        tabular/cuda/cudf_wrapper.cu
        PROPERTIES LANGUAGE CUDA
    )
endif()

if(APPLE)
    set_source_files_properties(
        image/apple/metal_image.mm
        audio/apple/accelerate_audio.mm
        PROPERTIES LANGUAGE OBJCXX
    )
    target_compile_options(xinfer_preproc PRIVATE -fobjc-arc)
endif()

# Enable SIMD instructions where available
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set_source_files_properties(image/cpu/neon_image.cpp audio/cpu/neon_audio.cpp
        PROPERTIES COMPILE_FLAGS "-O3 -march=native")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    # AVX2 should be enabled by default on modern x86, but this ensures it
    set_source_files_properties(image/cpu/avx_image.cpp
        PROPERTIES COMPILE_FLAGS "-O3 -mavx2 -mfma")
endif()