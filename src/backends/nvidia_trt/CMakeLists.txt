# src/backends/nvidia_trt/CMakeLists.txt

set(BACKEND_NAME "xinfer_backend_nvidia")

# 1. Find CUDA
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# 2. Find TensorRT
# Assumes TensorRT is in system paths or TENSORRT_ROOT env var is set
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} $ENV{TENSORRT_ROOT} /usr/include /usr/local/cuda/include)
find_library(TENSORRT_LIBRARY nvinfer
    HINTS ${TENSORRT_ROOT} $ENV{TENSORRT_ROOT} /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(STATUS "[xInfer] TensorRT not found. Skipping 'nvidia_trt' backend.")
    return()
endif()

message(STATUS "[xInfer] Enabling NVIDIA TensorRT Backend.")

# 3. Add Library
add_library(${BACKEND_NAME} OBJECT
    backend.cpp
)

# 4. Include Directories
target_include_directories(${BACKEND_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${TENSORRT_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# 5. Link Dependencies
target_link_libraries(${BACKEND_NAME} PRIVATE
    xinfer_core
    CUDA::cudart
    ${TENSORRT_LIBRARY}
)

# 6. Definitions
target_compile_definitions(${BACKEND_NAME} PRIVATE
    -DXINFER_ENABLE_TRT
)