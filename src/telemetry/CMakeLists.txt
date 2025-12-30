# src/telemetry/CMakeLists.txt

# Find sources
file(GLOB_RECURSE TELEMETRY_SRCS "*.cpp")

# Create Library
add_library(xinfer_telemetry OBJECT ${TELEMETRY_SRCS})

# Dependencies
target_link_libraries(xinfer_telemetry PRIVATE
        xinfer_core
)

# Include Paths
target_include_directories(xinfer_telemetry PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/third_party # For json.hpp (Metric export)
)

# Platform specific links (e.g., NVIDIA Management Library for GPU Stats)
if(XINFER_ENABLE_TRT)
    # find_library(NVML_LIB nvidia-ml)
    # target_link_libraries(xinfer_telemetry PRIVATE ${NVML_LIB})
    target_compile_definitions(xinfer_telemetry PRIVATE XINFER_HAS_GPU_MONITORING)
endif()